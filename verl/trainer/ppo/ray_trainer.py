# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(self, data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    
    # Read variance threshold from environment variable for GRPO filtering
    variance_threshold = float(os.environ.get('GRPO_VARIANCE_THRESHOLD', 0.0))
    
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards'] #[batch_size, max_length], 仍然是一个独热列表, 每个样本最多一个点不为0, 即确定回答的长度
        index = data.non_tensor_batch['uid'] # [batch_size]
        responses = data.batch['responses'] #[batch_size, max_length]
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask'] # [batch_size, 2*max_length]
        response_mask = attention_mask[:, -response_length:]  #[batch_size, max_length], 这里回答长度内为1, 其余为0, 和token_level_rewards对应
        strategy = os.environ.get('STRATEGY', None)
        if strategy == 'grpo':
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        variance_threshold=variance_threshold)#回答长度内为adv, 其余为0, 和token_level_rewards对应
        elif strategy == 'fspo':
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        variance_threshold=variance_threshold)
            # ============ start of step-wise GRPO  ============
            sentence_mask = data.batch['sentence_mask']
            flip_mask = (sentence_mask * advantages) >= 0.0
            advantages = torch.where(flip_mask, advantages, -advantages)
        elif strategy == 'fspo_evar':
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        variance_threshold=variance_threshold)
            
            sentence_mask = data.batch['sentence_mask']  # [bs, L]
            device = advantages.device
            #assert (advantages[0] != 0).nonzero()[-1].item() == (sentence_mask[0] != 0).nonzero()[-1].item() +4
            # 1) 形状/掩码对齐：仅在回答区间内起效，其余置 0
            assert sentence_mask.shape == advantages.shape, \
                f"sentence_mask.shape {sentence_mask.shape} 与 advantages.shape {advantages.shape} 不一致"

            sentence_mask = sentence_mask.to(dtype=advantages.dtype, device=device)
            # 直接用mask==1
            sentence_mask = sentence_mask * response_mask.to(sentence_mask.dtype)

            flip_mask = (sentence_mask * advantages) >= 0.0
            advantages = torch.where(flip_mask, advantages, -advantages)

        elif strategy == 'grpo_evar_rewardshaping':
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
                variance_threshold=variance_threshold
            )  
            sentence_mask = data.batch['sentence_mask']  # [bs, L]
            device = advantages.device
            #assert (advantages[0] != 0).nonzero()[-1].item() == (sentence_mask[0] != 0).nonzero()[-1].item() +4
            # 1) 形状/掩码对齐：仅在回答区间内起效，其余置 0
            assert sentence_mask.shape == advantages.shape, \
                f"sentence_mask.shape {sentence_mask.shape} 与 advantages.shape {advantages.shape} 不一致"

            sentence_mask = sentence_mask.to(dtype=advantages.dtype, device=device)
            #将回答部分替换成advantages
            two_mask = (sentence_mask == 2)
            if two_mask.any():
                first_adv = advantages[:, 0].unsqueeze(1)  # [bs, 1]
                sentence_mask = torch.where(two_mask, first_adv, sentence_mask)
            sentence_mask = sentence_mask * response_mask.to(sentence_mask.dtype)  # 屏蔽非回答 token
            
            # 4) 获取 PBRS 参数
            alpha = float(os.environ.get('PBRS_ALPHA', 0.1))  # 缩放系数，建议 0.01-0.1
            smoothing_window = int(os.environ.get('PBRS_SMOOTH_WINDOW', 3))  # 平滑窗口
            clip_value = float(os.environ.get('PBRS_CLIP', 2.0))  # 裁剪值
            gamma = float(os.environ.get('PBRS_GAMMA', 1.0))  # 通常与 GAE 的 gamma 一致
            
            # 5) 计算 PBRS shaping reward: F_t = gamma * phi(s_{t+1}) - phi(s_t)
            shaping_rewards = core_algos.compute_pbrs_shaping_reward(
                sentence_mask=sentence_mask,
                response_mask=response_mask,
                gamma=gamma,
                alpha=alpha,
                smoothing_window=smoothing_window,
                clip_value=clip_value
            )
            
            # 6) 将 shaping rewards 加到 token_level_rewards 上
            # R'_t = R_t + F_t
            token_level_rewards = token_level_rewards + shaping_rewards
            # 7) 用修改后的 rewards 计算 advantages（理论保证不改变最优策略）
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
                variance_threshold=variance_threshold
            )    

        elif strategy == 'grpo_evar_math_weighted':
            """
            乘性权重重分配方法
            
            数学原理：
            对于GRPO的outcome-based advantage A（所有token相同），通过sentence_mask s_i
            重新分配权重到各个token。
            
            权重计算（指数分配，类似softmax）：
              当 A > 0: w_i = exp(λ_pos * s_i) / Σ exp(λ_pos * s_j)
              当 A ≤ 0: w_i = exp(-λ_neg * s_i) / Σ exp(-λ_neg * s_j)
            
            新advantage：
              A_i' = A * w_i * N  （N是有效token数，保持总梯度流）
            
            效果：
            - A>0时，s_i=1的token权重最大（强化好步骤）
            - A≤0时，s_i=-1的token权重最大（强化坏步骤的惩罚）
            
            数学性质：
            1. 梯度守恒：Σ A_i' = A * N
            2. 可导性：exp函数处处可导
            3. 可控性：λ控制权重集中程度
            """               
            # 先计算基础的 GRPO outcome advantages，作为后续重分配的基础
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
                variance_threshold=variance_threshold
            )
            # 初始化 sentence_mask 与 device
            sentence_mask = data.batch['sentence_mask']  # [bs, L]
            device = advantages.device
            assert sentence_mask.shape == advantages.shape, \
                f"sentence_mask.shape {sentence_mask.shape} 与 advantages.shape {advantages.shape} 不一致"
            sentence_mask = sentence_mask.to(dtype=advantages.dtype, device=device)
            sentence_mask = sentence_mask * response_mask.to(sentence_mask.dtype)

            from collections import defaultdict
            uid_list = list(index)
            group_map = defaultdict(list)
            for i, uid in enumerate(uid_list):
                group_map[uid].append(i)
            
            
            weighted_advantages = advantages.clone()
            
            # 温度系数（控制权重集中程度）
            lambda_pos = float(os.environ.get('SENTENCE_LAMBDA_POS', 2.0))
            lambda_neg = float(os.environ.get('SENTENCE_LAMBDA_NEG', 2.0))
            
            
            # 用于统计分析的变量
            pos_samples = 0
            neg_samples = 0
            weight_concentration_stats = []
            
            for group_idx, (uid, row_indices) in enumerate(group_map.items()):
                rows = torch.tensor(row_indices, device=device, dtype=torch.long)
                group_adv = advantages.index_select(dim=0, index=rows)  # [N, L]
                group_sentence = sentence_mask.index_select(dim=0, index=rows)  # [N, L]
                group_mask = response_mask.index_select(dim=0, index=rows)  # [N, L]
                
                # 对每个样本分别处理（因为每个样本的advantage可能方向不同）
                group_result = torch.zeros_like(group_adv)
                
                for i in range(group_adv.shape[0]):
                    sample_adv = group_adv[i]  # [L]
                    sample_sentence = group_sentence[i]  # [L]
                    sample_mask = group_mask[i]  # [L]
                    
                    # 取第一个有效token的advantage值（GRPO中所有token相同）
                    valid_indices = sample_mask.nonzero(as_tuple=True)[0]
                    if len(valid_indices) == 0:
                        continue
                    
                    A_scalar = sample_adv[valid_indices[0]].item()  # 标量advantage
                    n_tokens = valid_indices.shape[0]  # 有效token数
                    
                    # 统计正负样本
                    if A_scalar > 0:
                        pos_samples += 1
                        # 正样本：强化好步骤（s=1权重最大）
                        logits = lambda_pos * sample_sentence  # [L]
                    else:
                        neg_samples += 1
                        # 负样本：强化坏步骤（s=-1权重最大）
                        logits = -lambda_neg * sample_sentence  # [L]，注意负号
                    
                    # 只在有效token上计算softmax
                    logits_masked = logits.clone()
                    logits_masked[~sample_mask.bool()] = -1e9  # mask掉无效位置
                    
                    # Softmax权重
                    weights = torch.softmax(logits_masked, dim=0)  # [L]
                    
                    # 统计权重集中度（前5个token的权重和）
                    valid_weights = weights[valid_indices]
                    top_weights_sum = torch.topk(valid_weights, min(5, len(valid_weights))).values.sum().item()
                    weight_concentration_stats.append(top_weights_sum)
                    
                    # 重分配advantage，保持总梯度流
                    # A_i' = A * w_i * N
                    sample_result = A_scalar * weights * n_tokens
                    
                    # 只在有效位置保留结果
                    sample_result = sample_result * sample_mask.to(sample_result.dtype)
                    group_result[i] = sample_result
                
                weighted_advantages.index_copy_(dim=0, index=rows, source=group_result)

            advantages = weighted_advantages
            returns = weighted_advantages
            
            # Final weighted advantages stats (debug prints removed)

        elif strategy == 'grpo_evar_math_scaled':
            """
            加性混合方法
            
            数学原理：
            将outcome-based advantage和sentence-based advantage线性组合
            
            A_final = α * A_outcome + β * A_sentence
            
            其中：
            - A_outcome: GRPO的outcome advantage（组内标准化）
              A_outcome_i^(g) = (R_i - μ_g) / (σ_g + ε)
            
            - A_sentence: 从sentence_mask构造的advantage（组内标准化）
              A_sentence_i^(g) = (s_i - μ_g^s) / (σ_g^s + ε)
            
            两者都在组内标准化，确保：
            1. 保持GRPO的相对比较特性（组内比较）
            2. 两种advantage量级一致（均值0，方差1）
            3. token间有相对差异（A_sentence引入token级信号）
            
            数学性质：
            1. 量级一致：避免一方主导
            2. 相对比较：保持GRPO核心
            3. Token差异：引入过程监督
            4. 可控性：α/β调节outcome vs process权重
            
            推荐配置：
            - 平衡型：α=0.7, β=0.3（outcome为主，sentence辅助）
            - 激进型：α=0.5, β=0.5（outcome和sentence同等重要）
            """
            # 先计算基础的 GRPO outcome advantages，作为加性混合的 A_outcome
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
                variance_threshold=variance_threshold
            )
            # 初始化 sentence_mask、device，并明确 outcome 优势
            sentence_mask = data.batch['sentence_mask']  # [bs, L]
            device = advantages.device
            assert sentence_mask.shape == advantages.shape, \
                f"sentence_mask.shape {sentence_mask.shape} 与 advantages.shape {advantages.shape} 不一致"
            sentence_mask = sentence_mask.to(dtype=advantages.dtype, device=device)
            sentence_mask = sentence_mask * response_mask.to(sentence_mask.dtype)
            advantages_outcome = advantages
            
            
            # 2. 统一尺度的 sentence-based advantage（全局EMA统计，暖启动后冻结），线性混合后不再组内二次标准化
            # 超参（通过环境变量配置）
            import math
            alpha = float(os.environ.get('OUTCOME_WEIGHT', 0.8))          # outcome 权重
            beta = float(os.environ.get('SENTENCE_WEIGHT', 0.2))          # sentence 权重
            mix_warmup_steps = int(os.environ.get('MIX_WARMUP_STEPS', 100))
            sent_stats_warmup_steps = int(os.environ.get('SENTENCE_STATS_WARMUP_STEPS', 100))
            sent_ema_momentum = float(os.environ.get('SENTENCE_EMA_MOMENTUM', 0.99))
            sent_clamp = float(os.environ.get('SENTENCE_CLAMP', 2.5))     # z-score 裁剪阈值
            adv_target_std = float(os.environ.get('ADV_FINAL_TARGET_STD', 1.0))  # 最终adv目标方差（<=0则不缩放）
            adv_std_ema_momentum = float(os.environ.get('ADV_STD_EMA_MOMENTUM', 0.99))
            eps = 1e-8
            
            # 归一化混合权重到和为1（固定风格由alpha直接控制）
            tw = alpha + beta
            if tw > eps:
                alpha = alpha / tw
                beta = beta / tw
            
            # 获取当前训练步数
            try:
                current_step = int(getattr(self, 'global_step', 0))
            except Exception:
                current_step = 0
            
            # 初始化/更新 sentence 全局EMA统计，仅在前 sent_stats_warmup_steps 内更新，随后冻结
            valid_mask = response_mask.bool()
            valid_sentence_vals = sentence_mask[valid_mask]
            if not hasattr(self, '_sent_ema_mean'):
                self._sent_ema_mean = torch.zeros(1, device=device, dtype=advantages.dtype)
                self._sent_ema_var = torch.ones(1, device=device, dtype=advantages.dtype)
            if valid_sentence_vals.numel() > 0 and current_step < sent_stats_warmup_steps:
                b_mean = valid_sentence_vals.mean()
                b_var = valid_sentence_vals.var(unbiased=False)
                self._sent_ema_mean = self._sent_ema_mean * sent_ema_momentum + (1.0 - sent_ema_momentum) * b_mean
                self._sent_ema_var = self._sent_ema_var * sent_ema_momentum + (1.0 - sent_ema_momentum) * b_var
            s_mean = self._sent_ema_mean
            s_std = torch.sqrt(torch.clamp(self._sent_ema_var, min=0.0)) + eps
            
            advantages_sentence = (sentence_mask - s_mean) / s_std
            if sent_clamp > 0.0:
                advantages_sentence = torch.clamp(advantages_sentence, -sent_clamp, sent_clamp)
            advantages_sentence = advantages_sentence * response_mask.to(advantages_sentence.dtype)
            
            # Sentence 权重在前 mix_warmup_steps 内线性升温，其后保持固定，确保风格全程一致
            if mix_warmup_steps > 0:
                mix_scale = min(1.0, float(current_step) / float(mix_warmup_steps))
            else:
                mix_scale = 1.0
            beta_eff = beta * mix_scale
            alpha_eff = 1.0 - beta_eff
            
            advantages_final = alpha_eff * advantages_outcome + beta_eff * advantages_sentence
            advantages_final = advantages_final * response_mask.to(advantages_final.dtype)
            
            # 最终adv固定尺度：使用EMA估计整体std，暖启动内更新，之后冻结；若adv_target_std<=0则不缩放
            if not hasattr(self, '_adv_std_ema'):
                self._adv_std_ema = torch.tensor(1.0, device=device, dtype=advantages.dtype)
                self._adv_scale = torch.tensor(1.0, device=device, dtype=advantages.dtype)
            if adv_target_std > 0.0:
                valid_final = advantages_final[valid_mask]
                if valid_final.numel() > 0 and current_step < sent_stats_warmup_steps:
                    b_std = valid_final.std(unbiased=False)
                    self._adv_std_ema = self._adv_std_ema * adv_std_ema_momentum + (1.0 - adv_std_ema_momentum) * b_std
                    self._adv_scale = torch.tensor(adv_target_std, device=device, dtype=advantages.dtype) / (self._adv_std_ema + eps)
            advantages = advantages_final * (self._adv_scale if adv_target_std > 0.0 else 1.0)
            returns = advantages
            

        elif strategy == 'grpo_evar_token_scaled':
            """
            Token级Reward调制方法（类似PRM）
            
            数学原理：
            两阶段设计，确保outcome为主导，sentence_mask用于token级差异
            
            阶段1：Token级Reward调制
              R_i^token = R_outcome * (1 + β * ŝ_i)
              
              其中：
              - R_outcome: 样本的最终答案得分（标量，主导因子）
              - ŝ_i: 组内标准化的sentence_mask（调制因子）
                ŝ_i = (s_i - μ_g^s) / (σ_g^s + ε)
              - β: 调制强度系数
              
            阶段2：组内标准化得到Advantage
              A_i^(g) = (R_i^token - μ_g^R) / (σ_g^R + ε)
            
            效果：
            - Outcome决定基础reward大小（主导）
            - Sentence_mask在sentence内部分配reward份额（辅助）
            - 好的step获得更多份额，坏的step获得更少
            - 类似PRM，但保持GRPO的组内比较特性
            
            数学性质：
            1. Outcome主导：乘性调制，outcome为基础
            2. Token级差异：ŝ_i引入token间的相对质量
            3. 保持GRPO：最终仍组内标准化
            4. 可控性：β控制sentence_mask的影响强度
            
            推荐配置：
            - 保守型：β=0.2（outcome为主，sentence辅助）
            - 平衡型：β=0.3（推荐）
            - 激进型：β=0.5（sentence影响较大）
            """
            device = token_level_rewards.device
            sentence_mask = data.batch['sentence_mask']
            sentence_mask = sentence_mask.to(dtype=token_level_rewards.dtype, device=device)
            sentence_mask = sentence_mask * response_mask.to(sentence_mask.dtype)
            
            # 处理sentence_mask=2的情况
            two_mask = (sentence_mask == 2)
            if two_mask.any():
                answer_scores = []
                for i in range(token_level_rewards.shape[0]):
                    valid_positions = response_mask[i].nonzero(as_tuple=True)[0]
                    if len(valid_positions) > 0:
                        last_pos = valid_positions[-1]
                        answer_score = token_level_rewards[i, last_pos]
                    else:
                        answer_score = 0.0
                    answer_scores.append(answer_score)
                
                answer_scores_tensor = torch.tensor(answer_scores, device=device, 
                                                   dtype=sentence_mask.dtype).unsqueeze(1)
                sentence_mask = torch.where(two_mask, answer_scores_tensor, sentence_mask)
            
            # ===== 阶段1：提取outcome reward（每个样本的标量结果得分）=====
            batch_size = token_level_rewards.shape[0]
            outcome_rewards = []
            
            for i in range(batch_size):
                valid_positions = response_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    # 取最后一个有效token的reward（GRPO中这就是outcome reward）
                    last_pos = valid_positions[-1]
                    outcome_reward = token_level_rewards[i, last_pos].item()
                else:
                    outcome_reward = 0.0
                outcome_rewards.append(outcome_reward)
            
            outcome_rewards_tensor = torch.tensor(outcome_rewards, device=device, 
                                                 dtype=token_level_rewards.dtype)
            
            # ===== 阶段2：标准化sentence_mask（组内token级标准化）=====
            from collections import defaultdict
            uid_list = list(index)
            group_map = defaultdict(list)
            for i, uid in enumerate(uid_list):
                group_map[uid].append(i)
            
            sentence_normalized = torch.zeros_like(sentence_mask)
            
            for uid, row_indices in group_map.items():
                rows = torch.tensor(row_indices, device=device, dtype=torch.long)
                group_sentence = sentence_mask.index_select(dim=0, index=rows)  # [N, L]
                group_mask = response_mask.index_select(dim=0, index=rows)  # [N, L]
                
                # 只在有效token上计算统计量（先mask再统计）
                valid_sentence = group_sentence[group_mask.bool()]  # [n_valid_tokens]
                
                if valid_sentence.numel() > 1:
                    # 组内标准化：零均值，单位方差
                    s_mean = valid_sentence.mean()
                    # 标准化整个组
                    group_normalized = group_sentence - s_mean
                    
                    # 应用mask
                    group_normalized = group_normalized * group_mask.to(group_normalized.dtype)
                    
                    sentence_normalized.index_copy_(dim=0, index=rows, source=group_normalized)
                elif valid_sentence.numel() == 1:
                    # 只有1个token，无法标准化，设为0
                    sentence_normalized.index_copy_(dim=0, index=rows, 
                                                   source=torch.zeros_like(group_sentence))
            
            # ===== 阶段3：Token级Reward调制（关键创新）=====
            # R_i^token = R_outcome * (1 + β * ŝ_i)
            beta = float(os.environ.get('SENTENCE_BETA', 0.3))
            
            # 将outcome_rewards广播到所有token
            outcome_broadcast = outcome_rewards_tensor.unsqueeze(1).expand_as(sentence_mask)
            
            # 乘性调制：outcome为基础，sentence_mask调整份额
            modulated_rewards = outcome_broadcast * (1.0 + beta * sentence_normalized)
            
            # 应用mask（确保padding位置为0）
            modulated_rewards = modulated_rewards * response_mask.to(modulated_rewards.dtype)
            
            # ===== 阶段4：组内标准化得到Advantage（保持GRPO特性）=====
            advantages = torch.zeros_like(modulated_rewards)
            
            for uid, row_indices in group_map.items():
                rows = torch.tensor(row_indices, device=device, dtype=torch.long)
                group_rewards = modulated_rewards.index_select(dim=0, index=rows)  # [N, L]
                group_mask = response_mask.index_select(dim=0, index=rows)  # [N, L]
                
                # 只在有效token上计算统计量
                valid_rewards = group_rewards[group_mask.bool()]  # [n_valid_tokens]
                
                if valid_rewards.numel() > 1:
                    # 组内标准化
                    r_mean = valid_rewards.mean()
                    r_std = valid_rewards.std(unbiased=False)
                    
                    group_adv = group_rewards - r_mean
                    
                    # 应用mask
                    group_adv = group_adv * group_mask.to(group_adv.dtype)
                    
                    advantages.index_copy_(dim=0, index=rows, source=group_adv)
                elif valid_rewards.numel() == 1:
                    # 只有1个token，设为0
                    advantages.index_copy_(dim=0, index=rows, 
                                         source=torch.zeros_like(group_rewards))
            
            # 最终确保padding位置为0（双重保险）
            advantages = advantages * response_mask.to(advantages.dtype)
            returns = advantages
        
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_reward_metrics(batch):
    reward_tensor = batch.batch['token_level_scores'].sum(-1)
    format_tensor = batch.batch['format_scores']
    answer_tensor = batch.batch['answer_scores']

    reward_metrics = dict()

    reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()

    # Calculate format_error ratio (value == -3)
    format_error = torch.sum(format_tensor == -2).float() / format_tensor.numel()
    reward_metrics["reward/format_error_ratio"] = format_error.detach().item()

    # Calculate average answer score
    reward_metrics["reward/answer_score"] = torch.mean(answer_tensor).detach().item()

    # Calculate average reasoning score
    if 'reasoning_scores' in batch.batch:
        reasoning_tensor = batch.batch['reasoning_scores']
        reward_metrics["reward/reasoning_score"] = torch.mean(reasoning_tensor).detach().item()

    return reward_metrics

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Format anchoring support (定期格式校准)
        self.format_anchor = None

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'remax':
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        from torch.utils.data import Subset
        # TODO: we have to make sure the batch size is divisible by the dp size
        # Read env to optionally disable unanswerable samples in TRAIN ONLY
        try:
            _env_disable_unans = os.environ.get('disable_unanswerable', 'false').strip().lower() in ('1', 'true', 'yes')
        except Exception:
            _env_disable_unans = False

        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=self.config.data.filter_overlong_prompts,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         disable_unanswerable=_env_disable_unans)
        # Optional: subset training data for flexible debugging
        start_index = int(self.config.data.get('start_index', 0) or 0)
        max_samples = self.config.data.get('max_samples', None)
        if max_samples is not None:
            try:
                max_samples = int(max_samples)
            except Exception:
                max_samples = None
        if start_index > 0 or (max_samples is not None and max_samples >= 0):
            dataset_length = len(self.train_dataset)
            start_index = max(0, min(start_index, dataset_length))
            end_index = dataset_length if max_samples is None or max_samples < 0 else min(dataset_length, start_index + max_samples)
            if end_index > start_index:
                indices = list(range(start_index, end_index))
                self.train_dataset = Subset(self.train_dataset, indices)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           num_workers=8,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=self.config.data.filter_overlong_prompts,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       disable_unanswerable=False)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         num_workers=8,
                                         shuffle=False,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self, log_sample=False, global_step=0):
        import os  # 确保在方法内部能访问到os模块
        answer_tensor_lst = []
        reward_tensor_lst = []
        base_reward_tensor_lst = []  # NEW: Store original judge results (without penalty)
        data_source_lst = []
        answerable_flags_lst = []  # NEW: Store answerable flags for each sample

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # Extract and store answerable flags for each sample in this batch
            batch_answerable_flags = []
            try:
                non_tensor = test_batch.non_tensor_batch
                # Safely get extra_info or extra_infos (avoid numpy array truth value error)
                extra_infos = non_tensor.get('extra_info', None)
                if extra_infos is None:
                    extra_infos = non_tensor.get('extra_infos', None)
                answerable_false_count = 0
                answerable_true_count = 0
                unknown_count = 0
                if isinstance(extra_infos, list):
                    for ei in extra_infos:
                        if isinstance(ei, dict):
                            flag = ei.get('answerable', None)
                            if flag is False:
                                answerable_false_count += 1
                                batch_answerable_flags.append(False)
                            elif flag is True:
                                answerable_true_count += 1
                                batch_answerable_flags.append(True)
                            else:
                                unknown_count += 1
                                batch_answerable_flags.append(None)
                        else:
                            batch_answerable_flags.append(None)
                else:
                    # If extra_infos is not a list, create None flags for all samples
                    batch_size = test_batch.batch['input_ids'].shape[0]
                    batch_answerable_flags = [None] * batch_size
            except Exception as e:
                # On error, create None flags
                batch_size = test_batch.batch['input_ids'].shape[0]
                batch_answerable_flags = [None] * batch_size
            
            answerable_flags_lst.extend(batch_answerable_flags)
            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            if log_sample:
                # Log the first sample of the first batch
                try:
                    # Decode prompt, handling padding correctly
                    prompt_ids = test_batch.batch['input_ids'][0]
                    prompt_mask = test_batch.batch['attention_mask'][0]
                    valid_prompt_ids = prompt_ids[prompt_mask.bool()]
                    prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

                    # Decode and clean response
                    response_ids = test_output_gen_batch.batch['responses'][0]
                    response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    cleaned_response = re.sub(r'\n{2,}', '\n', response_str).strip()

                    header = f" Initial Validation Sample " if global_step == 0 else f" Validation Sample @ Step {global_step} "
                    print("\n" + "*" * 80)
                    print(header.center(80, '*'))
                    print("--- PROMPT ---")
                    print(prompt_str)
                    print("\n--- RESPONSE (cleaned) ---")
                    print(cleaned_response)
                    print("*" * 80 + "\n")
                except Exception as e:
                    print(f"Failed to log validation sample: {e}")
                log_sample = False  # Only log one sample per validation run

            print('validation generation end AND validation evaluation start')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            # evaluate using reward_function
            reward_tensor, format_tensor, answer_tensor, base_reward_tensor = self.val_reward_fn(test_batch)
            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            answer_tensor_lst.append(answer_tensor)
            reward_tensor_lst.append(reward_tensor)
            base_reward_tensor_lst.append(base_reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        answer_tensor = torch.cat(answer_tensor_lst, dim=0).cpu()
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        base_reward_tensor = torch.cat(base_reward_tensor_lst, dim=0).cpu()  # Original judge results
        data_sources = np.concatenate(data_source_lst, axis=0)
        answerable_flags = np.array(answerable_flags_lst, dtype=object)  # Convert to numpy array

        # evaluate test_score based on data source
        data_source_answer_reward = {}
        data_source_reward = {}
        for i in range(answer_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_answer_reward:
                data_source_answer_reward[data_source] = []

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []

            data_source_answer_reward[data_source].append(answer_tensor[i].item())
            data_source_reward[data_source].append(reward_tensor[i].item())

        # compute n_correct / n_miss / hallucination / score per data_source
        # NEW: Use base_reward (original judge result) for test_score calculation
        # This ensures evaluation metrics are independent of training penalties
        # alignment: sample_outputs[i] corresponds to base_reward_tensor[i] and data_sources[i]
        # IMPORTANT: enforce mutual exclusivity based on mapped answer score {-1, 0, 1}
        counts = {}
        counts_answerable_true = {}  # NEW: Counts for answerable=True samples
        counts_answerable_false = {}  # NEW: Counts for answerable=False samples
        
        total_samples = len(sample_outputs)
        for i in range(len(data_sources)):
            ds = data_sources[i]
            answerable_flag = answerable_flags[i] if i < len(answerable_flags) else None
            
            # Overall counts
            if ds not in counts:
                counts[ds] = {"n": 0, "n_correct": 0, "n_miss": 0, "n_incorrect": 0}
            counts[ds]["n"] += 1

            # Use base_reward (without length penalty) for test_score calculation
            base_val = float(base_reward_tensor[i].item())
            if base_val >= 0.999:
                # correct
                counts[ds]["n_correct"] += 1
            elif base_val <= -0.999:
                # incorrect / hallucination
                counts[ds]["n_incorrect"] += 1
            else:
                # mapped as 0 → IDK/miss
                counts[ds]["n_miss"] += 1
            
            # Separate counts by answerable flag
            if answerable_flag is True:
                if ds not in counts_answerable_true:
                    counts_answerable_true[ds] = {"n": 0, "n_correct": 0, "n_miss": 0, "n_incorrect": 0}
                counts_answerable_true[ds]["n"] += 1
                if base_val >= 0.999:
                    counts_answerable_true[ds]["n_correct"] += 1
                elif base_val <= -0.999:
                    counts_answerable_true[ds]["n_incorrect"] += 1
                else:
                    counts_answerable_true[ds]["n_miss"] += 1
                    
            elif answerable_flag is False:
                if ds not in counts_answerable_false:
                    counts_answerable_false[ds] = {"n": 0, "n_correct": 0, "n_miss": 0, "n_incorrect": 0}
                counts_answerable_false[ds]["n"] += 1
                if base_val >= 0.999:
                    counts_answerable_false[ds]["n_correct"] += 1
                elif base_val <= -0.999:
                    counts_answerable_false[ds]["n_incorrect"] += 1
                else:
                    counts_answerable_false[ds]["n_miss"] += 1

        metric_dict = {}
        for data_source in data_source_answer_reward.keys():
            # answer_reward = data_source_answer_reward[data_source]
            # Removed duplicate test_answer_score as it's redundant with test_score
            
            # derive ratios for overall counts
            c = counts.get(data_source, {"n": 0, "n_correct": 0, "n_miss": 0})
            n = c["n"]
            if n <= 0:
                n_correct_ratio = 0.0
                n_miss_ratio = 0.0
                hallucination_ratio = 0.0
                final_score = 0.0
            else:
                n_correct_ratio = c["n_correct"] / n
                n_miss_ratio = c["n_miss"] / n
                # with mutual exclusivity, hallucination == incorrect ratio
                hallucination_ratio = c.get("n_incorrect", 0) / n
                final_score = 2.0 * n_correct_ratio + n_miss_ratio - 1.0

            metric_dict[f'val/test_n_correct/{data_source}'] = n_correct_ratio
            metric_dict[f'val/test_n_miss/{data_source}'] = n_miss_ratio
            metric_dict[f'val/test_hallucination/{data_source}'] = hallucination_ratio
            metric_dict[f'val/test_score/{data_source}'] = final_score

            # NEW: Compute and log metrics for answerable=True samples
            c_true = counts_answerable_true.get(data_source, {"n": 0, "n_correct": 0, "n_miss": 0, "n_incorrect": 0})
            n_true = c_true["n"]
            if n_true > 0:
                metric_dict[f'val/test_n_correct_answerable_true/{data_source}'] = c_true["n_correct"] / n_true
                metric_dict[f'val/test_n_miss_answerable_true/{data_source}'] = c_true["n_miss"] / n_true
                metric_dict[f'val/test_hallucination_answerable_true/{data_source}'] = c_true.get("n_incorrect", 0) / n_true
                # For answerable=True: correct should be high, miss should be low
                score_true = 2.0 * (c_true["n_correct"] / n_true) + (c_true["n_miss"] / n_true) - 1.0
                metric_dict[f'val/test_score_answerable_true/{data_source}'] = score_true

            # NEW: Compute and log metrics for answerable=False samples
            c_false = counts_answerable_false.get(data_source, {"n": 0, "n_correct": 0, "n_miss": 0, "n_incorrect": 0})
            n_false = c_false["n"]
            if n_false > 0:
                metric_dict[f'val/test_n_correct_answerable_false/{data_source}'] = c_false["n_correct"] / n_false
                metric_dict[f'val/test_n_miss_answerable_false/{data_source}'] = c_false["n_miss"] / n_false
                metric_dict[f'val/test_hallucination_answerable_false/{data_source}'] = c_false.get("n_incorrect", 0) / n_false
                # For answerable=False: correct (IDK) should be high, incorrect should be low
                score_false = 2.0 * (c_false["n_correct"] / n_false) + (c_false["n_miss"] / n_false) - 1.0
                metric_dict[f'val/test_score_answerable_false/{data_source}'] = score_false

        # Calculate and store THS (Truthful Helpfulness Score)
        # For initial validation (global_step=0), store x0 and y0
        if global_step == 0:
            # Compute average correct rate (x0) and hallucination rate (y0) across all data sources
            x0_values = [metric_dict.get(f'val/test_n_correct/{ds}', 0.0) for ds in counts.keys()]
            y0_values = [metric_dict.get(f'val/test_hallucination/{ds}', 0.0) for ds in counts.keys()]
            
            if x0_values and y0_values:
                x0 = sum(x0_values) / len(x0_values)  # Average correct rate
                y0 = sum(y0_values) / len(y0_values)  # Average hallucination rate
                self.initial_validation_metrics = {'x0': x0, 'y0': y0}
                print(f"\n🔍 Initial Validation Metrics stored for THS calculation:")
                print(f"   x0 (initial correct rate): {x0:.4f}")
                print(f"   y0 (initial hallucination rate): {y0:.4f}")
        else:
            # For subsequent validations, calculate THS
            if self.initial_validation_metrics is not None:
                x0 = self.initial_validation_metrics['x0']
                y0 = self.initial_validation_metrics['y0']
                
                # Get current correct rate (x1) and hallucination rate (y1)
                x1_values = [metric_dict.get(f'val/test_n_correct/{ds}', 0.0) for ds in counts.keys()]
                y1_values = [metric_dict.get(f'val/test_hallucination/{ds}', 0.0) for ds in counts.keys()]
                
                if x1_values and y1_values:
                    x1 = sum(x1_values) / len(x1_values)  # Average current correct rate
                    y1 = sum(y1_values) / len(y1_values)  # Average current hallucination rate
                    
                    # Calculate THS = (x1*y0 - y1*x0) / y0
                    if y0 > 0:  # Avoid division by zero
                        ths = (x1 * y0 - y1 * x0) / y0
                        metric_dict['val/ths'] = ths
                        print(f"\n📊 Truthful Helpfulness Score (THS): {ths:.4f}")
                        print(f"   x0={x0:.4f}, y0={y0:.4f}, x1={x1:.4f}, y1={y1:.4f}")
                    else:
                        print("\n⚠️ Cannot calculate THS: initial hallucination rate (y0) is zero")
                        metric_dict['val/ths'] = 0.0
            else:
                print("\n⚠️ Cannot calculate THS: initial validation metrics not available")
                metric_dict['val/ths'] = 0.0
        
        # Save validation test_score for dynamic IDK penalty
        # Use test_score (not just n_correct) as it considers both correct and miss
        import os
        enable_dynamic_idk = os.environ.get('ENABLE_DYNAMIC_IDK_PENALTY', 'false').lower() == 'true'
        if enable_dynamic_idk:
            # Calculate average test_score across all data sources
            test_scores = [metric_dict.get(f'val/test_score/{ds}', 0.0) for ds in counts.keys()]
            if test_scores:
                avg_test_score = sum(test_scores) / len(test_scores)
                # Save to a file in the checkpoint directory
                state_file = os.path.join(self.config.trainer.default_local_dir, 'dynamic_idk_state.txt')
                os.makedirs(os.path.dirname(state_file), exist_ok=True)
                with open(state_file, 'w') as f:
                    f.write(f"{avg_test_score:.6f}\n")
                print(f"\n🎯 [Dynamic IDK] Updated test_score: {avg_test_score:.4f}")
                print(f"   Saved to: {state_file}")
        
        # Enhanced print with detailed breakdowns by answerable flag
        try:
            print("\n" + "=" * 80)
            print(" Validation Results Summary ".center(80, "="))
            print("=" * 80)
            
            for ds, c in counts.items():
                print(f"\n[{ds}] Overall: n={c['n']}, correct={c['n_correct']}, miss={c['n_miss']}, incorrect={c.get('n_incorrect', 0)}")
                
                # Print answerable=True stats (all three categories are valid)
                c_true = counts_answerable_true.get(ds, {"n": 0, "n_correct": 0, "n_miss": 0, "n_incorrect": 0})
                if c_true["n"] > 0:
                    correct_pct = 100.0 * c_true["n_correct"] / c_true["n"]
                    miss_pct = 100.0 * c_true["n_miss"] / c_true["n"]
                    incorrect_pct = 100.0 * c_true.get("n_incorrect", 0) / c_true["n"]
                    print(f"  └─ [answerable=True]  n={c_true['n']}, correct={c_true['n_correct']} ({correct_pct:.1f}%), "
                          f"miss={c_true['n_miss']} ({miss_pct:.1f}%), incorrect={c_true.get('n_incorrect', 0)} ({incorrect_pct:.1f}%)")
                
                # Print answerable=False stats (only correct/incorrect should exist)
                c_false = counts_answerable_false.get(ds, {"n": 0, "n_correct": 0, "n_miss": 0, "n_incorrect": 0})
                if c_false["n"] > 0:
                    correct_pct = 100.0 * c_false["n_correct"] / c_false["n"]
                    incorrect_pct = 100.0 * c_false.get("n_incorrect", 0) / c_false["n"]
                    # For answerable=False, only show correct and incorrect (miss should be 0)
                    miss_count = c_false["n_miss"]
                    if miss_count > 0:
                        # Warning: miss should not exist for answerable=False
                        print(f"  └─ [answerable=False] n={c_false['n']}, correct={c_false['n_correct']} ({correct_pct:.1f}%), "
                              f"incorrect={c_false.get('n_incorrect', 0)} ({incorrect_pct:.1f}%) "
                              f"⚠️ UNEXPECTED miss={miss_count}")
                    else:
                        print(f"  └─ [answerable=False] n={c_false['n']}, correct={c_false['n_correct']} ({correct_pct:.1f}%), "
                              f"incorrect={c_false.get('n_incorrect', 0)} ({incorrect_pct:.1f}%)")
            
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"[Warning] Failed to print detailed validation counts: {e}")
            # Fallback to simple print
            for ds, c in counts.items():
                print(f"[Validation Counts] {ds} -> n={c['n']}, correct={c['n_correct']}, miss={c['n_miss']}, incorrect={c.get('n_incorrect', 0)}")

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)
        print(f'Save checkpoints in folder: {actor_local_path}')

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else \
                os.path.join(self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, 'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))
        print(f'Save latest checkpointed iteration tracker in folder: {local_latest_checkpointed_iteration}')

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        print(f'Setting global step to {self.global_steps}')

        # global_step_folder = "/home/project/11004114/rl/verl/checkpoints/GRPO-MATH/Qwen-7B-Base/global_step_2000"
        # global_step_folder = "/home/project/11004114/rl/verl/checkpoints/GRPO-Hotpot/Qwen-7B-Base/global_step_800"

        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        batches_per_epoch = len(self.train_dataloader)
        start_epoch = 0
        skip_batches_in_first_epoch = 0
        if self.global_steps > 0:
            start_epoch = self.global_steps // batches_per_epoch
            skip_batches_in_first_epoch = self.global_steps % batches_per_epoch
            print(f"Resuming from global_step {self.global_steps}. "
                  f"Starting from epoch {start_epoch} and skipping {skip_batches_in_first_epoch} batches.")


        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate(log_sample=True, global_step=0)
            print("\n" + "-" * 80)
            print(f" Initial Validation Score ".center(80, '-'))
            pprint(val_metrics)
            print("=" * 80 + "\n")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return
        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        best_test_score = float('-inf')  # Track best test score
        best_ckpt_path = None  # Track best checkpoint path
        latest_ckpt_path = None  # Track latest checkpoint path

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)  # train_batch_size

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])  # train_batch_size

                is_last_step = self.global_steps >= self.total_training_steps

                with (_timer('step', timing_raw)):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)  # train_batch_size * rollout.n

                    # Log one training sample at the same frequency as validation
                    if self.config.trainer.test_freq > 0 and \
                       (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        try:
                            # Decode prompt, handling padding correctly
                            prompt_ids = gen_batch.batch['input_ids'][0]
                            prompt_mask = gen_batch.batch['attention_mask'][0]
                            valid_prompt_ids = prompt_ids[prompt_mask.bool()]
                            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

                            # Decode and clean response from the first repetition
                            response_ids = gen_batch_output.batch['responses'][0]
                            response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                            cleaned_response = re.sub(r'\n{2,}', '\n', response_str).strip()

                            print("\n" + "*" * 80)
                            print(f" Training Sample @ Step {self.global_steps} ".center(80, '*'))
                            print("--- PROMPT ---")
                            print(prompt_str)
                            print("\n--- RESPONSE (cleaned) ---")
                            print(cleaned_response)
                            print("*" * 80 + "\n")
                        except Exception as e:
                            print(f"Failed to log training sample: {e}")

                    if self.config.algorithm.adv_estimator == 'remax':
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            rm_output = self.rm_wg.compute_rm_score(batch)
                            
                            # ===== Debug: 检查 rm_output 状态 =====
                            reward_tensor = rm_output.batch['reward_scores']
                            format_tensor = rm_output.batch['format_scores']
                            answer_tensor = rm_output.batch['answer_scores']
                            reasoning_tensor = rm_output.batch['reasoning_scores']
                            
                            if 'sentence_mask' in rm_output.batch:
                                sentence_mask = rm_output.batch['sentence_mask']
                            else:
                                sentence_mask = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)

                            batch.batch['reasoning_scores'] = reasoning_tensor
                            batch.batch['sentence_mask'] = sentence_mask

                            # 传递句子切分点
                            if 'sentence_split_points' in rm_output.batch:
                                batch.batch['sentence_split_points'] = rm_output.batch['sentence_split_points']
                        else:
                            reward_tensor, format_tensor, answer_tensor, base_reward_tensor = self.reward_fn(batch)
                            # Note: base_reward_tensor is not used in training, only in validation
                        
                        use_reward_mode = os.environ.get('USE_REWARD', 'TRUTHRL').strip().upper()
                        rt = reward_tensor
                        # 阈值：奖励函数通常仅在最后一个有效 token 上给出 {-1, 0, +1}
                        pos_mask = rt >= 0.999
                        neg_mask = rt <= -0.999
                        zero = torch.zeros_like(rt, dtype=rt.dtype, device=rt.device)
                        one = torch.ones_like(rt, dtype=rt.dtype, device=rt.device)
                        minus_one = -one
                        if use_reward_mode == 'GRPO':
                            # 正确 +1，错误 / miss 均 +0
                            mapped_rewards = torch.where(pos_mask, one, zero)
                        elif use_reward_mode == 'THS':
                            # 正确 +y0，错误 -x0，miss +0
                            if hasattr(self, 'initial_validation_metrics') and self.initial_validation_metrics:
                                x0 = float(self.initial_validation_metrics.get('x0', 1.0))
                                y0 = float(self.initial_validation_metrics.get('y0', 1.0))
                            else:
                                x0, y0 = 1.0, 1.0
                            mapped_rewards = rt.clone()
                            mapped_rewards[pos_mask] = y0
                            mapped_rewards[neg_mask] = -x0
                            # 其余位置保持为 0
                            mapped_rewards = mapped_rewards * one  # no-op, 保持 dtype/device 一致
                        elif use_reward_mode == 'TRUTHRL':
                            # TruthRL：正确 +1，错误 -1，miss +0
                            mapped_rewards = torch.where(
                                pos_mask, one, torch.where(neg_mask, minus_one, zero)
                            )
                        elif use_reward_mode == 'RLCR':
                            # RLCR: reward = 1 - (cr - conf)**2
                            # cr: 正确为1, 错误/miss为0
                            # conf: 从 <confidence> 标签中提取
                            
                            responses = batch.batch['responses']
                            # 如果 tensor 在 GPU，需转 CPU 以便解码
                            if isinstance(responses, torch.Tensor):
                                responses_cpu = responses.cpu().tolist()
                            else:
                                responses_cpu = responses.tolist()

                            conf_scores = []
                            for r_ids in responses_cpu:
                                r_str = self.tokenizer.decode(r_ids, skip_special_tokens=True)
                                match = re.search(r'<confidence>(.*?)</confidence>', r_str)
                                val = 0.5 # 默认值
                                if match:
                                    try:
                                        v = float(match.group(1).strip())
                                        val = max(0.0, min(1.0, v))
                                    except ValueError:
                                        pass
                                conf_scores.append(val)
                            
                            conf_tensor = torch.tensor(conf_scores, device=rt.device, dtype=rt.dtype)
                            
                            # cr: pos_mask -> 1.0, else -> 0.0
                            cr_tensor = torch.where(pos_mask, one, zero)
                            
                            # Calculate RLCR reward
                            mapped_rewards = one - (cr_tensor - conf_tensor) ** 2
                        else:
                            # 兜底：按 TruthRL 处理
                            mapped_rewards = torch.where(
                                pos_mask, one, torch.where(neg_mask, minus_one, zero)
                            )

                        # 将映射后的分数作为 token_level_scores（用于日志与后续 KL 处理）
                        batch.batch['token_level_scores'] = mapped_rewards
                        batch.batch['format_scores'] = format_tensor
                        batch.batch['answer_scores'] = answer_tensor

                        # 计算 rewards：如不开启 actor 内部 KL，则在 reward 层面施加 KL 惩罚
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            # 已在 token_level_scores 完成映射；此处直接作为 rewards
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = self.compute_advantage(batch,
                                                      adv_estimator=self.config.algorithm.adv_estimator,
                                                      gamma=self.config.algorithm.gamma,
                                                      lam=self.config.algorithm.lam,
                                                      num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # reward
                    reward_metrics = compute_reward_metrics(batch)
                    metrics.update(reward_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate(log_sample=True, global_step=self.global_steps)
                            if is_last_step:
                                last_val_metrics = val_metrics
                            else:
                                print("\n" + "-" * 80)
                                print(f" Validation Score @ step {self.global_steps} ".center(80, '-'))
                                pprint(val_metrics)
                                print("=" * 80 + "\n")
                        metrics.update(val_metrics)
                        
                        # Save checkpoint logic: keep best + latest (two checkpoints max)
                        # Check if we should use THS for checkpoint selection
                        use_ths_for_checkpoint = getattr(self.config.trainer, 'use_ths_for_checkpoint', False)
                        
                        if use_ths_for_checkpoint and 'val/ths' in val_metrics:
                            # Use THS for checkpoint selection
                            current_ths = val_metrics['val/ths']
                            
                            # Initialize best_ths if not already done
                            if not hasattr(self, 'best_ths'):
                                self.best_ths = float('-inf')
                            
                            is_new_best = current_ths > self.best_ths
                            
                            if is_new_best:
                                self.best_ths = current_ths
                                print("\n" + "🎯" * 40)
                                print(f" NEW BEST THS: {self.best_ths:.4f} (prev: {self.best_ths - (current_ths - self.best_ths):.4f}) ".center(80, '🎯'))
                                print("🎯" * 40 + "\n")
                        else:
                            # Default: use test_score for checkpoint selection
                            test_scores = [v for k, v in val_metrics.items() if k.startswith('val/test_score/') and not ('answerable' in k)]
                            if test_scores:
                                current_test_score = sum(test_scores) / len(test_scores)
                                is_new_best = current_test_score > best_test_score
                                
                                if is_new_best:
                                    best_test_score = current_test_score
                                    print("\n" + "🎉" * 40)
                                    print(f" NEW BEST SCORE: {best_test_score:.4f} (prev: {best_test_score - (current_test_score - best_test_score):.4f}) ".center(80, '🎉'))
                                    print("🎉" * 40 + "\n")
                        
                        # Save checkpoint regardless of selection method
                        with _timer('save_checkpoint', timing_raw):
                            import shutil
                            
                            # Compute new checkpoint path
                            new_ckpt_path = os.path.join(self.config.trainer.default_local_dir,
                                                        f'global_step_{self.global_steps}')
                            
                            # Save new checkpoint
                            self._save_checkpoint()
                            print(f"✅ Saved checkpoint: {new_ckpt_path}")
                            
                            # Update best checkpoint path if this is new best
                            if is_new_best:
                                # Delete old best checkpoint if it exists and is different from latest
                                if best_ckpt_path is not None and best_ckpt_path != latest_ckpt_path:
                                    if os.path.exists(best_ckpt_path):
                                        print(f"🗑️  Removing old best checkpoint: {best_ckpt_path}")
                                        shutil.rmtree(best_ckpt_path)
                                best_ckpt_path = new_ckpt_path
                                print(f"🏆 Best checkpoint: {best_ckpt_path}")
                            
                            # Delete old latest checkpoint if it exists and is different from best
                            if latest_ckpt_path is not None and latest_ckpt_path != best_ckpt_path:
                                if os.path.exists(latest_ckpt_path):
                                    print(f"🗑️  Removing old latest checkpoint: {latest_ckpt_path}")
                                    shutil.rmtree(latest_ckpt_path)
                            
                            # Update latest checkpoint path
                            latest_ckpt_path = new_ckpt_path
                            print(f"📌 Latest checkpoint: {latest_ckpt_path}")
                            
                            # Summary
                            if best_ckpt_path == latest_ckpt_path:
                                print(f"💾 Keeping 1 checkpoint (best == latest): {best_ckpt_path}")
                            else:
                                print(f"💾 Keeping 2 checkpoints - Best: {best_ckpt_path}, Latest: {latest_ckpt_path}")
                        
                        if use_ths_for_checkpoint and not is_new_best:
                            print(f"\nℹ️  THS {current_ths:.4f} <= best {self.best_ths:.4f}\n")
                        elif not use_ths_for_checkpoint and not is_new_best and 'current_test_score' in locals():
                            print(f"\nℹ️  Score {current_test_score:.4f} <= best {best_test_score:.4f}\n")

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # ========== FORMAT ANCHORING ==========
                # 定期进行 SFT 校准以维持指令遵循能力
                # 使用独立的数据（index 3001-4000）避免数据泄漏
                if hasattr(self, 'format_anchor') and self.format_anchor is not None:
                    if self.format_anchor.should_anchor(self.global_steps):
                        with _timer('format_anchor', timing_raw):
                            try:
                                # 通过 RPC 调用 worker 中的格式校准方法
                                anchor_outputs = self.actor_rollout_wg.format_anchor(
                                    format_anchor_config=self.format_anchor.config,
                                    format_anchor_dataset=self.format_anchor.dataset.samples
                                )
                                
                                # ONE_TO_ALL 模式返回列表，取第一个结果（所有 worker 返回相同的统计信息）
                                if isinstance(anchor_outputs, list):
                                    anchor_output = anchor_outputs[0]
                                else:
                                    anchor_output = anchor_outputs
                                
                                # 提取统计信息
                                anchor_stats = {
                                    'anchor_loss': anchor_output.meta_info.get('anchor_loss', 0.0),
                                    'anchor_samples': anchor_output.meta_info.get('anchor_samples', 0),
                                }
                                metrics.update(anchor_stats)
                                
                                # 更新历史记录
                                self.format_anchor.total_anchors += 1
                                self.format_anchor.anchor_history.append({
                                    'step': self.global_steps,
                                    'loss': anchor_stats['anchor_loss'],
                                    'samples': anchor_stats['anchor_samples']
                                })
                                
                                if self.format_anchor.config.verbose:
                                    print(f"\n{'─'*60}")
                                    print(f"🔧 格式校准完成 @ Step {self.global_steps}")
                                    print(f"  平均 loss: {anchor_stats['anchor_loss']:.4f}")
                                    print(f"  样本数: {anchor_stats['anchor_samples']}")
                                    print(f"{'─'*60}\n")
                            except AttributeError as e:
                                print(f"⚠️  格式校准失败: Worker 不支持 format_anchor 方法")
                                print(f"  请确保 FSDP worker 中已添加 format_anchor 方法")
                                print(f"  错误详情: {e}")
                            except Exception as e:
                                print(f"⚠️  格式校准失败: {e}")
                                import traceback
                                traceback.print_exc()
                # ========================================

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    print("\n" + "-" * 80)
                    print(f" Final Validation Score ".center(80, '-'))
                    pprint(last_val_metrics)
                    print("=" * 80 + "\n")
                    return

                self.global_steps += 1

# Bind module-level function as a method of RayPPOTrainer to ensure attribute access works
RayPPOTrainer.compute_advantage = compute_advantage
