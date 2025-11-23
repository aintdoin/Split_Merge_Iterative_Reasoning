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
The main entry point to run the PPO algorithm
"""

import logging
import os
import re
import ast
import nltk
import warnings
import numpy as np
import functools
import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, \
    load_fsdp_param_and_grad
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        raise ValueError(
            'HSDP is not supported yet because it produces incorrect results for now. Please set fsdp_size=-1')
        assert world_size % fsdp_size == 0
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                            self.ulysses_sequence_parallel_size)
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0
        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False,
                               use_liger=False,
                               role='actor'):
        from verl.utils.model import print_model_size, update_model_config, get_generation_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
        from torch import optim

        assert role in ['actor', 'ref']

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == 'actor' else CPUOffload(offload_params=True)
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if role == 'actor':
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            local_path = copy_local_path_from_hdfs(self.config.model.path)
            if vllm_mode == 'customized':
                rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                      config=self.config.rollout,
                                      tokenizer=self.tokenizer,
                                      model_hf_config=self.actor_model_config)
            elif vllm_mode == 'spmd':
                rollout = vLLMRollout(model_path=local_path,
                                      config=self.config.rollout,
                                      tokenizer=self.tokenizer,
                                      model_hf_config=self.actor_model_config,
                                      device_mesh=rollout_device_mesh)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                               inference_engine=rollout.inference_engine,
                                                               model_config=self.actor_model_config,
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False),
                use_liger=self.config.model.get('use_liger', False),
                role='actor')

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False),
                                                               use_liger=self.config.model.get('use_liger', False),
                                                               role='ref')[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(model=self.actor_module_fsdp,
                                                            optimizer=self.actor.actor_optimizer,
                                                            lr_scheduler=self.actor_lr_scheduler,
                                                            tokenizer=self.tokenizer)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name='update_policy', logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            log_gpu_memory_usage('After update policy', logger=logger)

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={'metrics': metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def format_anchor(self, format_anchor_config, format_anchor_dataset):
        """
        定期格式校准：执行SFT微调以维持指令遵循能力
        
        Args:
            format_anchor_config: FormatAnchorConfig 对象
            format_anchor_dataset: 预处理好的格式校准数据样本列表
        
        Returns:
            DataProto with metrics in meta_info
        """
        assert self._is_actor, "Format anchoring only works on actor"
        
        # Load model and optimizer if offloaded
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                    device_id=torch.cuda.current_device(),
                                    load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())
        
        # Save original learning rate
        original_lrs = [pg['lr'] for pg in self.actor_optimizer.param_groups]
        
        # Set format anchoring learning rate (lower than main training)
        anchor_lr = original_lrs[0] * format_anchor_config.lr_ratio
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = anchor_lr
        
        # Set model to training mode
        self.actor_module_fsdp.train()
        
        total_loss = 0.0
        num_samples = 0
        
        if format_anchor_config.verbose and torch.distributed.get_rank() == 0:
            print(f"\n{'─'*60}")
            print(f"🔧 格式校准中... (LR: {anchor_lr:.2e})")
        
        import random
        for step in range(format_anchor_config.steps_per_anchor):
            # Sample a batch from format_anchor_dataset
            if len(format_anchor_dataset) < format_anchor_config.batch_size:
                batch_samples = random.choices(format_anchor_dataset, k=format_anchor_config.batch_size)
            else:
                batch_samples = random.sample(format_anchor_dataset, format_anchor_config.batch_size)
            
            # Prepare batch data
            prompts = [sample['prompt'] for sample in batch_samples]
            responses = [sample['response'] for sample in batch_samples]
            full_texts = [p + r for p, r in zip(prompts, responses)]
            
            # Tokenize
            encodings = self.tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].cuda()
            attention_mask = encodings['attention_mask'].cuda()
            
            # Create labels (only compute loss on response part)
            labels = input_ids.clone()
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
                prompt_length = len(prompt_tokens)
                labels[i, :prompt_length] = -100  # Ignore prompt in loss
            
            # Forward pass
            outputs = self.actor_module_fsdp(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else self._compute_sft_loss(outputs.logits, labels)
            
            # Backward pass
            self.actor_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor_module_fsdp.parameters(), max_norm=1.0)
            
            self.actor_optimizer.step()
            
            total_loss += loss.item()
            num_samples += len(batch_samples)
            
            if format_anchor_config.verbose and torch.distributed.get_rank() == 0:
                print(f"  Step {step+1}/{format_anchor_config.steps_per_anchor}: loss={loss.item():.4f}")
        
        # Restore original learning rate
        for param_group, original_lr in zip(self.actor_optimizer.param_groups, original_lrs):
            param_group['lr'] = original_lr
        
        avg_loss = total_loss / format_anchor_config.steps_per_anchor
        
        if format_anchor_config.verbose and torch.distributed.get_rank() == 0:
            print(f"  ✓ 校准完成: 平均 loss={avg_loss:.4f}, 样本数={num_samples}")
            print(f"{'─'*60}\n")
        
        # Offload if needed
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        
        torch.cuda.empty_cache()
        
        # Return metrics
        metrics = {
            'anchor_loss': avg_loss,
            'anchor_samples': num_samples,
        }
        
        return DataProto(meta_info=metrics)
    
    def _compute_sft_loss(self, logits, labels):
        """Compute SFT loss manually if model doesn't return it"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to(torch.cuda.current_device())

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        data = data.to('cuda')
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info['temperature'] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'old_log_probs': output},
                                         meta_info={'temperature': self.config.rollout.temperature})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After compute_log_prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_policy.actor_module._handle.reshard(True)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)
            self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                      self.ulysses_sequence_parallel_size)
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=sharding_strategy,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False,
                             device_mesh=self.device_mesh,
                             cpu_offload=None)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(model=self.critic_module,
                                                        optimizer=self.critic_optimizer,
                                                        lr_scheduler=self.critic_lr_scheduler,
                                                        tokenizer=self.tokenizer)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=True):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        """
        Initialize MiniCheck API client instead of loading local model
        """
        import os
        import requests
        
        # Get MiniCheck API configuration from environment
        self.minicheck_api_base = os.environ.get('MINICHECK_API_BASE', '').strip()
        self.minicheck_use_api = bool(self.minicheck_api_base)
        self.minicheck_max_workers = int(os.environ.get('MINICHECK_MAX_WORKERS', '8'))
        self.minicheck_timeout = int(os.environ.get('MINICHECK_TIMEOUT', '30'))
        
        # Initialize tokenizer for token counting
        if self.config.model.input_tokenizer is not None:
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
        else:
            # Use a default tokenizer for token counting if not specified
            from transformers import AutoTokenizer
            self.input_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Test API connection only when explicitly configured
        if self.minicheck_use_api:
            try:
                response = requests.get(f"{self.minicheck_api_base}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"MiniCheck API is available at {self.minicheck_api_base}")
                else:
                    logger.warning(f"MiniCheck API health check failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to connect to MiniCheck API at {self.minicheck_api_base}: {e}")
                logger.error("Please ensure the MiniCheck server is running using start_minicheck_server.sh")
        else:
            logger.info("MiniCheck API disabled by default (set MINICHECK_API_BASE to enable)")
        
        # Return None as we're using API instead of local model
        return None

    def _call_minicheck_api(self, docs, claims):
        """
        Call MiniCheck API to score claims against documents
        
        Args:
            docs: List of documents
            claims: List of claims
            
        Returns:
            List of probabilities (raw_probs from MiniCheck)
        """
        import requests
        import json
        
        # If API not configured, return neutral scores
        if not getattr(self, 'minicheck_use_api', False):
            return [0.5] * len(docs)

        try:
            payload = {
                "docs": docs,
                "claims": claims,
                "chunk_size": 32768
            }
            
            response = requests.post(
                f"{self.minicheck_api_base}/score",
                json=payload,
                timeout=self.minicheck_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['raw_probs']
            else:
                logger.error(f"MiniCheck API error: {response.status_code} - {response.text}")
                # Return default scores on error
                return [0.5] * len(docs)
                
        except Exception as e:
            logger.error(f"Error calling MiniCheck API: {e}")
            # Return default scores on error
            return [0.5] * len(docs)
    
    def _predict_minicheck(self, doc_claim_pairs):
        """
        Predict scores for document-claim pairs using MiniCheck API
        
        Args:
            doc_claim_pairs: List of (doc, claim) tuples
            
        Returns:
            List of torch tensors with probabilities
        """
        if not doc_claim_pairs:
            return []
        
        docs = [pair[0] for pair in doc_claim_pairs]
        claims = [pair[1] for pair in doc_claim_pairs]
        
        # Call API
        probs = self._call_minicheck_api(docs, claims)
        
        # Convert to torch tensors to match HHEM interface
        return [torch.tensor(prob, dtype=torch.float32) for prob in probs]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)

    def _select_rm_score_fn(self, data_source):
        from verl.utils.reward_score import gsm8k, math, multiply, countdown, kk, halueval, hotpot
        if data_source == 'GSM8K':
            return gsm8k.compute_score
        elif data_source == 'MATH':
            return math.compute_score
        elif "multiply" in data_source or "arithmetic" in data_source:
            return multiply.compute_score
        elif "countdown" in data_source:
            return countdown.compute_score
        elif "kk" in data_source:
            return kk.compute_score
        elif data_source == 'halueval':
            return halueval.compute_score
        elif data_source == 'ASQA':
            return asqa.compute_score
        elif data_source in ['hotpot', '2wikimultihop', 'musique_ans', 'musique']:
            return hotpot.compute_score
        else:
            raise NotImplementedError

    def extract_solution(self, solution_str: str) -> str:
        """Extracts the final answer from the model's response string.

        Args:
            solution_str: Raw response string from the language model

        Returns:
            only response string
        """
        # Split response to isolate assistant output
        if "Assistant:" in solution_str:
            processed_str = solution_str.split("Assistant:", 1)[1]
        elif "<|im_start|>assistant" in solution_str:
            processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
        elif "<｜Assistant｜>" in solution_str:
            processed_str = solution_str.split("<｜Assistant｜>", 1)[1]
        elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
            processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
        else:
            return solution_str

        return processed_str.strip()

    def validate_model_reasoning_documents_only(self, documents, output_str):
        reasoning_pattern = r'<think>(.*?)</think>'
        response = output_str
        matches = list(re.finditer(reasoning_pattern, response, re.DOTALL))
        if not matches:
            reasoning_score = -2
            # Removed verbose print to reduce log clutter
            # print(f"\n[Reasoning Validation] Skipped due to missing reasoning text (Reasoning score: {reasoning_score})")
            return reasoning_score

        # Removed verbose print to reduce log clutter
        # print(f"\n[Reasoning Validation]")

        reasoning_str = matches[-1].group(1).strip()
        
        step_wise_scores = []
        sentence_mask = []
        sentences_list = nltk.sent_tokenize(reasoning_str)
        
        # Batch all predictions for efficiency
        doc_claim_pairs = [(documents, sentence) for sentence in sentences_list]
        sentence_scores = self._predict_minicheck(doc_claim_pairs)
        
        for i, (sentence, sentence_score) in enumerate(zip(sentences_list, sentence_scores)):
            if sentence_score.item() > 0.5:
                step_wise_scores.append(1.0)
                sentence_mask.extend([1.0] * len(self.input_tokenizer.tokenize(sentence)))
            else:
                step_wise_scores.append(-1.0)
                sentence_mask.extend([-1.0] * len(self.input_tokenizer.tokenize(sentence)))
        
        reasoning_score = np.mean(step_wise_scores)
        # Removed verbose prints to reduce log clutter
        # print(f"  Factuality judge: {step_wise_scores}")
        # print(f"  Reasoning score: {reasoning_score}")

        return reasoning_score, sentence_mask

    def validate_model_reasoning(self, output_str, documents, evidences):
        reasoning_pattern = r'<think>(.*?)</think>'
        response = self.extract_solution(output_str)
        matches = list(re.finditer(reasoning_pattern, response, re.DOTALL))
        if not matches:
            reasoning_score = -2
            # Removed verbose print to reduce log clutter
            # print(f"\n[Reasoning Validation] Skipped due to missing reasoning text (Reasoning score: {reasoning_score})")
            return reasoning_score

        # Removed verbose print to reduce log clutter
        # print(f"\n[Reasoning Validation]")
        reasoning_str = matches[-1].group(1).strip()

        step_wise_scores = []
        sentence_mask = []
        sentences_list = nltk.sent_tokenize(reasoning_str)
        
        for i, sentence in enumerate(sentences_list):
            # 首先比对和evidence的相关性
            evidence_list = eval(evidences)
            evidence_pairs = [(e, sentence) for e in evidence_list]
            evidence_scores = self._predict_minicheck(evidence_pairs)
            evidence_score = max(evidence_scores)
            
            if evidence_score.item() > 0.5:
                # 如果和evidence相关性高，步骤得分为1.0
                step_wise_scores.append(1.0)
                sentence_mask.extend([1.0] * len(self.input_tokenizer.tokenize(sentence)))
            else:
                # 如果和evidence相关性低，比对和document的相关性
                # 修复：将documents拆分，避免超过512 tokens限制
                try:
                    doc_list = eval(documents)
                    doc_texts = []
                    for doc_item in doc_list:
                        if isinstance(doc_item, list) and len(doc_item) >= 2:
                            # 结构：['title', ['sent1', 'sent2', ...]]
                            if isinstance(doc_item[1], list):
                                doc_texts.extend(doc_item[1])
                        elif isinstance(doc_item, str):
                            doc_texts.append(doc_item)
                    
                    if len(doc_texts) > 0:
                        doc_pairs = [(doc_text, sentence) for doc_text in doc_texts]
                        doc_scores = self._predict_minicheck(doc_pairs)
                        document_score = max(doc_scores)
                    else:
                        document_score = self._predict_minicheck([(documents, sentence)])[0]
                except:
                    # 降级：如果解析失败，直接用原documents
                    document_score = self._predict_minicheck([(documents, sentence)])[0]
                
                # ===== 增强判定逻辑：关键词检测 + 自适应阈值 =====
                import re
                
                # 检测是否明确引用documents
                mentions_doc = bool(re.search(r'\b(?:document|doc|evidence)\s+(?:\d+|[A-Z])\b', 
                                               sentence, re.IGNORECASE))
                
                # 检测总结性关键词（仅在最后一句启用）
                conclusion_keywords = r'\b(therefore|thus|hence|so|consequently|in conclusion|in summary|to conclude|overall|finally|it\'s clear|clear that|safe to conclude|putting it all)\b'
                is_conclusion = bool(re.search(conclusion_keywords, sentence, re.IGNORECASE))
                is_last_sentence = (i == len(sentences_list) - 1)
                
                score_value = document_score.item()
                
                # 四层判定逻辑（与stepwise方法一致）
                if is_conclusion and is_last_sentence:
                    # 总结性句子：至少是-0.2（背景信息），而不是-1.0（完全错误）
                    step_wise_scores.append(-0.2)
                    sentence_mask.extend([-0.2] * len(self.input_tokenizer.tokenize(sentence)))
                elif mentions_doc and score_value > 0.2:
                    # 明确引用documents，且分数>0.2 → 背景信息 (-0.2)
                    step_wise_scores.append(-0.2)
                    sentence_mask.extend([-0.2] * len(self.input_tokenizer.tokenize(sentence)))
                elif score_value > 0.35:
                    # 高相关性（降低阈值从0.5到0.35） → 背景信息 (-0.2)
                    step_wise_scores.append(-0.2)
                    sentence_mask.extend([-0.2] * len(self.input_tokenizer.tokenize(sentence)))
                else:
                    # 低相关性且无引用 → 完全错误 (-1.0)
                    step_wise_scores.append(-1.0)
                    sentence_mask.extend([-1.0] * len(self.input_tokenizer.tokenize(sentence)))
        
        reasoning_score = np.mean(step_wise_scores)
        # Removed verbose prints to reduce log clutter
        # print(f"  Factuality judge: {step_wise_scores}")
        # print(f"  Reasoning score: {reasoning_score}")

        return reasoning_score, sentence_mask

    def validate_model_reasoning_stepwise(self, output_str, documents, evidences, valid_response_ids=None, question=None, ground_truth=None, answer_aliases=None, answerable=True):
        reasoning_pattern = r'<think>(.*?)</think>'
        answer_pattern = r'<answer>(.*?)</answer>'

        response = output_str
        eos_tag = ' '

        # CRITICAL FIX: Ensure token alignment with valid_response_ids
        if valid_response_ids is not None:
            # Use the actual response token IDs to ensure perfect alignment
            num_tokens = len(valid_response_ids)
            
            # ===== METHOD 1: 逐个decode累加（当前方法，可能有问题）=====
            token_offsets = []
            char_pos = 0
            for token_id in valid_response_ids:
                token_text = self.input_tokenizer.decode([token_id])
                token_len = len(token_text)
                token_offsets.append((char_pos, char_pos + token_len))
                char_pos += token_len
            
        else:
            # Fallback: re-tokenize (may cause alignment issues!)
            enc = self.input_tokenizer(
                response,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            offsets = enc.get('offset_mapping')
            if isinstance(offsets[0], tuple) or isinstance(offsets[0], list):
                token_offsets = offsets
            else:
                token_offsets = offsets[0]
            num_tokens = len(token_offsets)
        
        sentence_mask = [0.0] * num_tokens  # 默认全0.0
        def set_span_value(char_start: int, char_end: int, value: float):
            for ti, (ts, te) in enumerate(token_offsets):
                if ts >= char_start and ts < char_end:
                    sentence_mask[ti] = value

        # 初始化分段分数与总体reasoning分数
        step_wise_scores = []
        reasoning_score = 0.0

        import re
        think_matches = list(re.finditer(reasoning_pattern, response, re.DOTALL))
        think_match = think_matches[-1] if len(think_matches) > 0 else None

        if think_match is not None:
            t_content_start = think_match.start(1)
            t_content_end = think_match.end(1)
            reasoning_str = response[t_content_start:t_content_end]
            enum_pattern = re.compile(r'(?m)^(\s*)(\d+)\.(\s*)')
            markers = list(enum_pattern.finditer(reasoning_str))
            n_marker = len(markers)
            last_end = 0
            for i, m in enumerate(markers):
                num_rel_start = m.start(2)
                num_rel_end = m.end(2)
                # 仅包含数字与紧随其后的点，不包含点后的空白
                dot_rel_end = m.end(2) + 1
                num_global_start = t_content_start + num_rel_start
                dot_global_end = t_content_start + dot_rel_end
                set_span_value(num_global_start, dot_global_end, 1.0)
                # 内容段从点后位置开始，使" 空格+词"类token归入内容
                seg_rel_start = dot_rel_end
                if i + 1 < n_marker:
                    seg_rel_end = markers[i+1].start(0)
                else:
                    seg_rel_end = len(reasoning_str)
                segment_text = reasoning_str[seg_rel_start:seg_rel_end]
                global_start = t_content_start + seg_rel_start
                global_end = t_content_start + seg_rel_end
                if not segment_text.strip():
                    last_end = seg_rel_end
                    continue
                try:
                    judge_score = self._evaluate_reasoning_segment(
                        segment_text=segment_text,
                        documents=documents,
                        evidences=evidences,
                        question=question,
                        ground_truth=ground_truth,
                        answer_aliases=answer_aliases,
                        answerable=answerable,
                    )
                except Exception:
                    judge_score = 0.0
                if judge_score is None:
                    judge_score = 0.0
                set_span_value(global_start, global_end, judge_score)
                step_wise_scores.append(float(judge_score))
                last_end = seg_rel_end
            # 补充末尾不在枚举内残段
            if n_marker == 0 or last_end < len(reasoning_str):
                seg_rel_start = last_end
                seg_rel_end = len(reasoning_str)
                segment_text = reasoning_str[seg_rel_start:seg_rel_end]
                global_start = t_content_start + seg_rel_start
                global_end = t_content_start + seg_rel_end
                if segment_text.strip():
                    try:
                        judge_score = self._evaluate_reasoning_segment(
                            segment_text=segment_text,
                            documents=documents,
                            evidences=evidences,
                            question=question,
                            ground_truth=ground_truth,
                            answer_aliases=answer_aliases,
                            answerable=answerable,
                        )
                    except Exception:
                        judge_score = 1.0
                    if judge_score is None:
                        judge_score = 1.0
                    set_span_value(global_start, global_end, judge_score)
                    step_wise_scores.append(float(judge_score))

        else:
            # 无<think>：若存在</think>，则默认将[开头, </think>)视作推理区间
            think_close_tag = '</think>'
            t_close_pos = response.find(think_close_tag)
            if t_close_pos != -1:
                t_content_start = 0
                t_content_end = t_close_pos
                reasoning_str = response[t_content_start:t_content_end]
                enum_pattern = re.compile(r'(?m)^(\s*)(\d+)\.(\s*)')
                markers = list(enum_pattern.finditer(reasoning_str))
                n_marker = len(markers)
                last_end = 0
                for i, m in enumerate(markers):
                    num_rel_start = m.start(2)
                    num_rel_end = m.end(2)
                    dot_rel_end = m.end(2) + 1
                    num_global_start = t_content_start + num_rel_start
                    dot_global_end = t_content_start + dot_rel_end
                    set_span_value(num_global_start, dot_global_end, 1.0)
                    seg_rel_start = dot_rel_end
                    if i + 1 < n_marker:
                        seg_rel_end = markers[i+1].start(0)
                    else:
                        seg_rel_end = len(reasoning_str)
                    segment_text = reasoning_str[seg_rel_start:seg_rel_end]
                    global_start = t_content_start + seg_rel_start
                    global_end = t_content_start + seg_rel_end
                    if not segment_text.strip():
                        last_end = seg_rel_end
                        continue
                    try:
                        judge_score = self._evaluate_reasoning_segment(
                            segment_text=segment_text,
                            documents=documents,
                            evidences=evidences,
                            question=question,
                            ground_truth=ground_truth,
                            answer_aliases=answer_aliases,
                            answerable=answerable,
                        )
                    except Exception:
                        judge_score = 1.0
                    if judge_score is None:
                        judge_score = 1.0
                    set_span_value(global_start, global_end, judge_score)
                    step_wise_scores.append(float(judge_score))
                    last_end = seg_rel_end
                if n_marker == 0 or last_end < len(reasoning_str):
                    seg_rel_start = last_end
                    seg_rel_end = len(reasoning_str)
                    segment_text = reasoning_str[seg_rel_start:seg_rel_end]
                    global_start = t_content_start + seg_rel_start
                    global_end = t_content_start + seg_rel_end
                    if segment_text.strip():
                        try:
                            judge_score = self._evaluate_reasoning_segment(
                                segment_text=segment_text,
                                documents=documents,
                                evidences=evidences,
                                question=question,
                                ground_truth=ground_truth,
                                answer_aliases=answer_aliases,
                                answerable=answerable,
                            )
                        except Exception:
                            judge_score = 0.0
                        if judge_score is None:
                            judge_score = 0.0
                        set_span_value(global_start, global_end, judge_score)
                        step_wise_scores.append(float(judge_score))
        
        
        # <think>区间外token都为0（默认cover）
        # 输出debug
        if torch.distributed.get_rank() == 0:
            pass
            char_pos = 0
            for i,(score,ids) in enumerate(zip(sentence_mask,valid_response_ids)):
                token_text = self.input_tokenizer.decode([ids])
                token_len = len(token_text)
                char_pos += token_len

        # Removed legacy tag/answer span assignments; outside <think> stays 0.0
        for m in re.finditer(r'</?think>', response):
            set_span_value(m.start(), m.end(), 1.0)
        for m in re.finditer(r'</?answer>', response):
            set_span_value(m.start(), m.end(), 1.0)
        # 计算总体reasoning得分（仅基于<think>内容分段）
        if len(step_wise_scores) > 0:
            try:
                reasoning_score = float(np.mean(step_wise_scores))
            except Exception:
                reasoning_score = float(sum(step_wise_scores) / len(step_wise_scores))

        return reasoning_score, sentence_mask

    def _evaluate_reasoning_segment(self, segment_text, documents, evidences, question=None, ground_truth=None, answer_aliases=None, answerable=True):
        """
        使用 LLM 评判该分段推理是否与 evidences 一致：一致->1，不一致/不相关/矛盾->0。
        特殊处理：若 answerable=False 且分段明确表达"insufficient information"，直接返回 1.0。
        优先参考 answer_postprocessor 的 API 配置方式，通过其全局单例调用底层 LLM API。
        若评判不可用或失败，返回 None（调用侧回退为默认分值）。
        """
        try:
            has_idk = self._check_insufficient_info(segment_text, question=question)
            if has_idk and answerable is False:
                return 1.0
            elif has_idk and answerable is True:
                return -1.0
            
            # ===== 常规评分：判断与 evidences 的一致性 =====
            # 解析 evidences（兼容字符串/对象）
            import ast
            evd = evidences
            if isinstance(evidences, str):
                try:
                    evd = ast.literal_eval(evidences)
                except Exception:
                    evd = evidences
            # 仅做提示文本化，避免超长
            def _to_text(x):
                try:
                    return str(x)
                except Exception:
                    return ''
            evd_list = evd if isinstance(evd, (list, tuple)) else [evd]
            evd_texts = [_to_text(x) for x in evd_list if _to_text(x)]
            # 控制条目数量，防止 prompt 过长
            max_evd = 8
            evd_text = '\n'.join([f"- {t}" for t in evd_texts[:max_evd]]) if evd_texts else ""

            # 组织答案与别名
            answers_for_context = []
            if ground_truth is not None:
                if isinstance(ground_truth, (list, tuple)):
                    answers_for_context.extend([_to_text(a) for a in ground_truth if _to_text(a)])
                else:
                    answers_for_context.append(_to_text(ground_truth))
            if answer_aliases:
                if isinstance(answer_aliases, (list, tuple)):
                    answers_for_context.extend([_to_text(a) for a in answer_aliases if _to_text(a)])

            answers_for_context = [a for a in answers_for_context if a]
            answers_block = '\n'.join([f"- {a}" for a in answers_for_context]) if answers_for_context else ""

            # 构造评判提示词
            q_text = (question or '').strip()
            seg_text = (segment_text or '').strip()

            prompt_lines = []
            prompt_lines.append("You are a strict reasoning consistency judge. Decide if the reasoning segment is CONSISTENT with the evidences.")
            prompt_lines.append("Rules:")
            prompt_lines.append("1) Output only one digit: 1 if consistent; 0 if inconsistent, irrelevant, or contradictory.")
            prompt_lines.append("2) Base the decision strictly on the evidences; ignore world knowledge.")
            prompt_lines.append("3) Do not provide explanations.")
            if q_text:
                prompt_lines.append("")
                prompt_lines.append(f"Question:\n{q_text}")
            if answers_block:
                prompt_lines.append("")
                prompt_lines.append(f"Known Answer(s):\n{answers_block}")
            if evd_text:
                prompt_lines.append("")
                prompt_lines.append(f"Evidences:\n{evd_text}")
            prompt_lines.append("")
            prompt_lines.append(f"Reasoning Segment:\n{seg_text}")
            prompt_lines.append("")
            prompt_lines.append("Output (only 0 or 1):")
            judge_prompt = '\n'.join(prompt_lines)

            # 通过全局后处理器复用 LLM Judge API 调用
            from verl.utils.reward_score.answer_postprocessor import get_postprocessor
            post = get_postprocessor()
            if not getattr(post, 'use_judge_api', False):
                return None

            result_text = post._call_judge_api(judge_prompt)
            if not isinstance(result_text, str):
                return None
            s = result_text.strip()
            # 严格解析 0/1；若解析失败，返回 None 交由上层回退
            if s in ('0', '1'):
                return 1.0 if s == '1' else 0.0

            import re as _re
            m = _re.match(r'^\s*([01])', s)
            if m:
                return 1.0 if m.group(1) == '1' else 0.0

            return None
        except Exception:
            return None

    def _check_insufficient_info(self, segment_text, question=None):
        """
        判断推理分段是否明确表达 "insufficient information" 语义。
        仅针对 answerable=False 的样本调用。
        返回 True 表示该分段明确表达信息不足；False 表示未明确表达；None 表示检测失败。
        """
        try:
            seg_text = (segment_text or '').strip()
            if not seg_text:
                return False
            
            q_text = (question or '').strip()
            
            # 构造判断提示词
            prompt_lines = []
            prompt_lines.append("You are a strict semantic judge. Decide if the reasoning segment EXPLICITLY expresses 'insufficient information' or similar meanings.")
            prompt_lines.append("Rules:")
            prompt_lines.append("1) Output only one digit: 1 if the segment explicitly states lack of information (e.g., 'insufficient information', 'not enough data', 'cannot determine'); 0 otherwise.")
            prompt_lines.append("2) Be strict: only return 1 if the meaning is clearly and directly stated.")
            prompt_lines.append("3) Do not provide explanations.")
            if q_text:
                prompt_lines.append("")
                prompt_lines.append(f"Question:\n{q_text}")
            prompt_lines.append("")
            prompt_lines.append(f"Reasoning Segment:\n{seg_text}")
            prompt_lines.append("")
            prompt_lines.append("Output (only 0 or 1):")
            judge_prompt = '\n'.join(prompt_lines)
            
            # 通过全局后处理器复用 LLM Judge API 调用
            from verl.utils.reward_score.answer_postprocessor import get_postprocessor
            post = get_postprocessor()
            if not getattr(post, 'use_judge_api', False):
                return None
            
            result_text = post._call_judge_api(judge_prompt)
            if not isinstance(result_text, str):
                return None
            s = result_text.strip()
            
            # 严格解析 0/1
            if s in ('0', '1'):
                return s == '1'
            
            import re as _re
            m = _re.match(r'^\s*([01])', s)
            if m:
                return m.group(1) == '1'
            
            return None
        except Exception:
            return None


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to(torch.cuda.current_device())

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        answer_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        reasoning_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        sentence_mask_tensor = torch.ones_like(data.batch['responses'], dtype=torch.float32)

        # perform forward computation
        with self.ulysses_sharding_manager:
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.input_tokenizer.decode(sequences)
                sequences_input = self.input_tokenizer.decode(valid_response_ids)
                documents = data_item.non_tensor_batch['documents'] # [[key, [list]]]
                ground_truth = data_item.non_tensor_batch['answer'] # str
                data_source = data_item.non_tensor_batch['data_source']
                strategy = os.environ.get('STRATEGY', None)
                compute_score_fn = self._select_rm_score_fn(data_source)
                # extract answerable strictly from extra_info
                extra_info = data_item.non_tensor_batch.get('extra_info', {})
                if isinstance(extra_info, str):
                    try:
                        import json as _json
                        extra_info = _json.loads(extra_info)
                    except Exception:
                        extra_info = {}
                if not isinstance(extra_info, dict):
                    extra_info = {}
                raw_flag = extra_info.get('answerable', True)
                if isinstance(raw_flag, str):
                    rf = raw_flag.strip().lower()
                    answerable_flag = True if rf == 'true' else False if rf == 'false' else True
                elif isinstance(raw_flag, bool):
                    answerable_flag = raw_flag
                else:
                    answerable_flag = True
                
                # Extract answer_aliases for data sources that support it
                answer_aliases = extra_info.get('answer_aliases', []) if isinstance(extra_info, dict) else []
                
                # Call compute_score_fn with or without answer_aliases based on data source
                if data_source in ['hotpot', '2wikimultihop', 'musique']:
                    format_score, answer_score, base_reward = compute_score_fn(
                        response=sequences_str,
                        ground_truth=ground_truth,
                        documents=documents,
                        answerable=answerable_flag,
                        answer_aliases=answer_aliases,
                        enable_postprocessing=True,
                    )
                else:
                    format_score, answer_score, base_reward = compute_score_fn(
                        response=sequences_str,
                        ground_truth=ground_truth,
                        documents=documents,
                        answerable=answerable_flag,
                    )
                # Note: base_reward is not used in training, only stored for validation
                evidences = data_item.non_tensor_batch['evidences']
                if strategy == 'grpo':
                    reasoning_score = 0
                    sentence_mask_tensor[i, :] = 0.0
                elif strategy == 'fspo':
                    if data_source in ["MATH", "GSM8K"]:
                        reasoning_score = 0
                        sentence_mask_tensor[i, :] = 0.0
                    elif format_score == -2:
                        reasoning_score = 0
                        sentence_mask_tensor[i, :] = 0.0
                        # Removed verbose print to reduce log clutter
                        # print(f"\n[Reasoning Validation] Skipped due to format errors (Reasoning score: {reasoning_score})")
                    else:
                        reasoning_score, sentence_mask = self.validate_model_reasoning_documents_only(documents, sequences_input) # 标量, 一维列表且取值均为-1, 表示每个token的mask
                        mask_length = min(len(response_ids), len(sentence_mask)) # token数
                        sentence_mask_tensor[i, :mask_length] = torch.tensor(sentence_mask[:mask_length], dtype=torch.float32)# [batch_size, max_seq_len], 相同于把每个样本的sentence_mask打到batch里 
                elif 'evar' in strategy:
                    sentence_mask_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
                    if format_score == -2:
                        reasoning_score = 0
                        sentence_mask_tensor[i, :] = 0.0
                        # Removed verbose print to reduce log clutter
                        # print(f"\n[Reasoning Validation] Skipped due to format errors (Reasoning score: {reasoning_score})")
                    else:
                        # CRITICAL: Pass valid_response_ids directly to ensure alignment
                        # try to get question if available in batch (optional)
                        try:
                            question_text = data_item.non_tensor_batch['question']
                        except Exception:
                            question_text = None

                        reasoning_score, sentence_mask = self.validate_model_reasoning_stepwise(
                            sequences_input,
                            documents,
                            evidences,
                            valid_response_ids=valid_response_ids,
                            question=question_text,
                            ground_truth=ground_truth,
                            answer_aliases=answer_aliases,
                            answerable=answerable_flag,
                        )
                        # sentence_mask length MUST match valid_response_ids length
                        mask_length = min(len(valid_response_ids), len(sentence_mask))
                        sentence_mask_tensor[i, :mask_length] = torch.tensor(sentence_mask[:mask_length], dtype=torch.float32)

                # Removed verbose prints to reduce log clutter
                # print("\n" + "-" * 80)
                # print(f" Final Score ".center(80, '-'))
                # print(f"  Format score: {format_score}")
                # print(f"  Answer score: {answer_score}")
                # print(f"  Reasoning score: {reasoning_score}")
                try:
                    resp_len_int = int(valid_response_length.item() if hasattr(valid_response_length, "item") else int(valid_response_length))
                except Exception:
                    # Fallback: best-effort conversion
                    resp_len_int = int(valid_response_length) if not isinstance(valid_response_length, torch.Tensor) else int(valid_response_length.detach().cpu().item())
                # 当token超过500时，每多100token，reward减去0.1
                if resp_len_int > 500:
                    excess_tokens = resp_len_int - 500
                    # 计算超出的100token的数量，向上取整
                    excess_hundreds = (excess_tokens + 99) // 100
                    length_penalty = 0.1 * excess_hundreds
                else:
                    length_penalty = 0.0
                total_score = float(answer_score) - length_penalty

                reward_tensor[i, valid_response_length - 1] = total_score

                format_reward_tensor[i] = format_score
                answer_reward_tensor[i] = answer_score
                reasoning_reward_tensor[i] = reasoning_score
                               
            # 显示每个样本的统计
            for sample_idx in range(min(3, sentence_mask_tensor.shape[0])):
                sample_mask = sentence_mask_tensor[sample_idx]
                nonzero_count = (sample_mask != 0).sum().item()
                unique_values = torch.unique(sample_mask).cpu().tolist()
                    
            output = DataProto.from_dict(tensors={'reward_scores': reward_tensor,
                                                  'format_scores': format_reward_tensor,
                                                  'answer_scores': answer_reward_tensor,
                                                  'reasoning_scores': reasoning_reward_tensor,
                                                  'sentence_mask': sentence_mask_tensor})
            
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
        
        output = output.to('cpu')
        return output
