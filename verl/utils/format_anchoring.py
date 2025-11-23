"""
格式校准模块 - 定期 SFT 微调来维持指令遵循能力
严格避免数据泄漏：使用训练集之外的数据（index 3001-4000）
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional
import pyarrow.parquet as pq
import random
from dataclasses import dataclass


@dataclass
class FormatAnchorConfig:
    """格式校准配置"""
    frequency: int = 50              # 每N步校准一次
    steps_per_anchor: int = 2        # 每次校准的步数
    lr_ratio: float = 0.1            # 相对于主训练的学习率比例
    batch_size: int = 16             # 校准batch大小
    data_start_idx: int = 3001       # 数据起始索引（避免泄漏）
    data_end_idx: int = 4000         # 数据结束索引
    format_check_strict: bool = True  # 是否严格检查格式
    verbose: bool = True             # 是否打印详细信息


class FormatAnchoringDataset:
    """
    格式校准数据集
    直接加载预处理好的SFT格式数据（来自 format_anchor_data.py）
    """
    
    def __init__(
        self,
        data_file: str,  # 预处理好的格式校准数据文件路径
        tokenizer,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_file: 预处理好的格式校准数据parquet文件路径
            tokenizer: tokenizer
            max_samples: 最多加载多少个样本（None表示全部加载）
        """
        self.tokenizer = tokenizer
        self.samples = []
        
        print(f"\n{'='*80}")
        print(f"📋 加载格式校准数据")
        print(f"{'='*80}")
        print(f"  文件: {data_file}")
        
        self._load_dataset(data_file, max_samples)
        
        print(f"\n✓ 总共加载 {len(self.samples)} 个格式校准样本")
        print(f"{'='*80}\n")
    
    def _load_dataset(self, file_path: str, max_samples: Optional[int]):
        """从预处理好的数据文件加载样本"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"  ✗ 文件不存在: {file_path}")
                return
            
            # 读取 parquet 文件
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            total_rows = len(df)
            print(f"  ✓ 数据集大小: {total_rows} 个样本")
            
            # 确定实际加载数量
            if max_samples is not None and max_samples < total_rows:
                print(f"  ℹ️  只加载前 {max_samples} 个样本")
                df = df.iloc[:max_samples]
            
            # 加载所有样本
            loaded_count = 0
            for idx, row in df.iterrows():
                # 预处理数据已经包含 prompt 和 response
                prompt = row.get('prompt', '')
                response = row.get('response', '')
                
                if not prompt or not response:
                    continue
                
                # 验证格式
                if self._is_format_valid(response):
                    self.samples.append({
                        'prompt': prompt,
                        'response': response,
                        'dataset': row.get('data_source', 'unknown'),
                        'original_idx': row.get('original_idx', idx),
                        'question': row.get('question', ''),
                        'answer': row.get('answer', ''),
                    })
                    loaded_count += 1
            
            # 统计各数据集样本数
            from collections import Counter
            source_counts = Counter(s['dataset'] for s in self.samples)
            for source, count in source_counts.items():
                print(f"    {source}: {count} 个样本")
            
            print(f"  ✓ 成功加载 {loaded_count} 个有效样本")
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _is_format_valid(self, response: str) -> bool:
        """检查格式是否有效"""
        # 必须有 <answer> 和 </answer>
        if '<answer>' not in response.lower() or '</answer>' not in response.lower():
            return False
        
        # 如果有 <think>，必须有 </think>
        has_think = '<think>' in response.lower()
        has_think_end = '</think>' in response.lower()
        if has_think != has_think_end:
            return False
        
        return True
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """随机采样一个 batch"""
        if len(self.samples) < batch_size:
            # 如果样本不够，允许重复采样
            return random.choices(self.samples, k=batch_size)
        else:
            return random.sample(self.samples, batch_size)
    
    def __len__(self):
        return len(self.samples)


class FormatAnchor:
    """
    格式校准器
    在 GRPO 训练过程中定期进行 SFT 微调以维持格式能力
    """
    
    def __init__(
        self,
        config: FormatAnchorConfig,
        tokenizer,
        data_file: str,
    ):
        """
        Args:
            config: 格式校准配置
            tokenizer: tokenizer
            data_file: 预处理好的格式校准数据文件路径
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # 加载校准数据集
        self.dataset = FormatAnchoringDataset(
            data_file=data_file,
            tokenizer=tokenizer,
            max_samples=None,  # 加载所有样本
        )
        
        # 统计信息
        self.total_anchors = 0
        self.anchor_history = []
    
    def should_anchor(self, global_step: int) -> bool:
        """判断是否应该进行校准"""
        if global_step == 0:
            return False
        return global_step % self.config.frequency == 0
    
    def anchor(
        self,
        actor_module,
        optimizer,
        device='cuda'
    ) -> Dict[str, float]:
        """
        执行格式校准
        
        Args:
            actor_module: actor 模型
            optimizer: 优化器
            device: 设备
            
        Returns:
            Dict: 校准统计信息
        """
        if len(self.dataset) == 0:
            print("⚠️  警告: 没有可用的格式校准数据")
            return {'anchor_loss': 0.0, 'samples': 0}
        
        # 保存原始学习率
        original_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
        # 设置校准学习率（更小）
        anchor_lr = original_lrs[0] * self.config.lr_ratio
        for param_group in optimizer.param_groups:
            param_group['lr'] = anchor_lr
        
        # 设置为训练模式
        actor_module.train()
        
        total_loss = 0.0
        num_samples = 0
        
        if self.config.verbose:
            print(f"\n{'─'*60}")
            print(f"🔧 格式校准中... (LR: {anchor_lr:.2e})")
        
        for step in range(self.config.steps_per_anchor):
            # 采样一个 batch
            batch_samples = self.dataset.sample_batch(self.config.batch_size)
            
            # 准备数据
            batch_data = self._prepare_batch(batch_samples, device)
            
            # 计算 SFT loss
            loss = self._compute_sft_loss(actor_module, batch_data)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(actor_module.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_samples += len(batch_samples)
            
            if self.config.verbose:
                print(f"  Step {step+1}/{self.config.steps_per_anchor}: loss={loss.item():.4f}")
        
        # 恢复原始学习率
        for param_group, original_lr in zip(optimizer.param_groups, original_lrs):
            param_group['lr'] = original_lr
        
        avg_loss = total_loss / self.config.steps_per_anchor
        
        # 记录统计
        self.total_anchors += 1
        self.anchor_history.append({
            'step': self.total_anchors * self.config.frequency,
            'loss': avg_loss,
            'samples': num_samples
        })
        
        if self.config.verbose:
            print(f"  ✓ 校准完成: 平均 loss={avg_loss:.4f}, 样本数={num_samples}")
            print(f"{'─'*60}\n")
        
        return {
            'anchor_loss': avg_loss,
            'anchor_samples': num_samples,
            'total_anchors': self.total_anchors
        }
    
    def _prepare_batch(self, batch_samples: List[Dict], device) -> Dict:
        """准备训练 batch"""
        prompts = [sample['prompt'] for sample in batch_samples]
        responses = [sample['response'] for sample in batch_samples]
        
        # 拼接 prompt + response
        full_texts = [p + r for p, r in zip(prompts, responses)]
        
        # Tokenize
        encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # 创建 labels（只在 response 部分计算 loss）
        labels = input_ids.clone()
        
        # 对于每个样本，mask 掉 prompt 部分
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            prompt_length = len(prompt_tokens)
            # Mask prompt 部分
            labels[i, :prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _compute_sft_loss(self, model, batch_data: Dict) -> torch.Tensor:
        """计算 SFT loss"""
        input_ids = batch_data['input_ids']
        labels = batch_data['labels']
        attention_mask = batch_data['attention_mask']
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 如果模型返回 loss，直接使用
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return outputs.loss
        
        # 否则手动计算
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算 cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if len(self.anchor_history) == 0:
            return {
                'total_anchors': 0,
                'avg_loss': 0.0,
                'latest_loss': 0.0
            }
        
        avg_loss = np.mean([h['loss'] for h in self.anchor_history])
        latest_loss = self.anchor_history[-1]['loss']
        
        return {
            'total_anchors': self.total_anchors,
            'avg_loss': avg_loss,
            'latest_loss': latest_loss,
            'history': self.anchor_history
        }


def integrate_format_anchoring(trainer_instance, config: FormatAnchorConfig, data_file: str):
    """
    将格式校准集成到 RayPPOTrainer 中
    
    Args:
        trainer_instance: RayPPOTrainer 实例
        config: 格式校准配置
        data_file: 预处理好的格式校准数据文件路径
        
    使用示例:
        from verl.utils.format_anchoring import integrate_format_anchoring, FormatAnchorConfig
        
        # 配置
        anchor_config = FormatAnchorConfig(
            frequency=50,
            steps_per_anchor=2,
            lr_ratio=0.1,
            batch_size=16,
        )
        
        # 预处理好的数据文件（使用 preprocess_format_anchor.sh 生成）
        data_file = 'data/format_anchor/deepseek-r1-distill-qwen/format_anchor.parquet'
        
        # 集成
        trainer = RayPPOTrainer(config)
        integrate_format_anchoring(trainer, anchor_config, data_file)
        
        # 正常训练，自动包含格式校准
        trainer.fit()
    """
    # 创建格式校准器
    format_anchor = FormatAnchor(
        config=config,
        tokenizer=trainer_instance.tokenizer,
        data_file=data_file
    )
    
    # 保存到 trainer 实例
    trainer_instance.format_anchor = format_anchor
    
    return trainer_instance

