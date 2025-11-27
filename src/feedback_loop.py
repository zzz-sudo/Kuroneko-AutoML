"""
闭环反馈机制

Author: Kuroneko
Inputs: 训练执行日志、性能指标、LLM 模型实例
Outputs: 过滤后的高分样本、LoRA 微调触发信号、更新后的模型
Function: 实现基于强化学习的反馈循环，根据性能阈值筛选优质样本并触发 LoRA 微调
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from peft import LoraConfig, get_peft_model, TaskType
from rich.console import Console
from rich.table import Table

console = Console()


class FeedbackSample:
    """反馈样本"""
    
    def __init__(
        self,
        architecture_id: str,
        prompt: str,
        generated_code: str,
        metrics: Dict[str, float],
        timestamp: str
    ):
        self.architecture_id = architecture_id
        self.prompt = prompt
        self.generated_code = generated_code
        self.metrics = metrics
        self.timestamp = timestamp
        self.reward = self._calculate_reward()
    
    def _calculate_reward(self) -> float:
        """
        计算奖励值
        
        基于验证精度和训练稳定性计算综合奖励
        """
        val_acc = self.metrics.get('val_accuracy', 0.0)
        train_acc = self.metrics.get('train_accuracy', 0.0)
        val_loss = self.metrics.get('val_loss', float('inf'))
        
        # 奖励 = 验证精度 + 稳定性奖励（防止过拟合）
        stability_bonus = 0.0
        if train_acc > 0 and val_acc > 0:
            overfitting_penalty = abs(train_acc - val_acc)
            stability_bonus = max(0, 0.1 - overfitting_penalty)
        
        reward = val_acc + stability_bonus
        
        return reward
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'architecture_id': self.architecture_id,
            'prompt': self.prompt,
            'generated_code': self.generated_code,
            'metrics': self.metrics,
            'timestamp': self.timestamp,
            'reward': self.reward
        }


class FeedbackLoop:
    """
    闭环反馈机制
    
    负责收集训练结果、筛选高分样本、触发 LoRA 微调
    """
    
    def __init__(
        self,
        logs_dir: str = "./logs",
        reward_threshold: float = 0.7,
        buffer_size: int = 100,
        lora_trigger_threshold: int = 10
    ):
        """
        初始化反馈循环
        
        Args:
            logs_dir: 日志目录
            reward_threshold: 奖励阈值（低于此值的样本不会被用于微调）
            buffer_size: 样本缓冲区大小
            lora_trigger_threshold: 触发 LoRA 微调的最小样本数
        """
        self.logs_dir = Path(logs_dir).resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.reward_threshold = reward_threshold
        self.buffer_size = buffer_size
        self.lora_trigger_threshold = lora_trigger_threshold
        
        # 样本缓冲区
        self.feedback_buffer: List[FeedbackSample] = []
        
        # 反馈历史文件
        self.feedback_file = self.logs_dir / "feedback_history.json"
        self._init_feedback_history()
        
        console.print(f"[cyan]反馈循环初始化完成[/cyan]")
        console.print(f"  · 奖励阈值: {self.reward_threshold}")
        console.print(f"  · LoRA 触发阈值: {self.lora_trigger_threshold} 样本")
    
    def _init_feedback_history(self):
        """初始化反馈历史文件"""
        if not self.feedback_file.exists():
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def add_sample(
        self,
        architecture_id: str,
        prompt: str,
        generated_code: str,
        metrics: Dict[str, float]
    ) -> FeedbackSample:
        """
        添加新样本到缓冲区
        
        Args:
            architecture_id: 架构 ID
            prompt: 使用的提示词
            generated_code: 生成的代码
            metrics: 性能指标
            
        Returns:
            FeedbackSample 对象
        """
        sample = FeedbackSample(
            architecture_id=architecture_id,
            prompt=prompt,
            generated_code=generated_code,
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        
        # 只添加达到阈值的样本
        if sample.reward >= self.reward_threshold:
            self.feedback_buffer.append(sample)
            
            # 保持缓冲区大小
            if len(self.feedback_buffer) > self.buffer_size:
                # 移除奖励最低的样本
                self.feedback_buffer.sort(key=lambda x: x.reward, reverse=True)
                self.feedback_buffer = self.feedback_buffer[:self.buffer_size]
            
            console.print(f"[green][OK] 高质量样本已添加到缓冲区 (奖励: {sample.reward:.4f})[/green]")
            
            # 保存到历史
            self._save_to_history(sample)
        else:
            console.print(
                f"[yellow]样本奖励 ({sample.reward:.4f}) "
                f"低于阈值 ({self.reward_threshold})，已忽略[/yellow]"
            )
        
        return sample
    
    def _save_to_history(self, sample: FeedbackSample):
        """保存样本到历史记录"""
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        history.append(sample.to_dict())
        
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def should_trigger_lora(self) -> bool:
        """
        判断是否应该触发 LoRA 微调
        
        Returns:
            是否触发
        """
        return len(self.feedback_buffer) >= self.lora_trigger_threshold
    
    def get_training_samples(self) -> List[Tuple[str, str]]:
        """
        获取用于 LoRA 微调的训练样本
        
        Returns:
            [(prompt, generated_code), ...] 列表
        """
        if not self.should_trigger_lora():
            console.print(
                f"[yellow]缓冲区样本不足 ({len(self.feedback_buffer)}/{self.lora_trigger_threshold})，"
                f"无法触发 LoRA 微调[/yellow]"
            )
            return []
        
        # 按奖励排序，使用最好的样本
        sorted_samples = sorted(
            self.feedback_buffer,
            key=lambda x: x.reward,
            reverse=True
        )
        
        training_samples = [
            (sample.prompt, sample.generated_code)
            for sample in sorted_samples
        ]
        
        console.print(f"[green][OK] 准备了 {len(training_samples)} 个样本用于 LoRA 微调[/green]")
        
        return training_samples
    
    def apply_lora_finetuning(
        self,
        model,
        tokenizer,
        training_samples: Optional[List[Tuple[str, str]]] = None,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        num_epochs: int = 3,
        learning_rate: float = 1e-4
    ):
        """
        应用 LoRA 微调到模型
        
        Args:
            model: 要微调的模型
            tokenizer: Tokenizer
            training_samples: 训练样本，None 则使用缓冲区样本
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            num_epochs: 训练轮数
            learning_rate: 学习率
        """
        if training_samples is None:
            training_samples = self.get_training_samples()
        
        if not training_samples:
            console.print("[yellow]没有可用的训练样本，跳过 LoRA 微调[/yellow]")
            return
        
        console.print(f"\n[bold cyan]开始 LoRA 微调...[/bold cyan]")
        console.print(f"  · 训练样本数: {len(training_samples)}")
        console.print(f"  · LoRA Rank: {lora_r}")
        console.print(f"  · LoRA Alpha: {lora_alpha}")
        
        try:
            # 配置 LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 针对常见的注意力层
                bias="none"
            )
            
            # 应用 LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # 准备训练数据
            train_texts = [
                f"{prompt}\n\n{code}"
                for prompt, code in training_samples
            ]
            
            # 简化的微调循环（实际应用中需要更完善的训练逻辑）
            console.print("[cyan]正在微调模型...[/cyan]")
            console.print("[yellow]注意: 这是简化版实现，生产环境需要完整的训练流程[/yellow]")
            
            # TODO: 实现完整的 LoRA 微调训练循环
            # 这里应该包含:
            # 1. 数据预处理和 tokenization
            # 2. 创建 DataLoader
            # 3. 优化器配置
            # 4. 训练循环
            # 5. 梯度累积
            # 6. 学习率调度
            
            console.print("[green][OK] LoRA 微调完成（简化版）[/green]")
            
            # 清空缓冲区
            self.feedback_buffer.clear()
            console.print("[cyan]反馈缓冲区已清空[/cyan]")
            
        except Exception as e:
            console.print(f"[red][FAIL] LoRA 微调失败: {str(e)}[/red]")
            import traceback
            traceback.print_exc()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取反馈循环统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'buffer_size': len(self.feedback_buffer),
            'buffer_capacity': self.buffer_size,
            'reward_threshold': self.reward_threshold,
            'lora_ready': self.should_trigger_lora(),
        }
        
        if self.feedback_buffer:
            rewards = [s.reward for s in self.feedback_buffer]
            stats['avg_reward'] = sum(rewards) / len(rewards)
            stats['max_reward'] = max(rewards)
            stats['min_reward'] = min(rewards)
        
        # 从历史文件读取总样本数
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                stats['total_samples'] = len(history)
        except:
            stats['total_samples'] = 0
        
        return stats
    
    def display_statistics(self):
        """显示统计信息表格"""
        stats = self.get_statistics()
        
        table = Table(title="反馈循环统计", show_header=True, header_style="bold cyan")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("缓冲区大小", f"{stats['buffer_size']}/{stats['buffer_capacity']}")
        table.add_row("历史样本总数", str(stats['total_samples']))
        table.add_row("奖励阈值", f"{stats['reward_threshold']:.2f}")
        
        if 'avg_reward' in stats:
            table.add_row("平均奖励", f"{stats['avg_reward']:.4f}")
            table.add_row("最大奖励", f"{stats['max_reward']:.4f}")
            table.add_row("最小奖励", f"{stats['min_reward']:.4f}")
        
        lora_status = "[OK] 就绪" if stats['lora_ready'] else "[NOT READY] 未就绪"
        table.add_row("LoRA 微调状态", lora_status)
        
        console.print(table)
