"""
训练执行器

Author: Kuroneko
Inputs: 验证通过的 PyTorch 代码、数据加载器、训练配置
Outputs: 训练日志、性能指标、模型权重文件
Function: 在隔离的沙盒环境中执行模型训练，捕获日志并记录性能指标
"""

import os
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()


class ExecutionResult:
    """训练执行结果"""
    
    def __init__(self):
        self.success: bool = False
        self.metrics: Dict[str, float] = {}
        self.logs: str = ""
        self.errors: str = ""
        self.model_path: Optional[str] = None
        self.execution_time: float = 0.0


class Executor:
    """
    训练执行器
    
    负责在受控环境中执行 LLM 生成的模型训练代码
    """
    
    def __init__(
        self,
        logs_dir: str = "./logs",
        models_save_dir: str = "./logs/models",
        device: Optional[str] = None
    ):
        """
        初始化执行器
        
        Args:
            logs_dir: 日志目录
            models_save_dir: 模型保存目录
            device: 训练设备（cuda/cpu），None 则自动检测
        """
        self.logs_dir = Path(logs_dir).resolve()
        self.models_save_dir = Path(models_save_dir).resolve()
        
        # 创建目录
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置设备（RTX 5070 暂不支持，强制使用 CPU）
        if device is None:
            # 强制使用 CPU，避免 RTX 5070 兼容性问题
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        console.print(f"[cyan]执行器初始化完成，使用设备: {self.device}[/cyan]")
    
    def execute_training(
        self,
        code: str,
        train_loader,
        test_loader,
        num_classes: int,
        epochs: int = 10,
        learning_rate: float = 0.001,
        architecture_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        执行模型训练
        
        Args:
            code: 模型代码
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            num_classes: 类别数量
            epochs: 训练轮数
            learning_rate: 学习率
            architecture_id: 架构 ID
            
        Returns:
            ExecutionResult 对象
        """
        result = ExecutionResult()
        
        if architecture_id is None:
            architecture_id = f"arch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        console.print(f"\n[bold cyan]开始执行训练: {architecture_id}[/bold cyan]")
        
        start_time = datetime.now()
        
        # 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # 步骤 1: 动态加载模型
            console.print("[cyan]正在加载模型...[/cyan]")
            model = self._load_model_from_code(code, num_classes)
            model = model.to(self.device)
            
            # 检查模型是否有参数
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                raise ValueError(
                    "模型没有可训练参数！生成的代码可能有误。\n"
                    "请确保模型定义了网络层（如 nn.Conv2d, nn.Linear 等）"
                )
            
            console.print(f"[green][OK] 模型加载成功（参数数量: {param_count:,}）[/green]")
            self._print_model_summary(model)
            
            # 步骤 2: 设置优化器和损失函数
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # 步骤 3: 执行训练
            console.print(f"\n[cyan]开始训练 ({epochs} epochs)...[/cyan]")
            
            best_val_acc = 0.0
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                
                epoch_task = progress.add_task("[cyan]训练进度", total=epochs)
                
                for epoch in range(epochs):
                    # 训练阶段
                    train_loss, train_acc = self._train_epoch(
                        model, train_loader, criterion, optimizer
                    )
                    
                    # 验证阶段
                    val_loss, val_acc = self._validate_epoch(
                        model, test_loader, criterion
                    )
                    
                    # 更新进度
                    progress.update(
                        epoch_task,
                        advance=1,
                        description=f"[cyan]Epoch {epoch+1}/{epochs} - "
                                   f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
                                   f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}"
                    )
                    
                    # 保存最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        model_save_path = self.models_save_dir / f"{architecture_id}_best.pth"
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_accuracy': val_acc,
                            'val_loss': val_loss,
                        }, model_save_path)
                        result.model_path = str(model_save_path)
                    
                    # 记录到日志
                    log_line = (
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n"
                    )
                    stdout_capture.write(log_line)
            
            # 步骤 4: 收集结果
            result.success = True
            result.metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'best_val_accuracy': best_val_acc,
            }
            
            console.print(f"\n[bold green][OK] 训练完成！[/bold green]")
            console.print(f"  · 最佳验证精度: {best_val_acc:.2%}")
            console.print(f"  · 模型已保存: {result.model_path}")
            
        except Exception as e:
            result.success = False
            error_msg = f"训练执行失败: {str(e)}"
            stderr_capture.write(error_msg)
            console.print(f"[red][FAIL] {error_msg}[/red]")
            import traceback
            traceback.print_exc()
        
        finally:
            # 记录执行时间
            end_time = datetime.now()
            result.execution_time = (end_time - start_time).total_seconds()
            
            # 保存日志
            result.logs = stdout_capture.getvalue()
            result.errors = stderr_capture.getvalue()
            
            self._save_execution_log(architecture_id, result)
        
        return result
    
    def _load_model_from_code(self, code: str, num_classes: int) -> nn.Module:
        """
        从代码字符串动态加载模型
        
        Args:
            code: 模型代码
            num_classes: 类别数量
            
        Returns:
            模型实例
        """
        # 创建一个新的命名空间来执行代码
        namespace = {
            'torch': torch,
            'nn': nn,
            'F': torch.nn.functional,
        }
        
        # 执行代码
        exec(code, namespace)
        
        # 获取模型类
        model_class = namespace.get('GeneratedModel')
        
        if model_class is None:
            raise ValueError("未找到 'GeneratedModel' 类")
        
        # 实例化模型
        model = model_class(num_classes=num_classes)
        
        return model
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader,
        criterion,
        optimizer
    ) -> Tuple[float, float]:
        """
        训练一个 epoch
        
        Returns:
            (平均损失, 准确率)
        """
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self,
        model: nn.Module,
        test_loader,
        criterion
    ) -> Tuple[float, float]:
        """
        验证一个 epoch
        
        Returns:
            (平均损失, 准确率)
        """
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _print_model_summary(self, model: nn.Module):
        """打印模型摘要"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        console.print(f"  · 总参数量: {total_params:,}")
        console.print(f"  · 可训练参数: {trainable_params:,}")
    
    def _save_execution_log(self, architecture_id: str, result: ExecutionResult):
        """保存执行日志"""
        log_file = self.logs_dir / f"{architecture_id}_execution.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Architecture ID: {architecture_id}\n")
            f.write(f"Execution Time: {result.execution_time:.2f}s\n")
            f.write(f"Success: {result.success}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"METRICS\n")
            f.write(f"{'='*60}\n")
            for key, value in result.metrics.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"LOGS\n")
            f.write(f"{'='*60}\n")
            f.write(result.logs)
            
            if result.errors:
                f.write(f"\n{'='*60}\n")
                f.write(f"ERRORS\n")
                f.write(f"{'='*60}\n")
                f.write(result.errors)
