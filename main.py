"""
Kuroneko-AutoML 主入口文件

Author: Kuroneko
Inputs: 用户通过 CLI 输入的任务描述和配置参数
Outputs: 训练好的模型、性能报告、生成的架构代码
Function: 整合所有模块，提供友好的命令行交互界面，实现完整的 AutoML 闭环流程
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model_manager import ModelManager
from src.data_manager import DataManager
from src.prompt_engine import PromptEngine
from src.validator import Validator
from src.executor import Executor
from src.feedback_loop import FeedbackLoop

console = Console()


class KuronekoAutoML:
    """
    Kuroneko-AutoML 主控制器
    
    整合所有模块，实现完整的 AutoML 流程
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 AutoML 系统
        
        Args:
            config_path: 配置文件路径（YAML）
        """
        self.config = self._load_config(config_path)
        
        # 初始化各个管理器
        self.model_manager = None
        self.data_manager = None
        self.prompt_engine = None
        self.validator = None
        self.executor = None
        self.feedback_loop = None
        
        self.is_initialized = False
    
    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """加载配置文件"""
        default_config = {
            'model': {
                'name': 'deepseek-ai/deepseek-coder-7b-instruct-v1.5',
                'models_dir': './models',
                'use_quantization': True,
                'quantization_bits': 4
            },
            'data': {
                'dataset_name': 'cifar10',
                'data_dir': './data',
                'batch_size': 128,
                'num_workers': 4
            },
            'training': {
                'epochs': 10,
                'learning_rate': 0.001
            },
            'feedback': {
                'reward_threshold': 0.7,
                'lora_trigger_threshold': 10
            },
            'logs_dir': './logs'
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                # 合并配置
                default_config.update(user_config)
        
        return default_config
    
    def initialize(self):
        """初始化所有模块"""
        console.print(Panel.fit(
            "[bold cyan]Kuroneko-AutoML[/bold cyan]\n"
            "[dim]具备持续学习能力的自动化机器学习系统[/dim]",
            border_style="cyan"
        ))
        
        console.print("\n[bold cyan]>>> 正在初始化系统...[/bold cyan]\n")
        
        # 1. 初始化模型管理器
        console.print("[cyan]Step 1/6: 初始化大语言模型管理器...[/cyan]")
        self.model_manager = ModelManager(
            model_name=self.config['model']['name'],
            models_dir=self.config['model']['models_dir'],
            use_quantization=self.config['model']['use_quantization'],
            quantization_bits=self.config['model']['quantization_bits']
        )
        
        # 2. 初始化数据管理器
        console.print("\n[cyan]Step 2/6: 初始化数据集管理器...[/cyan]")
        self.data_manager = DataManager(
            dataset_name=self.config['data']['dataset_name'],
            data_dir=self.config['data']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers']
        )
        
        # 3. 初始化提示词引擎
        console.print("\n[cyan]Step 3/6: 初始化提示词生成引擎...[/cyan]")
        self.prompt_engine = PromptEngine(
            logs_dir=self.config['logs_dir']
        )
        console.print("[green][OK] 提示词引擎初始化完成[/green]")
        
        # 4. 初始化验证器
        console.print("\n[cyan]Step 4/6: 初始化代码验证器...[/cyan]")
        self.validator = Validator(strict_mode=True)
        console.print("[green][OK] 验证器初始化完成[/green]")
        
        # 5. 初始化执行器
        console.print("\n[cyan]Step 5/6: 初始化训练执行器...[/cyan]")
        self.executor = Executor(
            logs_dir=self.config['logs_dir']
        )
        
        # 6. 初始化反馈循环
        console.print("\n[cyan]Step 6/6: 初始化反馈循环...[/cyan]")
        self.feedback_loop = FeedbackLoop(
            logs_dir=self.config['logs_dir'],
            reward_threshold=self.config['feedback']['reward_threshold'],
            lora_trigger_threshold=self.config['feedback']['lora_trigger_threshold']
        )
        
        console.print("\n[bold green][OK] 系统初始化完成！[/bold green]\n")
        self.is_initialized = True
    
    def load_resources(self):
        """加载模型和数据集"""
        if not self.is_initialized:
            raise RuntimeError("系统未初始化，请先调用 initialize()")
        
        console.print("[bold cyan]>>> 正在加载资源...[/bold cyan]\n")
        
        # 加载 LLM 模型
        console.print("[cyan]加载大语言模型...[/cyan]")
        self.model_manager.load_model()
        
        # 加载数据集
        console.print("\n[cyan]加载数据集...[/cyan]")
        self.data_manager.prepare_data()
        
        console.print("\n[bold green][OK] 所有资源加载完成！[/bold green]\n")
    
    def run_iteration(self, task_description: str, max_retries: int = 3):
        """
        运行一次 AutoML 迭代
        
        Args:
            task_description: 用户任务描述
            max_retries: 最大重试次数（如果代码生成或验证失败）
        """
        architecture_id = f"arch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        console.print(f"\n[bold cyan]>>> 开始新的 AutoML 迭代: {architecture_id}[/bold cyan]\n")
        
        # Step 1: 构建提示词
        console.print("[cyan]Step 1/5: 构建提示词...[/cyan]")
        dataset_info = self.data_manager.get_dataset_info()
        prompt = self.prompt_engine.build_prompt(
            task_description=task_description,
            dataset_info=dataset_info,
            use_few_shot=True
        )
        console.print("[green][OK] 提示词构建完成[/green]")
        
        # Step 2: 生成架构代码
        generated_code = None
        for attempt in range(max_retries):
            console.print(f"\n[cyan]Step 2/5: 生成神经网络架构代码 (尝试 {attempt + 1}/{max_retries})...[/cyan]")
            console.print("[dim]思考中...[/dim]")
            
            try:
                generated_code = self.model_manager.generate(
                    prompt=prompt,
                    max_new_tokens=2048,
                    temperature=0.7
                )
                console.print("[green][OK] 代码生成完成[/green]")
                
                # Step 3: 验证代码
                console.print(f"\n[cyan]Step 3/5: 验证生成的代码...[/cyan]")
                validation_result = self.validator.validate(generated_code)
                
                if validation_result.is_valid:
                    console.print("[green][OK] 代码验证通过[/green]")
                    generated_code = validation_result.cleaned_code
                    break
                else:
                    console.print("[red][FAIL] 代码验证失败[/red]")
                    console.print(self.validator.format_validation_report(validation_result))
                    
                    if attempt < max_retries - 1:
                        console.print("[yellow]将重新生成代码...[/yellow]")
                    else:
                        console.print("[red]已达到最大重试次数，跳过此次迭代[/red]")
                        return
                        
            except Exception as e:
                console.print(f"[red][FAIL] 生成失败: {str(e)}[/red]")
                if attempt < max_retries - 1:
                    console.print("[yellow]将重试...[/yellow]")
                else:
                    console.print("[red]已达到最大重试次数，跳过此次迭代[/red]")
                    return
        
        # Step 4: 执行训练
        console.print(f"\n[cyan]Step 4/5: 执行模型训练...[/cyan]")
        execution_result = self.executor.execute_training(
            code=generated_code,
            train_loader=self.data_manager.train_loader,
            test_loader=self.data_manager.test_loader,
            num_classes=self.data_manager.num_classes,
            epochs=self.config['training']['epochs'],
            learning_rate=self.config['training']['learning_rate'],
            architecture_id=architecture_id
        )
        
        if not execution_result.success:
            console.print("[red][FAIL] 训练执行失败[/red]")
            return
        
        # Step 5: 反馈和学习
        console.print(f"\n[cyan]Step 5/5: 反馈和学习...[/cyan]")
        
        # 保存架构日志
        self.prompt_engine.save_architecture_log(
            architecture_id=architecture_id,
            code=generated_code,
            metrics=execution_result.metrics,
            description=task_description
        )
        
        # 添加到反馈循环
        self.feedback_loop.add_sample(
            architecture_id=architecture_id,
            prompt=prompt,
            generated_code=generated_code,
            metrics=execution_result.metrics
        )
        
        # 显示本次迭代结果
        self._display_iteration_results(architecture_id, execution_result)
        
        # 检查是否触发 LoRA 微调
        if self.feedback_loop.should_trigger_lora():
            console.print("\n[bold yellow]>>> 检测到足够的高质量样本，准备触发 LoRA 微调...[/bold yellow]")
            
            should_finetune = Confirm.ask(
                "是否现在执行 LoRA 微调？（这将更新 LLM 模型）",
                default=False
            )
            
            if should_finetune:
                self.feedback_loop.apply_lora_finetuning(
                    model=self.model_manager.model,
                    tokenizer=self.model_manager.tokenizer
                )
    
    def _display_iteration_results(self, architecture_id: str, execution_result):
        """显示迭代结果"""
        console.print("\n" + "="*60)
        console.print("[bold green]迭代完成！[/bold green]")
        console.print("="*60 + "\n")
        
        # 创建结果表格
        table = Table(title=f"架构: {architecture_id}", show_header=True, header_style="bold cyan")
        table.add_column("指标", style="cyan", width=20)
        table.add_column("值", style="green", width=20)
        
        table.add_row("架构 ID", architecture_id)
        table.add_row("训练精度", f"{execution_result.metrics.get('train_accuracy', 0):.2%}")
        table.add_row("验证精度", f"{execution_result.metrics.get('val_accuracy', 0):.2%}")
        table.add_row("最佳验证精度", f"{execution_result.metrics.get('best_val_accuracy', 0):.2%}")
        table.add_row("训练损失", f"{execution_result.metrics.get('train_loss', 0):.4f}")
        table.add_row("验证损失", f"{execution_result.metrics.get('val_loss', 0):.4f}")
        table.add_row("训练时长", f"{execution_result.execution_time:.2f}s")
        
        if execution_result.model_path:
            table.add_row("模型路径", str(Path(execution_result.model_path).name))
        
        console.print(table)
        console.print()
    
    def interactive_mode(self):
        """交互式模式"""
        console.print("\n[bold cyan]进入交互模式[/bold cyan]")
        console.print("[dim]输入 'quit' 或 'exit' 退出，输入 'stats' 查看统计信息[/dim]\n")
        
        while True:
            try:
                user_input = Prompt.ask(
                    "\n[bold cyan]请输入您希望解决的 AI 任务[/bold cyan]\n"
                    "[dim](例如: '使用新的 ResNet 风格模块提升 CIFAR-10 的分类精度')[/dim]"
                )
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]退出系统...[/yellow]")
                    break
                
                if user_input.lower() == 'stats':
                    self.feedback_loop.display_statistics()
                    continue
                
                if not user_input.strip():
                    console.print("[yellow]请输入有效的任务描述[/yellow]")
                    continue
                
                # 运行迭代
                self.run_iteration(user_input)
                
                # 询问是否继续
                if not Confirm.ask("\n是否继续下一次迭代？", default=True):
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]检测到中断信号，退出系统...[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]错误: {str(e)}[/red]")
                import traceback
                traceback.print_exc()


def main():
    """主函数"""
    # 检查配置文件
    config_path = Path("./configs/config.yaml")
    if not config_path.exists():
        console.print("[yellow]未找到配置文件，使用默认配置[/yellow]")
        config_path = None
    
    # 创建 AutoML 实例
    automl = KuronekoAutoML(config_path=str(config_path) if config_path else None)
    
    try:
        # 初始化系统
        automl.initialize()
        
        # 加载资源
        automl.load_resources()
        
        # 进入交互模式
        automl.interactive_mode()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]程序被用户中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]发生错误: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        console.print("\n[cyan]感谢使用 Kuroneko-AutoML！[/cyan]")
        console.print("[dim]作者: Kuroneko[/dim]\n")


if __name__ == "__main__":
    main()
