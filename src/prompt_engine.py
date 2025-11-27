"""
提示词生成引擎

Author: Kuroneko
Inputs: 用户任务描述、数据集信息、历史高分架构日志
Outputs: 结构化的 Few-shot Prompt（JSON/YAML 格式）
Function: 根据任务需求和历史经验构建高质量的提示词，引导 LLM 生成优质的神经网络架构代码
"""

import json
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime


class PromptEngine:
    """
    提示词生成引擎
    
    负责构建结构化的提示词，结合任务描述和历史经验引导 LLM 生成代码
    """
    
    def __init__(self, logs_dir: str = "./logs", top_k_examples: int = 3):
        """
        初始化提示词引擎
        
        Args:
            logs_dir: 日志目录路径
            top_k_examples: 使用前 K 个高分架构作为示例
        """
        self.logs_dir = Path(logs_dir).resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_examples = top_k_examples
        
        # 历史架构日志文件
        self.history_file = self.logs_dir / "architecture_history.json"
        self._init_history()
    
    def _init_history(self):
        """初始化历史记录文件"""
        if not self.history_file.exists():
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def _load_top_architectures(self) -> List[Dict[str, Any]]:
        """
        加载历史最佳架构
        
        Returns:
            按性能排序的架构列表
        """
        if not self.history_file.exists():
            return []
        
        with open(self.history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 按验证精度降序排序
        sorted_history = sorted(
            history,
            key=lambda x: x.get('val_accuracy', 0.0),
            reverse=True
        )
        
        return sorted_history[:self.top_k_examples]
    
    def build_prompt(
        self,
        task_description: str,
        dataset_info: Dict[str, Any],
        use_few_shot: bool = True
    ) -> str:
        """
        构建完整的提示词
        
        Args:
            task_description: 用户任务描述
            dataset_info: 数据集信息字典
            use_few_shot: 是否使用 Few-shot 示例
            
        Returns:
            格式化的提示词字符串
        """
        # 基础系统提示
        system_prompt = self._build_system_prompt()
        
        # 数据集上下文
        dataset_context = self._build_dataset_context(dataset_info)
        
        # Few-shot 示例
        few_shot_examples = ""
        if use_few_shot:
            few_shot_examples = self._build_few_shot_examples()
        
        # 任务描述
        task_prompt = self._build_task_prompt(task_description)
        
        # 组合完整提示词
        full_prompt = f"""{system_prompt}

{dataset_context}

{few_shot_examples}

{task_prompt}

请生成完整的 PyTorch 模型代码，包含所有必要的导入语句和类定义。
"""
        
        return full_prompt
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一位专业的深度学习架构设计专家，擅长使用 PyTorch 构建高性能的神经网络模型。

你的任务是根据给定的任务描述和数据集信息，设计并生成一个完整的 PyTorch 神经网络模型代码。

代码要求：
1. 必须是完整可运行的 Python 代码
2. 包含所有必要的 import 语句
3. 定义一个继承自 nn.Module 的模型类，类名必须为 GeneratedModel
4. 模型必须实现 __init__ 和 forward 方法
5. __init__ 中必须定义实际的网络层（如 nn.Conv2d, nn.Linear 等）
6. forward 方法中必须使用定义的网络层进行前向传播
7. 代码风格清晰，添加适当的注释
8. 考虑模型的效率和性能

代码示例（针对图像分类，输入 3x32x32）：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratedModel(nn.Module):
    def __init__(self, num_classes):
        super(GeneratedModel, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 特征提取
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

请严格按照上述格式和要求生成代码。"""
    
    def _build_dataset_context(self, dataset_info: Dict[str, Any]) -> str:
        """构建数据集上下文"""
        input_shape = dataset_info.get('input_shape', 'Unknown')
        num_classes = dataset_info.get('num_classes', 'Unknown')
        dataset_name = dataset_info.get('dataset_name', 'Unknown')
        
        return f"""## 数据集信息

- 数据集名称: {dataset_name.upper()}
- 输入形状: {input_shape}
- 类别数量: {num_classes}
- 批次大小: {dataset_info.get('batch_size', 'Unknown')}
"""
    
    def _build_few_shot_examples(self) -> str:
        """构建 Few-shot 示例"""
        top_archs = self._load_top_architectures()
        
        if not top_archs:
            return "## 历史最佳架构\n\n暂无历史记录，这是第一次生成。请设计一个创新且高效的架构。\n"
        
        examples_text = "## 历史最佳架构（供参考）\n\n"
        examples_text += "以下是历史上表现最好的架构，你可以参考这些设计思路：\n\n"
        
        for i, arch in enumerate(top_archs, 1):
            examples_text += f"### 示例 {i}（验证精度: {arch.get('val_accuracy', 0.0):.2%}）\n\n"
            examples_text += f"**描述**: {arch.get('description', 'N/A')}\n\n"
            
            # 如果有代码片段，展示关键部分
            if 'code_snippet' in arch:
                examples_text += f"**关键代码**:\n```python\n{arch['code_snippet']}\n```\n\n"
            
            examples_text += f"**性能指标**:\n"
            examples_text += f"- 训练精度: {arch.get('train_accuracy', 0.0):.2%}\n"
            examples_text += f"- 验证精度: {arch.get('val_accuracy', 0.0):.2%}\n"
            examples_text += f"- 训练损失: {arch.get('train_loss', 0.0):.4f}\n\n"
        
        return examples_text
    
    def _build_task_prompt(self, task_description: str) -> str:
        """构建任务提示词"""
        return f"""## 当前任务

{task_description}

请基于以上信息和历史经验，设计一个创新且高效的神经网络架构。
你的设计应该：
1. 充分利用历史最佳架构的设计思路
2. 针对当前任务进行优化和创新
3. 保持代码的可读性和可维护性
4. 考虑计算效率和训练稳定性
"""
    
    def save_architecture_log(
        self,
        architecture_id: str,
        code: str,
        metrics: Dict[str, float],
        description: str = "",
        code_snippet: Optional[str] = None
    ):
        """
        保存架构日志到历史记录
        
        Args:
            architecture_id: 架构唯一标识
            code: 完整代码
            metrics: 性能指标字典
            description: 架构描述
            code_snippet: 关键代码片段（可选）
        """
        # 加载现有历史
        with open(self.history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 创建新记录
        log_entry = {
            "architecture_id": architecture_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "train_accuracy": metrics.get('train_accuracy', 0.0),
            "val_accuracy": metrics.get('val_accuracy', 0.0),
            "train_loss": metrics.get('train_loss', float('inf')),
            "val_loss": metrics.get('val_loss', float('inf')),
            "code_snippet": code_snippet or code[:500]  # 保存前500字符作为片段
        }
        
        # 添加并保存
        history.append(log_entry)
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        # 同时保存完整代码到独立文件
        code_file = self.logs_dir / f"{architecture_id}.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
    
    def export_prompt_to_yaml(self, prompt: str, output_file: str):
        """
        将提示词导出为 YAML 格式
        
        Args:
            prompt: 提示词字符串
            output_file: 输出文件路径
        """
        prompt_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(prompt_data, f, allow_unicode=True, default_flow_style=False)
