"""
架构代码校验器

Author: Kuroneko
Inputs: LLM 生成的 PyTorch 代码文本
Outputs: 校验结果（通过/失败）、错误信息、清理后的代码
Function: 使用 AST 和 Pydantic 对生成的代码进行严格的格式和语法校验，确保代码可安全执行
"""

import ast
import re
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from rich.console import Console

console = Console()


class CodeValidationResult(BaseModel):
    """代码验证结果模型"""
    is_valid: bool = Field(description="代码是否有效")
    cleaned_code: Optional[str] = Field(default=None, description="清理后的代码")
    errors: List[str] = Field(default_factory=list, description="错误列表")
    warnings: List[str] = Field(default_factory=list, description="警告列表")
    has_model_class: bool = Field(default=False, description="是否包含模型类")
    model_class_name: Optional[str] = Field(default=None, description="模型类名称")


class Validator:
    """
    架构代码校验器
    
    负责对 LLM 生成的 PyTorch 代码进行全面校验
    """
    
    REQUIRED_MODEL_CLASS = "GeneratedModel"
    REQUIRED_METHODS = ["__init__", "forward"]
    DANGEROUS_MODULES = ["os", "subprocess", "sys", "eval", "exec"]
    
    def __init__(self, strict_mode: bool = True):
        """
        初始化校验器
        
        Args:
            strict_mode: 是否使用严格模式（更严格的检查）
        """
        self.strict_mode = strict_mode
    
    def validate(self, code: str) -> CodeValidationResult:
        """
        执行完整的代码校验流程
        
        Args:
            code: 待校验的代码字符串
            
        Returns:
            CodeValidationResult 对象
        """
        result = CodeValidationResult(is_valid=False)
        
        # 步骤 1: 提取代码块（如果是 Markdown 格式）
        cleaned_code = self._extract_code_from_markdown(code)
        result.cleaned_code = cleaned_code
        
        # 步骤 2: 语法检查
        syntax_valid, syntax_errors = self._check_syntax(cleaned_code)
        if not syntax_valid:
            result.errors.extend(syntax_errors)
            return result
        
        # 步骤 3: 安全检查
        security_valid, security_errors = self._check_security(cleaned_code)
        if not security_valid:
            result.errors.extend(security_errors)
            if self.strict_mode:
                return result
            else:
                result.warnings.extend(security_errors)
        
        # 步骤 4: 结构检查
        structure_valid, structure_info = self._check_structure(cleaned_code)
        if not structure_valid:
            result.errors.extend(structure_info.get('errors', []))
            return result
        
        result.has_model_class = structure_info.get('has_model_class', False)
        result.model_class_name = structure_info.get('model_class_name')
        result.warnings.extend(structure_info.get('warnings', []))
        
        # 步骤 5: PyTorch 特定检查
        pytorch_valid, pytorch_errors = self._check_pytorch_specifics(cleaned_code)
        if not pytorch_valid:
            result.errors.extend(pytorch_errors)
            return result
        
        # 所有检查通过
        result.is_valid = True
        
        return result
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """
        从 Markdown 格式中提取 Python 代码
        
        Args:
            text: 可能包含 Markdown 的文本
            
        Returns:
            提取的纯 Python 代码
        """
        # 匹配 ```python ... ``` 代码块
        pattern = r'```python\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # 返回第一个代码块
            return matches[0].strip()
        
        # 如果没有 Markdown 格式，返回原文本
        return text.strip()
    
    def _check_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """
        检查 Python 语法
        
        Args:
            code: Python 代码
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            errors.append(f"语法错误 (行 {e.lineno}): {e.msg}")
            return False, errors
        except Exception as e:
            errors.append(f"解析错误: {str(e)}")
            return False, errors
    
    def _check_security(self, code: str) -> Tuple[bool, List[str]]:
        """
        检查代码安全性
        
        Args:
            code: Python 代码
            
        Returns:
            (是否安全, 错误列表)
        """
        errors = []
        
        # 检查危险的内置函数
        dangerous_builtins = ['eval', 'exec', 'compile', '__import__']
        for builtin in dangerous_builtins:
            if re.search(rf'\b{builtin}\s*\(', code):
                errors.append(f"安全警告: 检测到危险函数 '{builtin}'")
        
        # 检查危险的模块导入
        for module in self.DANGEROUS_MODULES:
            if re.search(rf'import\s+{module}', code) or re.search(rf'from\s+{module}', code):
                errors.append(f"安全警告: 检测到危险模块导入 '{module}'")
        
        # 检查文件操作
        if re.search(r'\bopen\s*\(', code):
            errors.append("安全警告: 检测到文件操作")
        
        is_safe = len(errors) == 0
        return is_safe, errors
    
    def _check_structure(self, code: str) -> Tuple[bool, Dict[str, Any]]:
        """
        检查代码结构
        
        Args:
            code: Python 代码
            
        Returns:
            (是否有效, 结构信息字典)
        """
        info = {
            'errors': [],
            'warnings': [],
            'has_model_class': False,
            'model_class_name': None
        }
        
        try:
            tree = ast.parse(code)
        except:
            info['errors'].append("无法解析代码结构")
            return False, info
        
        # 查找类定义
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            info['errors'].append("未找到任何类定义")
            return False, info
        
        # 查找目标模型类
        model_class = None
        for cls in classes:
            if cls.name == self.REQUIRED_MODEL_CLASS:
                model_class = cls
                break
        
        if not model_class:
            # 检查是否有其他继承 nn.Module 的类
            nn_module_classes = []
            for cls in classes:
                for base in cls.bases:
                    if isinstance(base, ast.Attribute):
                        if base.attr == 'Module':
                            nn_module_classes.append(cls.name)
                    elif isinstance(base, ast.Name):
                        if base.id == 'Module':
                            nn_module_classes.append(cls.name)
            
            if nn_module_classes:
                info['warnings'].append(
                    f"未找到 '{self.REQUIRED_MODEL_CLASS}' 类，但找到其他模型类: {nn_module_classes}"
                )
                # 使用第一个找到的类
                model_class = next(cls for cls in classes if cls.name == nn_module_classes[0])
            else:
                info['errors'].append(f"未找到必需的 '{self.REQUIRED_MODEL_CLASS}' 类")
                return False, info
        
        info['has_model_class'] = True
        info['model_class_name'] = model_class.name
        
        # 检查必需的方法
        methods = [node.name for node in model_class.body if isinstance(node, ast.FunctionDef)]
        
        for required_method in self.REQUIRED_METHODS:
            if required_method not in methods:
                info['errors'].append(f"模型类缺少必需的方法: '{required_method}'")
        
        if info['errors']:
            return False, info
        
        return True, info
    
    def _check_pytorch_specifics(self, code: str) -> Tuple[bool, List[str]]:
        """
        检查 PyTorch 特定的要求
        
        Args:
            code: Python 代码
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        warnings = []
        
        # 检查是否导入了 PyTorch
        if 'import torch' not in code:
            errors.append("未导入 'torch' 模块")
        
        # 检查是否导入了 nn
        if 'torch.nn' not in code and 'import torch.nn' not in code:
            errors.append("未导入 'torch.nn' 模块")
        
        # 检查是否继承了 nn.Module
        if 'nn.Module' not in code and 'Module' not in code:
            errors.append("模型类未继承 'nn.Module'")
        
        # 检查是否定义了网络层（确保模型有参数）
        common_layers = ['nn.Conv2d', 'nn.Linear', 'nn.BatchNorm', 'nn.Embedding', 
                        'Conv2d', 'Linear', 'BatchNorm', 'Embedding']
        has_layers = any(layer in code for layer in common_layers)
        
        if not has_layers:
            errors.append(
                "模型中未找到常见的网络层（如 nn.Conv2d, nn.Linear 等）。"
                "模型必须包含至少一个可训练的网络层。"
            )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def quick_validate(self, code: str) -> bool:
        """
        快速验证（仅检查基本语法）
        
        Args:
            code: Python 代码
            
        Returns:
            是否通过基本验证
        """
        try:
            cleaned_code = self._extract_code_from_markdown(code)
            ast.parse(cleaned_code)
            return True
        except:
            return False
    
    def format_validation_report(self, result: CodeValidationResult) -> str:
        """
        格式化验证报告
        
        Args:
            result: 验证结果
            
        Returns:
            格式化的报告字符串
        """
        report = "=" * 60 + "\n"
        report += "代码验证报告\n"
        report += "=" * 60 + "\n\n"
        
        if result.is_valid:
            report += "[OK] 验证状态: 通过\n"
        else:
            report += "[FAIL] 验证状态: 失败\n"
        
        if result.has_model_class:
            report += f"[OK] 模型类: {result.model_class_name}\n"
        
        if result.errors:
            report += "\n错误列表:\n"
            for i, error in enumerate(result.errors, 1):
                report += f"  {i}. {error}\n"
        
        if result.warnings:
            report += "\n警告列表:\n"
            for i, warning in enumerate(result.warnings, 1):
                report += f"  {i}. {warning}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report
