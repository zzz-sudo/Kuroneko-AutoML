"""
大语言模型管理器

Author: Kuroneko
Inputs: 模型名称、本地路径、量化配置
Outputs: 加载的 LLM 模型实例和 Tokenizer
Function: 负责本地 LLM 权重的下载、加载和管理，支持量化加载以节省显存
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


class ModelManager:
    """
    大语言模型管理器
    
    负责管理本地 LLM 模型的下载、加载和缓存
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        models_dir: str = "./models",
        use_quantization: bool = True,
        quantization_bits: int = 4
    ):
        """
        初始化模型管理器
        
        Args:
            model_name: HuggingFace 模型名称
            models_dir: 本地模型存储目录
            use_quantization: 是否使用量化加载（节省显存）
            quantization_bits: 量化位数（4 或 8）
        """
        self.model_name = model_name
        self.models_dir = Path(models_dir).resolve()
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        
        # 创建模型目录
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 本地模型路径
        self.local_model_path = self.models_dir / model_name.replace("/", "_")
        
        self.model = None
        self.tokenizer = None
        
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        获取量化配置
        
        Returns:
            BitsAndBytesConfig 实例或 None
        """
        if not self.use_quantization:
            return None
            
        if self.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            console.print(f"[yellow]警告：不支持的量化位数 {self.quantization_bits}，将不使用量化[/yellow]")
            return None
    
    def _download_model(self) -> Path:
        """
        从 HuggingFace Hub 下载模型到本地
        
        Returns:
            本地模型路径
        """
        console.print(f"[cyan]正在从 HuggingFace 下载模型: {self.model_name}...[/cyan]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"下载 {self.model_name}", total=None)
                
                # 下载模型到本地目录
                local_path = snapshot_download(
                    repo_id=self.model_name,
                    local_dir=str(self.local_model_path),
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                
                progress.update(task, completed=True)
                
            console.print(f"[green][OK] 模型下载完成: {local_path}[/green]")
            return Path(local_path)
            
        except Exception as e:
            console.print(f"[red][FAIL] 模型下载失败: {str(e)}[/red]")
            raise
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        加载模型和 Tokenizer
        
        Returns:
            (model, tokenizer) 元组
        """
        # 检查本地是否已存在模型
        if not self.local_model_path.exists() or not any(self.local_model_path.iterdir()):
            console.print(f"[yellow]本地未找到模型，开始下载...[/yellow]")
            self._download_model()
        else:
            console.print(f"[green][OK] 发现本地模型: {self.local_model_path}[/green]")
        
        # 加载 Tokenizer
        console.print("[cyan]正在加载 Tokenizer...[/cyan]")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.local_model_path),
                trust_remote_code=True,
                use_fast=False
            )
            console.print("[green][OK] Tokenizer 加载成功[/green]")
        except Exception as e:
            console.print(f"[red][FAIL] Tokenizer 加载失败: {str(e)}[/red]")
            raise
        
        # 加载模型
        console.print("[cyan]正在加载模型...[/cyan]")
        quantization_config = self._get_quantization_config()
        
        try:
            # RTX 5070 暂不支持，强制使用 CPU
            device_map = "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.local_model_path),
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # CPU 使用 float32
                low_cpu_mem_usage=True
            )
            
            quant_info = f"{self.quantization_bits}-bit" if self.use_quantization else "Full Precision"
            console.print(f"[green][OK] 模型加载成功 ({quant_info}, Device: {device_map})[/green]")
            
        except Exception as e:
            console.print(f"[red][FAIL] 模型加载失败: {str(e)}[/red]")
            raise
        
        return self.model, self.tokenizer
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        使用模型生成文本
        
        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成 token 数量
            temperature: 温度参数
            top_p: nucleus sampling 参数
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 移动到模型设备
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入部分，只返回新生成的文本
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        info = {
            "model_name": self.model_name,
            "local_path": str(self.local_model_path),
            "quantization": f"{self.quantization_bits}-bit" if self.use_quantization else "None",
            "device": str(self.model.device) if self.model else "Not loaded",
            "loaded": self.model is not None
        }
        
        if self.model:
            info["parameters"] = sum(p.numel() for p in self.model.parameters())
            info["trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return info
