"""
数据集管理器

Author: Kuroneko
Inputs: 数据集名称、本地路径、预处理配置
Outputs: 处理后的 DataLoader 对象、数据集统计信息
Function: 负责数据集的下载、本地存储、加载和预处理，支持多种常见数据集
"""

import os
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class DataManager:
    """
    数据集管理器
    
    负责管理数据集的下载、加载、预处理和 DataLoader 创建
    """
    
    SUPPORTED_DATASETS = {
        "cifar10": torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
        "mnist": torchvision.datasets.MNIST,
        "fashionmnist": torchvision.datasets.FashionMNIST,
    }
    
    def __init__(
        self,
        dataset_name: str = "cifar10",
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        download: bool = True
    ):
        """
        初始化数据管理器
        
        Args:
            dataset_name: 数据集名称（cifar10, cifar100, mnist, fashionmnist）
            data_dir: 本地数据存储目录
            batch_size: 批次大小
            num_workers: 数据加载线程数
            download: 是否自动下载数据集
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        
        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证数据集名称
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"不支持的数据集: {dataset_name}. "
                f"支持的数据集: {list(self.SUPPORTED_DATASETS.keys())}"
            )
        
        self.train_loader = None
        self.test_loader = None
        self.num_classes = None
        self.input_shape = None
        
    def _get_transforms(self, is_training: bool = True) -> transforms.Compose:
        """
        获取数据预处理变换
        
        Args:
            is_training: 是否为训练集（决定是否使用数据增强）
            
        Returns:
            transforms.Compose 对象
        """
        if self.dataset_name in ["cifar10", "cifar100"]:
            if is_training:
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010]
                    )
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010]
                    )
                ])
        
        elif self.dataset_name in ["mnist", "fashionmnist"]:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        else:
            # 默认变换
            return transforms.Compose([
                transforms.ToTensor(),
            ])
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        准备数据集，返回训练和测试 DataLoader
        
        Returns:
            (train_loader, test_loader) 元组
        """
        dataset_class = self.SUPPORTED_DATASETS[self.dataset_name]
        
        # 检查本地是否已有数据
        dataset_path = self.data_dir / self.dataset_name
        if dataset_path.exists() and any(dataset_path.iterdir()):
            console.print(f"[green][OK] 发现本地数据集: {dataset_path}[/green]")
        elif self.download:
            console.print(f"[yellow]本地未找到数据集，开始下载 {self.dataset_name.upper()}...[/yellow]")
        
        # 加载训练集
        console.print("[cyan]正在加载训练集...[/cyan]")
        try:
            train_dataset = dataset_class(
                root=str(self.data_dir),
                train=True,
                download=self.download,
                transform=self._get_transforms(is_training=True)
            )
            console.print(f"[green][OK] 训练集加载成功 ({len(train_dataset)} 样本)[/green]")
        except Exception as e:
            console.print(f"[red][FAIL] 训练集加载失败: {str(e)}[/red]")
            raise
        
        # 加载测试集
        console.print("[cyan]正在加载测试集...[/cyan]")
        try:
            test_dataset = dataset_class(
                root=str(self.data_dir),
                train=False,
                download=self.download,
                transform=self._get_transforms(is_training=False)
            )
            console.print(f"[green][OK] 测试集加载成功 ({len(test_dataset)} 样本)[/green]")
        except Exception as e:
            console.print(f"[red][FAIL] 测试集加载失败: {str(e)}[/red]")
            raise
        
        # 创建 DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # 获取数据集信息
        sample_data, sample_label = next(iter(self.train_loader))
        self.input_shape = tuple(sample_data.shape[1:])  # (C, H, W)
        self.num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else len(set(train_dataset.targets))
        
        console.print(f"[green][OK] DataLoader 创建成功[/green]")
        console.print(f"  · 输入形状: {self.input_shape}")
        console.print(f"  · 类别数量: {self.num_classes}")
        console.print(f"  · 批次大小: {self.batch_size}")
        
        return self.train_loader, self.test_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Returns:
            包含数据集信息的字典
        """
        info = {
            "dataset_name": self.dataset_name,
            "data_dir": str(self.data_dir),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
        }
        
        if self.train_loader and self.test_loader:
            info["train_samples"] = len(self.train_loader.dataset)
            info["test_samples"] = len(self.test_loader.dataset)
            info["train_batches"] = len(self.train_loader)
            info["test_batches"] = len(self.test_loader)
        
        return info
    
    def get_sample_batch(self, from_test: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本批次
        
        Args:
            from_test: 是否从测试集获取
            
        Returns:
            (data, labels) 元组
        """
        loader = self.test_loader if from_test else self.train_loader
        
        if loader is None:
            raise RuntimeError("数据集未准备，请先调用 prepare_data()")
        
        return next(iter(loader))
