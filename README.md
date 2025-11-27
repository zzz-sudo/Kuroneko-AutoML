# Kuroneko-AutoML 完整文档

作者: Kuroneko

---

## 项目简介

Kuroneko-AutoML 是一个模块化且具备闭环反馈机制的 AutoML 框架。系统使用本地加载的大语言模型（DeepSeek-7B）来生成神经架构代码、执行训练，并从结果中学习，通过 LoRA 微调实现持续改进。

基于 NNGPT 架构思想，使用大语言模型自动生成和优化神经网络架构。

---

## 核心特性

- **智能架构生成**: 使用 LLM 根据任务描述自动生成 PyTorch 模型代码
- **闭环反馈机制**: 基于训练结果进行强化学习式的 LoRA 微调
- **严格代码验证**: 使用 AST 和 Pydantic 确保生成代码的安全性和正确性
- **完整日志系统**: 记录所有架构、训练过程和性能指标
- **友好 CLI 界面**: 使用 Rich 库提供专业的命令行交互体验
- **模块化设计**: 高内聚、低耦合，易于扩展和维护

---

## 项目结构

### 目录结构

```
Kuroneko-AutoML/
│
├── models/                          # 本地 LLM 权重存储目录
│   └── deepseek-ai_deepseek-coder-7b-instruct-v1.5/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer.json
│
├── data/                            # 数据集存储目录
│   ├── cifar-10-batches-py/         # CIFAR-10 数据集
│   └── ...
│
├── src/                             # 核心源代码目录
│   ├── __init__.py                  # 包初始化文件
│   ├── model_manager.py             # LLM 模型管理器
│   ├── data_manager.py              # 数据集管理器
│   ├── prompt_engine.py             # 提示词生成引擎
│   ├── validator.py                 # 架构代码校验器
│   ├── executor.py                  # 训练执行器
│   └── feedback_loop.py             # 闭环反馈机制
│
├── logs/                            # 日志和输出目录
│   ├── architecture_history.json    # 架构历史记录
│   ├── feedback_history.json        # 反馈历史记录
│   ├── arch_20241127_094000.py      # 生成的架构代码
│   ├── arch_20241127_094000_execution.log  # 训练日志
│   └── models/                      # 训练好的模型权重
│       └── arch_20241127_094000_best.pth
│
├── configs/                         # 配置文件目录
│   └── config.yaml                  # 主配置文件
│
├── main.py                          # 项目主入口文件
├── requirements.txt                 # Python 依赖列表
└── .gitignore                       # Git 忽略规则
```

---

## 环境配置指南

### 环境要求

- Python 3.9+
- CUDA 11.8+ (推荐，用于 GPU 加速)
- 至少 16GB 内存
- 至少 20GB 磁盘空间（用于模型和数据集）

### 使用 Conda 创建环境（推荐）

#### 1. 创建新的 Conda 环境
```bash
conda create -n kuroneko-automl python=3.9 -y
```

#### 2. 激活环境
```bash
conda activate kuroneko-automl
```

#### 3. 安装 PyTorch（根据您的 CUDA 版本选择）

**CUDA 11.8**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**CUDA 12.1**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**CPU 版本（无 GPU）**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

#### 4. 安装其他依赖
```bash
pip install -r requirements.txt
```

#### 5. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 使用 pip 创建环境（替代方案）

#### 1. 创建虚拟环境
```bash
python -m venv venv
```

#### 2. 激活虚拟环境
- Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- Linux/Mac:
  ```bash
  source venv/bin/activate
  ```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 运行项目
```bash
python main.py
```

---

## 核心模块说明

### 1. ModelManager - 模型管理器

- **作者**: Kuroneko
- **输入**: 模型名称、本地路径、量化配置
- **输出**: 加载的 LLM 模型实例和 Tokenizer
- **功能**: 负责 LLM 权重的下载、加载和管理，支持量化加载

**特性**:
- 自动下载和缓存 LLM 模型
- 支持 4-bit/8-bit 量化加载
- 提供代码生成接口

### 2. DataManager - 数据管理器

- **作者**: Kuroneko
- **输入**: 数据集名称、本地路径、预处理配置
- **输出**: 处理后的 DataLoader 对象
- **功能**: 负责数据集的下载、存储、加载和预处理

**特性**:
- 支持 CIFAR-10/100, MNIST, FashionMNIST
- 自动数据增强和预处理
- 提供 DataLoader 接口

### 3. PromptEngine - 提示词引擎

- **作者**: Kuroneko
- **输入**: 用户任务描述、数据集信息、历史高分架构
- **输出**: 结构化的 Few-shot Prompt
- **功能**: 根据任务需求构建高质量提示词

**特性**:
- Few-shot learning 支持
- 基于历史最佳架构构建提示
- 结构化提示词生成

### 4. Validator - 代码验证器

- **作者**: Kuroneko
- **输入**: LLM 生成的 PyTorch 代码
- **输出**: 验证结果、错误信息、清理后的代码
- **功能**: 使用 AST 和 Pydantic 进行严格的代码验证

**特性**:
- AST 语法检查
- 安全性检查（防止危险代码）
- PyTorch 特定验证

### 5. Executor - 训练执行器

- **作者**: Kuroneko
- **输入**: 验证通过的代码、数据加载器、训练配置
- **输出**: 训练日志、性能指标、模型权重
- **功能**: 在沙盒环境中执行训练并记录结果

**特性**:
- 沙盒环境执行
- 实时训练监控
- 完整日志记录

### 6. FeedbackLoop - 反馈循环

- **作者**: Kuroneko
- **输入**: 训练日志、性能指标、LLM 模型
- **输出**: 高分样本、LoRA 微调触发信号
- **功能**: 实现基于强化学习的反馈循环和 LoRA 微调

**特性**:
- 基于奖励的样本筛选
- LoRA 微调触发
- 持续学习机制

### 7. main.py - 主控制器

- **作者**: Kuroneko
- **输入**: 用户任务描述和配置参数
- **输出**: 训练好的模型、性能报告
- **功能**: 整合所有模块，提供 CLI 交互界面

---

## 使用示例

### 交互式模式

```
>>> 请输入您希望解决的 AI 任务:
使用新的 ResNet 风格模块提升 CIFAR-10 的分类精度

[系统开始工作...]
提示词构建完成
代码生成完成
代码验证通过
训练完成！

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ 指标               ┃ 值                 ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ 验证精度           │ 87.45%             │
│ 训练精度           │ 92.31%             │
│ 训练时长           │ 245.67s            │
└────────────────────┴────────────────────┘
```

### 配置文件示例

编辑 `configs/config.yaml` 来自定义参数：

```yaml
# 大语言模型配置
model:
  name: "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
  use_quantization: true
  quantization_bits: 4

# 数据集配置
data:
  dataset_name: "cifar10"
  batch_size: 128

# 训练配置
training:
  epochs: 10
  learning_rate: 0.001

# 反馈循环配置
feedback:
  reward_threshold: 0.7
  lora_trigger_threshold: 10
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch, torchvision |
| LLM | transformers, accelerate, bitsandbytes |
| 微调 | peft (LoRA) |
| 数据验证 | pydantic |
| CLI | rich, typer |
| 配置 | pyyaml |

---

## 工作流程

```
用户输入任务描述
        ↓
构建 Few-shot Prompt (PromptEngine)
        ↓
LLM 生成架构代码 (ModelManager)
        ↓
代码验证 (Validator)
        ↓
执行训练 (Executor)
        ↓
收集性能指标
        ↓
反馈循环评估 (FeedbackLoop)
        ↓
[达到阈值] LoRA 微调 LLM
        ↓
重复循环，持续改进
```

### 数据流程详解

```
用户输入任务
    ↓
PromptEngine 构建提示词
    ↓
ModelManager 生成架构代码
    ↓
Validator 验证代码
    ↓
Executor 执行训练
    ↓
FeedbackLoop 收集反馈
    ↓
(达到阈值) LoRA 微调 LLM
    ↓
循环往复
```

---

## 日志和输出

系统会在 `logs/` 目录下生成以下文件：

- `architecture_history.json`: 所有架构的历史记录
- `feedback_history.json`: 反馈样本历史
- `arch_*.py`: 生成的模型代码
- `arch_*_execution.log`: 训练日志
- `models/arch_*_best.pth`: 训练好的模型权重

---

## 自定义扩展

### 添加新数据集

在 `src/data_manager.py` 中添加：

```python
SUPPORTED_DATASETS = {
    "your_dataset": YourDatasetClass,
    # ...
}
```

### 自定义验证规则

在 `src/validator.py` 中扩展 `Validator` 类的检查方法。

### 调整 LoRA 参数

在 `configs/config.yaml` 中修改 `lora` 配置。

---

## 故障排除

### 问题1：内存不足

**解决方案**：
1. 启用模型量化：`use_quantization: true`
2. 减小批次大小：`batch_size: 64`
3. 使用更小的模型

### 问题2：下载速度慢

**解决方案**：
1. 配置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题3：CUDA Out of Memory

**解决方案**：
1. 使用 4-bit 量化
2. 减少 `num_workers`
3. 降低 `max_new_tokens`

---

## 设计原则

1. **模块化**: 高内聚、低耦合
2. **可扩展**: 工厂模式、适配器模式
3. **安全性**: 代码沙盒、严格验证
4. **可追溯**: 完整的日志记录
5. **持续学习**: LoRA 微调闭环

---

## 注意事项

1. 确保您的系统有足够的存储空间（LLM 模型约 15GB）
2. 推荐使用 GPU 以加速模型推理和训练
3. 首次运行会自动下载 DeepSeek-7B 模型和 CIFAR-10 数据集
4. 训练过程中会生成大量日志文件，请定期清理

---

## 许可证

本项目仅供学习和研究使用。

---

## 致谢

- 灵感来源于 NNGPT 论文
- 基于 DeepSeek-Coder 模型
- 使用 HuggingFace Transformers 库

---

## 联系方式

**作者**: Kuroneko

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**如果这个项目对你有帮助，请给个 Star！**


