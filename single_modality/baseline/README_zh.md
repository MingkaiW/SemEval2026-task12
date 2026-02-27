# Baseline 1: 单模态 LLM 基线方法

## 方法概述

Baseline 1 是基于大语言模型 (LLM) 的因果推理基线方法。该方法直接使用预训练的 LLM 进行零样本推理，不进行任何微调。

### 核心思路

1. **输入**: 给定一个事件和相关的文档上下文
2. **任务**: 从候选因果事件列表中选择正确的因果关系
3. **输出**: 预测的因果事件 ID 列表

### 支持的模型类型

| 模型类型 | 说明 | 示例 |
|---------|------|------|
| `openai` | OpenAI API | gpt-4, gpt-3.5-turbo |
| `anthropic` | Anthropic API | claude-3-opus, claude-3-sonnet |
| `huggingface` | HuggingFace 本地模型 | meta-llama/Llama-2-7b |
| `ollama` | Ollama 本地服务 | qwen3:1.7b, llama3:8b |
| `vllm` | vLLM 推理服务 | 各种开源模型 |

## 运行指南

### 环境准备

```bash
# 激活 conda 环境
conda activate py310

# 进入 baseline 目录
cd single_modality/baseline
```

### 使用 Ollama 运行 (推荐本地运行)

#### 1. 启动 Ollama 服务

```bash
# 启动 Ollama 服务 (如果尚未运行)
ollama serve &

# 下载模型 (首次运行需要)
ollama pull qwen3:1.7b
```

#### 2. 运行 dev 数据集

```bash
python run_baseline.py \
    --model-type ollama \
    --model-name qwen3:1.7b \
    --data-split dev \
    --output results_qwen3_dev.json
```

#### 3. 运行 test 数据集

```bash
python run_baseline.py \
    --model-type ollama \
    --model-name qwen3:1.7b \
    --data-split test \
    --output results_qwen3_test.json
```

### 命令行参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--model-type` | 模型类型 (openai/anthropic/huggingface/ollama/vllm) | openai |
| `--model-name` | 模型名称 | gpt-4 |
| `--data-split` | 数据集划分 (sample/train/dev/test) | sample |
| `--output` | 输出文件名 | results.json |
| `--no-context` | 不使用文档上下文 | False |
| `--max-samples` | 最大样本数 (调试用) | None |
| `--submission-file` | 提交文件保存路径 (JSONL格式) | None |

### 提交文件格式

`--submission-file` 选项生成符合比赛要求格式的文件：
```jsonl
{"id": "q-2020", "answer": "A"}
{"id": "q-2021", "answer": "B,D"}
```

示例用法：
```bash
python run_baseline.py \
    --model-type ollama \
    --model-name qwen3:1.7b \
    --data-split test \
    --submission-file submission.jsonl
```

### 数据集规模

| 数据集 | 样本数 |
|-------|-------|
| sample | 200 |
| train | 1,819 |
| dev | 400 |
| test | 612 |

## 结果记录

### 输出文件格式

运行完成后，结果保存为 JSON 文件，包含以下内容：

```json
{
  "config": {
    "model_type": "ollama",
    "model_name": "qwen3:1.7b",
    "data_split": "dev",
    "use_context": true
  },
  "results": {
    "total_samples": 400,
    "exact_matches": 45,
    "partial_matches": 18,
    "wrong": 337,
    "score": 0.135,
    "exact_match_rate": 0.1125,
    "partial_match_rate": 0.045
  },
  "predictions": [...]
}
```

### 评分标准

- **完全匹配 (Exact Match)**: 预测与答案完全一致，得 1.0 分
- **部分匹配 (Partial Match)**: 预测是答案的子集或超集，得 0.5 分
- **错误 (Wrong)**: 其他情况，得 0.0 分

**官方分数计算**: `Score = (完全匹配数 × 1.0 + 部分匹配数 × 0.5) / 总样本数`

## 实验结果记录

### qwen3:1.7b on dev_data

| 指标 | 数值 |
|-----|------|
| 官方分数 (Score) | 0.1350 |
| 完全匹配率 | 11.25% (45/400) |
| 部分匹配率 | 4.50% (18/400) |
| 错误率 | 84.25% (337/400) |

**运行时间**: 约 2 小时 (因 Qwen3 思考模式导致部分样本处理时间较长)

**结果文件**: `results_qwen3_dev.json`

### 结果分析

qwen3:1.7b 作为小参数模型 (1.7B)，在复杂因果推理任务上表现有限。可尝试:

1. 使用更大参数的模型 (如 qwen3:8b, llama3:70b)
2. 使用闭源 API (如 GPT-4, Claude-3)
3. 进行 prompt 优化

## 快速开始示例

```bash
# 1. 激活环境
conda activate py310

# 2. 进入目录
cd single_modality/baseline

# 3. 确保 Ollama 运行
ollama serve &

# 4. 运行小规模测试
python run_baseline.py \
    --model-type ollama \
    --model-name qwen3:1.7b \
    --data-split sample \
    --max-samples 10 \
    --output test_run.json

# 5. 查看结果
cat test_run.json | python -m json.tool
```

## 注意事项

1. **处理时间**: 完整 dev 数据集 (400 样本) 使用 qwen3:1.7b 约需 2 小时
2. **内存需求**: Ollama 运行 qwen3:1.7b 约需 4GB 显存/内存
3. **API 调用**: 使用 OpenAI/Anthropic 需配置相应的 API Key 环境变量
