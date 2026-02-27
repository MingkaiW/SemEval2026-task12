# Baseline 2: UnifiedQA / RoBERTa 基线方法

## 方法概述

Baseline 2 提供两种基线方法:

| 方法 | 类型 | 训练需求 | 适用场景 |
|------|------|---------|---------|
| **UnifiedQA** | 生成式 (T5) | 零样本 | 快速验证、无标注数据 |
| **RoBERTa** | 判别式 | 需要微调 | 有标注数据、追求更高精度 |

---

## 方法一: UnifiedQA (零样本)

### 原理

UnifiedQA 是 AllenAI 开发的统一问答模型，基于 T5 架构:
- 将多选题转换为 "问题 + 选项" 的文本格式
- 直接生成答案字母 (A/B/C/D)
- 无需训练，开箱即用

### 支持的模型

| 模型 | 大小 | 参数量 | 推荐场景 |
|------|------|--------|---------|
| allenai/unifiedqa-t5-small | ~250MB | 60M | 快速测试 |
| allenai/unifiedqa-t5-base | ~900MB | 220M | 平衡选择 |
| allenai/unifiedqa-t5-large | ~3GB | 770M | 更高精度 |
| allenai/unifiedqa-v2-t5-base-1363200 | - | - | 增强版 |

---

## 运行指南

### 环境准备

```bash
# 激活 conda 环境
conda activate py310

# 进入目录
cd single_modality/baseline2
```

### 步骤 1: 数据预处理

```bash
python run_baseline2.py preprocess --dataset-dir ../../
```

输出文件:
- `processed_data/dev/unifiedqa.jsonl`
- `processed_data/test/unifiedqa.jsonl`

### 步骤 2: 运行 UnifiedQA (dev)

```bash
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/dev/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-t5-small \
    --output results_unifiedqa_dev.json
```

### 步骤 3: 运行 UnifiedQA (test)

```bash
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/test/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-t5-small \
    --output results_unifiedqa_test.json
```

---

## 命令行参数

### 预处理参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--dataset-dir` | 原始数据集目录 | 必填 |
| `--output-dir` | 输出目录 | ./processed_data |
| `--no-context` | 不包含上下文文档 | False |

### UnifiedQA 参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--data-path` | 预处理后的数据路径 | 必填 |
| `--model-name` | 模型名称 | allenai/unifiedqa-t5-base |
| `--output` | 结果保存路径 | None |
| `--batch-size` | 批次大小 | 8 |
| `--submission-file` | 提交文件保存路径 (JSONL格式) | None |

### 提交文件格式

`--submission-file` 选项生成符合比赛要求格式的文件：
```jsonl
{"id": "q-2020", "answer": "A"}
{"id": "q-2021", "answer": "B,D"}
```

示例用法：
```bash
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/test/unifiedqa.jsonl \
    --submission-file submission.jsonl
```

---

## 结果记录

### 输出文件格式

```json
{
  "config": {
    "model_name": "allenai/unifiedqa-t5-small",
    "data_path": "./processed_data/dev/unifiedqa.jsonl"
  },
  "results": {
    "score": 0.0625,
    "exact_match_rate": 0.03,
    "partial_match_rate": 0.065
  },
  "predictions": [...]
}
```

### 评分标准

- **完全匹配**: 预测与答案完全一致，得 1.0 分
- **部分匹配**: 预测是答案的子集或超集，得 0.5 分
- **错误**: 其他情况，得 0.0 分

**官方分数**: `Score = (完全匹配数 × 1.0 + 部分匹配数 × 0.5) / 总样本数`

---

## 实验结果

### unifiedqa-t5-small on dev_data

| 指标 | 数值 |
|-----|------|
| 官方分数 (Score) | 0.0625 |
| 完全匹配率 | 3.00% (12/400) |
| 部分匹配率 | 6.50% (26/400) |

**运行时间**: ~30 秒 (400 样本, GPU)

### 对比: Baseline 1 vs Baseline 2

| 方法 | 模型 | Score | 时间 |
|------|------|-------|------|
| Baseline 1 | qwen3:1.7b | 0.1350 | ~2小时 |
| Baseline 2 | unifiedqa-t5-small | 0.0625 | ~30秒 |

**分析**: UnifiedQA-t5-small 参数量较小 (60M)，在因果推理任务上效果有限。建议:
1. 使用更大模型 (unifiedqa-t5-base/large)
2. 使用 UnifiedQA-v2 增强版
3. 使用 RoBERTa 微调方法

---

## 常见问题

### 1. torch 版本不兼容

**错误**: `ValueError: requires torch >= 2.6`

**解决**: 代码已包含 `patch_torch.py` 自动绕过版本检查

### 2. CUDA 内存不足

**解决**: 减小 batch-size
```bash
python run_baseline2.py unifiedqa --batch-size 4 ...
```

### 3. 模型下载慢

**解决**: 设置 HuggingFace 镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 方法二: RoBERTa 微调 (可选)

如需更高精度，可使用 RoBERTa 微调:

```bash
# 训练
python run_baseline2.py roberta --mode train \
    --train-data ./processed_data/train/roberta_mcqa.jsonl \
    --dev-data ./processed_data/dev/roberta_mcqa.jsonl \
    --model-name microsoft/deberta-v3-base

# 预测
python run_baseline2.py roberta --mode predict \
    --data-path ./processed_data/test/roberta_mcqa.jsonl \
    --model-name ./roberta_output \
    --output results_roberta_test.json
```

---

## 参考资料

- [UnifiedQA 论文](https://arxiv.org/abs/2005.00700)
- [UnifiedQA GitHub](https://github.com/allenai/unifiedqa)
- [HuggingFace 模型](https://huggingface.co/allenai/unifiedqa-t5-base)

---

*最后更新: 2025-01-23*
