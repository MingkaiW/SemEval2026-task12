# Baseline 3: KG-Enhanced LLM QA 基线方法

## 方法概述

Baseline 3 将**知识图谱 (KG)** 与**大语言模型 (LLM)** 结合，实现增强的因果推理能力。

| 特性 | 说明 |
|------|------|
| **方法类型** | 知识增强 + 零样本推理 |
| **核心技术** | KG-Augmented Prompt |
| **训练需求** | 无需训练 (QA部分) |
| **LLM支持** | OpenAI API / DeepSeek / Anthropic |

---

## 方法原理

### KG-Augmented Prompt

1. **知识获取**: 从知识库中检索与事件相关的因果知识
2. **Prompt增强**: 将知识嵌入到提示词中
3. **LLM推理**: 使用增强后的提示词进行多选题推理

### 融合方法

| 融合方式 | 说明 | 适用场景 |
|----------|------|----------|
| `prompt` | 将KG知识以自然语言形式添加到prompt | **推荐**，简单高效 |
| `retrieval` | 从预构建的KG中检索相关三元组 | 需要先构建KG |

---

## 运行指南

### 环境准备

```bash
# 激活 conda 环境
conda activate py310

# 进入目录
cd single_modality/baseline3

# 安装依赖 (如需要)
pip install openai tqdm
```

### 使用 DeepSeek API 运行

#### 步骤 1: 运行 dev 评估

```bash
python run_baseline3.py qa \
    --data-path ../../dev_data \
    --fusion prompt \
    --llm-type openai \
    --llm-model deepseek-chat \
    --api-base https://api.deepseek.com \
    --api-key YOUR_DEEPSEEK_API_KEY \
    --output results_deepseek_dev.json
```

#### 步骤 2: 运行 test 评估

```bash
python run_baseline3.py qa \
    --data-path ../../test_data \
    --fusion prompt \
    --llm-type openai \
    --llm-model deepseek-chat \
    --api-base https://api.deepseek.com \
    --api-key YOUR_DEEPSEEK_API_KEY \
    --output results_deepseek_test.json
```

### 使用 OpenAI API 运行

```bash
export OPENAI_API_KEY=your_openai_key

python run_baseline3.py qa \
    --data-path ../../dev_data \
    --fusion prompt \
    --llm-type openai \
    --llm-model gpt-4o-mini \
    --output results_gpt4o_dev.json
```

---

## 命令行参数

### QA 命令参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--data-path` | 数据目录路径 | 必填 |
| `--fusion` | 融合方法 (prompt/retrieval) | prompt |
| `--llm-type` | LLM类型 (openai/anthropic) | openai |
| `--llm-model` | 模型名称 | gpt-4o-mini |
| `--api-base` | 自定义API端点 | None |
| `--api-key` | API密钥 | None (使用环境变量) |
| `--output` | 结果保存路径 | None |
| `--max-samples` | 最大样本数 | None (全部) |
| `--use-comet` | 使用COMET生成知识 | False |
| `--submission-file` | 提交文件保存路径 (JSONL格式) | None |

### 提交文件格式

`--submission-file` 选项生成符合比赛要求格式的文件：
```jsonl
{"id": "q-2020", "answer": "A"}
{"id": "q-2021", "answer": "B,D"}
```

示例用法：
```bash
python run_baseline3.py qa \
    --data-path ../../test_data \
    --fusion prompt \
    --llm-model deepseek-chat \
    --api-base https://api.deepseek.com \
    --api-key YOUR_KEY \
    --submission-file submission.jsonl
```

---

## 实验结果

### DeepSeek (deepseek-chat) 结果

| 数据集 | 样本数 | Score | 完全匹配率 | 部分匹配率 | 运行时间 |
|--------|--------|-------|-----------|-----------|----------|
| dev_data | 400 | **0.5713** | 40.75% | 32.75% | ~8.5分钟 |
| test_data | 612 | - | - | - | ~13分钟 |

**注**: test_data 无标签，需提交到官方评测系统获取分数。

### 对比: 三种 Baseline 方法

| 方法 | 模型 | Score | 说明 |
|------|------|-------|------|
| Baseline 1 | qwen3:1.7b | 0.1350 | 本地LLM直接推理 |
| Baseline 2 | unifiedqa-t5-small | 0.0625 | T5问答模型 |
| **Baseline 3** | **deepseek-chat** | **0.5713** | **KG增强+LLM** |

**分析**: Baseline 3 的分数显著高于 Baseline 1 和 2，证明了知识增强方法的有效性。

---

## 高级用法

### 构建知识图谱 (可选)

如需使用 retrieval 融合方法，需要先构建知识图谱:

```bash
# 构建KG (使用简单知识库)
python run_baseline3.py build-kg \
    --data-path ../../train_data \
    --output-dir ./kg_output

# 构建KG + 训练Embedding
python run_baseline3.py build-kg \
    --data-path ../../train_data \
    --output-dir ./kg_output \
    --train-embedding \
    --kg-model TransE
```

### 使用 COMET 知识生成

COMET 是一个强大的常识知识生成模型:

```bash
python run_baseline3.py qa \
    --data-path ../../dev_data \
    --fusion prompt \
    --use-comet \
    --llm-model deepseek-chat \
    --api-base https://api.deepseek.com \
    --api-key YOUR_KEY
```

**注意**: COMET 需要约 3GB 显存。

---

## 输出文件格式

```json
{
  "config": {
    "llm_model": "deepseek-chat",
    "fusion_method": "prompt"
  },
  "results": {
    "score": 0.5713,
    "exact_match_rate": 0.4075,
    "partial_match_rate": 0.3275
  },
  "predictions": [
    {
      "id": "sample_001",
      "prediction": ["A", "C"],
      "golden": ["A", "C"],
      "match_type": "exact"
    }
  ]
}
```

---

## 评分标准

- **完全匹配 (Exact Match)**: 预测与答案完全一致，得 1.0 分
- **部分匹配 (Partial Match)**: 预测是答案的子集或超集，得 0.5 分
- **错误 (Wrong)**: 其他情况，得 0.0 分

**官方分数计算**:
```
Score = (完全匹配数 × 1.0 + 部分匹配数 × 0.5) / 总样本数
```

---

## 常见问题

### 1. API 调用失败

**错误**: `openai.APIError` 或连接超时

**解决**:
- 检查 API Key 是否正确
- 检查网络连接
- 对于 DeepSeek，确保 `--api-base https://api.deepseek.com`

### 2. 速度慢

**原因**: 每个样本需要单独调用 API

**解决**:
- 使用 `--max-samples 100` 先测试
- 考虑使用本地部署的模型

### 3. 内存不足 (使用COMET时)

**解决**:
- 不使用 `--use-comet`，默认使用简单知识库
- 使用更小的 batch size

---

## 参考资料

- [QA-GNN: Reasoning with Language Models and KGs](https://arxiv.org/abs/2104.06378) (NAACL 2021)
- [KG-BERT: BERT for Knowledge Graph Completion](https://arxiv.org/abs/1909.03193)
- [DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining](https://arxiv.org/abs/2210.09338) (NeurIPS 2022)
- [DeepSeek API Documentation](https://platform.deepseek.com/docs)

---

*最后更新: 2025-01-23*
