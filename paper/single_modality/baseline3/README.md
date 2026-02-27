# Baseline 3: 知识图谱增强 LLM QA (KG + LLM)

将知识图谱嵌入与大语言模型结合进行因果推理的baseline实现。

## 目录
- [方法概述](#方法概述)
- [环境配置](#环境配置)
- [完整流程](#完整流程)
- [步骤1: 构建知识图谱](#步骤1-构建知识图谱)
- [步骤2: 训练KG嵌入](#步骤2-训练kg嵌入)
- [步骤3: KG增强QA](#步骤3-kg增强qa)
- [评估与结果](#评估与结果)
- [实验结果汇报](#实验结果汇报)
- [KG Embedding详解](#kg-embedding详解)
- [常见问题](#常见问题)

---

## 方法概述

### 核心思想

将**结构化因果知识**与**LLM的推理能力**结合：

```
┌─────────────────┐     ┌─────────────────┐
│   事件/选项      │────▶│  COMET知识生成   │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  因果知识图谱    │◀────│   三元组抽取     │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  KG Embedding    │────▶│  知识增强Prompt  │
│ (TransE/ComplEx) │     └────────┬────────┘
└─────────────────┘              │
                                 ▼
                        ┌─────────────────┐
                        │    LLM 推理      │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │    因果答案      │
                        └─────────────────┘
```

### 融合方法

| 方法 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| `prompt` | 将KG知识转为自然语言加入prompt | 简单直接 | 上下文长度限制 |
| `retrieval` | 从KG检索相关三元组 | 针对性强 | 需要好的检索策略 |
| `embedding` | 将KG嵌入与文本嵌入融合 | 端到端 | 需要额外训练 |

---

## 环境配置

### 1. 安装依赖

```bash
cd baseline3
pip install -r requirements.txt
```

### 2. 可选依赖

```bash
# COMET知识生成（需要~3GB显存）
pip install transformers sentencepiece

# 更多KG嵌入方法
pip install torchkge  # TorchKGE库
```

### 3. API配置

```bash
# 用于KG-LLM融合QA
export OPENAI_API_KEY="your-key"
# 或
export ANTHROPIC_API_KEY="your-key"
```

---

## 完整流程

### 快速开始（最简版本）

```bash
cd baseline3

# 1. 构建知识图谱（使用简单知识库，无需GPU）
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_output \
    --max-samples 100

# 2. 运行KG增强QA
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-model gpt-4o-mini \
    --max-samples 50 \
    --output ./results/kg_llm_quick.json
```

### 完整流程（推荐）

```bash
# 1. 构建知识图谱 + 训练嵌入
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_output \
    --train-embedding \
    --kg-model TransE \
    --embedding-dim 256 \
    --epochs 100

# 2. 运行评估
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-model gpt-4o-mini \
    --output ./results/kg_llm_dev.json
```

### 使用COMET（需要GPU）

```bash
# 1. 使用COMET生成知识
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_output_comet \
    --use-comet \
    --train-embedding \
    --kg-model RotatE

# 2. 运行QA
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --use-comet \
    --llm-model gpt-4o \
    --output ./results/kg_comet_llm_dev.json
```

---

## 步骤1: 构建知识图谱

### 命令参数

```bash
python run_baseline3.py build-kg [OPTIONS]

必选:
  --data-path PATH       数据目录路径

可选:
  --output-dir DIR       输出目录 (默认: ./kg_output)
  --use-comet            使用COMET生成知识 (需要GPU)
  --train-embedding      训练KG嵌入
  --kg-model MODEL       嵌入模型: TransE, ComplEx, RotatE (默认: TransE)
  --embedding-dim N      嵌入维度 (默认: 256)
  --epochs N             训练轮数 (默认: 100)
  --batch-size N         批次大小 (默认: 256)
  --max-samples N        最大样本数
```

### 输出文件

```
kg_output/
├── knowledge_graph.json   # 知识图谱 (实体、关系、三元组)
├── kg_model.pt            # 训练好的KG嵌入模型
└── embeddings.npz         # 实体嵌入向量
```

### 知识图谱格式

```json
{
  "entities": {
    "Economic recession": 0,
    "Unemployment rises": 1,
    "Interest rate cut": 2,
    ...
  },
  "relations": {
    "causes": 0,
    "is_caused_by": 1,
    "enables": 2,
    ...
  },
  "triples": [
    [0, 0, 1],  // Economic recession causes Unemployment rises
    [2, 0, 3],  // Interest rate cut causes Stock market rises
    ...
  ]
}
```

### Python API

```python
from kg_embedding import CausalKnowledgeGraph, KGEmbeddingTrainer, KGEConfig
from comet_knowledge import build_event_knowledge_graph

# 方法1: 手动构建
kg = CausalKnowledgeGraph()
kg.add_causal_relation("利率下调", "股市上涨")
kg.add_causal_relation("经济衰退", "失业率上升")
kg.save("./kg_output/knowledge_graph.json")

# 方法2: 从数据自动构建
events = ["Event1", "Event2", "Option1", "Option2", ...]
enhanced_events, kg = build_event_knowledge_graph(
    events,
    use_comet=True  # 使用COMET生成因果知识
)
```

---

## 步骤2: 训练KG嵌入

### 支持的模型

| 模型 | 核心思想 | 得分函数 | 适用场景 |
|------|----------|----------|----------|
| **TransE** | h + r ≈ t | \|\|h+r-t\|\| | 快速原型、简单关系 |
| **ComplEx** | 复数空间 | Re(⟨h,r,t̄⟩) | 非对称关系 |
| **RotatE** | 旋转操作 | \|\|h∘r-t\|\| | 复杂关系模式 |

### 训练命令

```bash
# TransE (最快)
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_transe \
    --train-embedding \
    --kg-model TransE \
    --embedding-dim 256 \
    --epochs 100

# RotatE (最强)
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_rotate \
    --train-embedding \
    --kg-model RotatE \
    --embedding-dim 256 \
    --epochs 200
```

### 训练日志示例

```
Building Causal Knowledge Graph
============================================================
Loaded 2000 instances
Collected 5234 unique events

Knowledge Graph Statistics:
  Entities: 5234
  Relations: 8
  Triples: 12456

Training TransE embeddings...
Epoch 10/100, Loss: 0.8234
Epoch 20/100, Loss: 0.5421
Epoch 30/100, Loss: 0.3218
...
Epoch 100/100, Loss: 0.0842

Training complete!
  Final loss: 0.0842
  Model saved: ./kg_output/kg_model.pt
  Embeddings saved: ./kg_output/embeddings.npz
```

### Python API

```python
from kg_embedding import KGEmbeddingTrainer, KGEConfig

# 配置
config = KGEConfig(
    embedding_dim=256,
    num_epochs=100,
    batch_size=256,
    learning_rate=0.001,
    margin=1.0  # TransE
)

# 训练
trainer = KGEmbeddingTrainer(kg, model_type="TransE", config=config)
results = trainer.train()

# 保存
trainer.save_model("./kg_model.pt")
trainer.save_embeddings("./embeddings.npz")

# 获取嵌入
embeddings = trainer.get_entity_embeddings()
print(embeddings["Economic recession"].shape)  # (256,)
```

### 使用嵌入

```python
import numpy as np

# 加载嵌入
embeddings = np.load("./kg_output/embeddings.npz")

# 计算相似度
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

event1_emb = embeddings["Interest rate cut"]
event2_emb = embeddings["Stock market rises"]
similarity = cosine_sim(event1_emb, event2_emb)
print(f"相似度: {similarity:.4f}")
```

---

## 步骤3: KG增强QA

### 命令参数

```bash
python run_baseline3.py qa [OPTIONS]

必选:
  --data-path PATH       数据目录路径

可选:
  --fusion METHOD        融合方法: prompt, retrieval (默认: prompt)
  --llm-type TYPE        LLM类型: openai, anthropic (默认: openai)
  --llm-model MODEL      具体模型名称 (默认: gpt-4o-mini)
  --use-comet            使用COMET生成知识
  --kg-path PATH         KG输出目录 (用于retrieval)
  --output PATH          结果保存路径
  --max-samples N        最大样本数
  --submission-file PATH 提交文件保存路径 (JSONL格式)
```

### 提交文件格式

`--submission-file` 选项生成符合比赛要求格式的文件：
```jsonl
{"id": "q-2020", "answer": "A"}
{"id": "q-2021", "answer": "B,D"}
```

示例用法：
```bash
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/test_data \
    --fusion prompt \
    --llm-model gpt-4o-mini \
    --submission-file submission.jsonl
```

### 融合方法详解

#### 方法1: Prompt增强 (`--fusion prompt`)

将KG知识转换为自然语言，添加到LLM prompt中：

```
Target Event: Stock market crashes

Causal Knowledge:
- Common causes of similar events: Economic recession; Trade war; Interest rate hike
- If Option A happens, likely effects: Unemployment rises; Consumer spending decreases
- If Option B happens, likely effects: Currency weakens; Import costs increase

Options:
A. Economic recession deepens
B. Trade deficit grows
C. New product launches
D. Weather changes

Answer:
```

#### 方法2: 检索增强 (`--fusion retrieval`)

从KG中检索与问题相关的三元组：

```
Target Event: Stock market crashes

Related causal facts:
- Economic recession causes Stock market crash
- Trade war leads_to Economic uncertainty
- Interest rate hike causes Stock market decline

Options:
...
```

### 运行示例

```bash
# Prompt增强 + GPT-4o-mini
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-model gpt-4o-mini \
    --output ./results/prompt_gpt4omini.json

# Prompt增强 + GPT-4o
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-model gpt-4o \
    --output ./results/prompt_gpt4o.json

# 检索增强
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion retrieval \
    --kg-path ./kg_output \
    --llm-model gpt-4o-mini \
    --output ./results/retrieval_gpt4omini.json

# 使用Claude
python run_baseline3.py qa \
    --data-path ../data/semeval2026-task12-dataset/dev_data \
    --fusion prompt \
    --llm-type anthropic \
    --llm-model claude-3-5-sonnet-20241022 \
    --output ./results/prompt_claude.json
```

### Python API

```python
from kg_llm_qa import KGLLMQA, KGLLMConfig

# 配置
config = KGLLMConfig(
    fusion_method="prompt",
    llm_type="openai",
    llm_model="gpt-4o-mini",
    use_comet=False,
    max_kg_context_length=500
)

# 初始化
qa_system = KGLLMQA(config)
qa_system.setup()

# 单条预测
from baseline.data_loader import AERInstance
instance = AERInstance(
    id="test_001",
    topic_id="topic_001",
    target_event="Stock market crashes",
    options={"A": "Economic recession", "B": "Good weather", ...},
    golden_answer=["A"]
)
prediction = qa_system.predict(instance)
print(prediction)  # {"A"}

# 批量评估
results = qa_system.evaluate(instances, output_path="./results.json")
```

---

## 评估与结果

### 评估指标

与Baseline 1/2相同：

| 情况 | 得分 |
|------|------|
| 完全匹配 | 1.0 |
| 部分匹配 | 0.5 |
| 错误 | 0.0 |

### 输出文件格式

```json
{
  "config": {
    "fusion_method": "prompt",
    "llm_model": "gpt-4o-mini",
    "use_comet": false
  },
  "results": {
    "score": 0.6720,
    "exact_match_rate": 0.6040,
    "partial_match_rate": 0.1360,
    "wrong_rate": 0.2600
  },
  "predictions": [
    {
      "id": "dev_001",
      "target_event": "Stock market crashes",
      "prediction": ["A"],
      "golden": ["A"]
    }
  ]
}
```

---

## 实验结果汇报

### 汇报模板 (格式参考)

```markdown
## 实验结果报告

### 实验设置

| 项目 | 值 |
|------|-----|
| 任务 | SemEval 2026 Task 12: AER |
| 数据集 | Dev Set / Test Set |
| KG嵌入模型 | TransE / RotatE / RotH 等 |
| 知识来源 | Simple KB / COMET |
| 硬件 | NVIDIA RTX 3090/4090 |

### 主要结果

| 方法 | LLM | Score | Exact Match |
|------|-----|-------|-------------|
| Baseline (无KG) | [LLM名称] | [分数] | [分数] |
| KG-Prompt | [LLM名称] | [分数] | [分数] |
| KG-Retrieval | [LLM名称] | [分数] | [分数] |

### KG嵌入模型对比

| 模型 | 训练时间 | 最终Loss | 下游Score |
|------|---------|----------|----------|
| TransE | [时间] | [loss] | [分数] |
| RotatE | [时间] | [loss] | [分数] |
| RotH | [时间] | [loss] | [分数] |
```

---

## KG Embedding详解

详细技术文档见 [`KG_EMBEDDING_DOC.md`](./KG_EMBEDDING_DOC.md)，包括：

- TransE/ComplEx/RotatE 数学原理
- 损失函数与训练流程
- 超参数调优指南
- 在AER任务中的应用

### 快速参考

#### TransE
```
h + r ≈ t
Loss = max(0, margin + ||h+r-t|| - ||h'+r-t'||)
```

#### ComplEx
```
score = Re(⟨h, r, t̄⟩)
Loss = BCE(score) + L2_reg
```

#### RotatE
```
t = h ◦ r (复数乘法/旋转)
Loss = -log σ(γ - ||h◦r - t||)
```

---

## 常见问题

### Q: COMET模型下载失败？

```bash
# 手动下载
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    "mismayil/comet-bart-ai2",
    cache_dir="./model_cache"  # 指定缓存目录
)
```

### Q: KG太小怎么办？

1. 使用COMET生成更多知识
2. 增加训练数据
3. 使用外部知识库（如ConceptNet、ATOMIC）

### Q: 如何添加外部知识库？

```python
# 加载ConceptNet
import requests

def query_conceptnet(concept):
    url = f"http://api.conceptnet.io/c/en/{concept}"
    response = requests.get(url).json()
    return response.get("edges", [])

# 添加到KG
for edge in query_conceptnet("economic_recession"):
    if edge["rel"]["label"] == "Causes":
        kg.add_triple(
            edge["start"]["label"],
            "causes",
            edge["end"]["label"]
        )
```

### Q: 嵌入维度如何选择？

| KG规模 | 推荐维度 |
|--------|---------|
| < 1K 实体 | 64-128 |
| 1K-10K | 128-256 |
| 10K-100K | 256-512 |
| > 100K | 512+ |

---

## 文件说明

```
baseline3/
├── kg_embedding.py            # KG嵌入模型 (TransE/ComplEx/RotatE)
├── comet_knowledge.py         # COMET知识生成
├── kg_llm_qa.py               # KG-LLM融合QA
├── run_baseline3.py           # 统一运行脚本
├── requirements.txt           # 依赖
├── KG_EMBEDDING_DOC.md        # KG嵌入技术文档
└── README.md                  # 本文档
```

---

## 参考文献

### KG Embedding
- [TransE (NeurIPS 2013)](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- [ComplEx (ICML 2016)](https://arxiv.org/abs/1606.06357)
- [RotatE (ICLR 2019)](https://arxiv.org/abs/1902.10197)

### 知识增强QA
- [QA-GNN (NAACL 2021)](https://arxiv.org/abs/2104.06378)
- [DRAGON (NeurIPS 2022)](https://arxiv.org/abs/2210.09338)
- [KG-BERT (2019)](https://arxiv.org/abs/1909.03193)

### 常识知识
- [COMET-ATOMIC 2020 (EMNLP 2020)](https://arxiv.org/abs/2010.05953)
- [ATOMIC (AAAI 2019)](https://arxiv.org/abs/1811.00146)

### 代码库
- [OpenKE](https://github.com/thunlp/OpenKE)
- [TorchKGE](https://github.com/torchkge-team/torchkge)
- [COMET-ATOMIC 2020](https://github.com/allenai/comet-atomic-2020)

---

## 实验结果 (Test Set)

| 方法 | Score |
|------|-------|
| DeepSeek LLM (无KG) | 0.58 |
| DeepSeek + COMET | 0.54 |

---

## 改进技术路线

本项目包含两条改进技术路线，可单独或组合使用：

| 路线 | 核心思想 | 依赖 | 适用场景 |
|------|----------|------|----------|
| **Route A** | Lorentz双曲空间KG嵌入 | geoopt | 结构化因果KG |
| **Route B** | 多阶段Qwen3-0.5B训练 | TRL | 自由文本因果推理 |

---

## Route A: Lorentz双曲空间KG嵌入

### 概述

将欧几里得空间的KG嵌入扩展到双曲空间，利用双曲空间的层次结构特性更好地表示因果关系。

**为什么使用Lorentz模型而非Poincaré?**

| 特性 | Poincaré球模型 | Lorentz/Hyperboloid模型 |
|------|---------------|------------------------|
| 数值稳定性 | 差 (边界发散) | **好** (无边界问题) |
| 梯度计算 | 复杂 | **简洁** |
| 可视化 | 好 | 需投影 |
| **实际训练** | 易崩溃 | **稳定** |

### 新增模型

| 模型 | 核心思想 | 参考论文 |
|------|----------|----------|
| **RotH** | 双曲空间Givens旋转 | ACL 2020 |
| **RefH** | 双曲空间反射变换 | ACL 2020 |
| **AttH** | 注意力加权旋转+反射 | ACL 2020 |
| **LorentzKG** | 完整Lorentz变换(boost+rotation) | ACL 2024 |

### 安装依赖

```bash
pip install geoopt  # 黎曼优化库，包含Lorentz流形和RiemannianAdam
```

### 使用方法

```bash
# 训练RotH (Lorentz空间旋转)
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_roth \
    --train-embedding \
    --kg-model RotH \
    --embedding-dim 32 \
    --epochs 100

# 训练LorentzKG (完整Lorentz变换)
python run_baseline3.py build-kg \
    --data-path ../data/semeval2026-task12-dataset/train_data \
    --output-dir ./kg_lorentz \
    --train-embedding \
    --kg-model LorentzKG \
    --embedding-dim 32 \
    --epochs 200
```

### Python API

```python
from kg_embedding import (
    RotH, RefH, AttH, LorentzKG,
    KGEmbeddingTrainer, KGEConfig,
    get_available_models
)

# 查看可用模型
print(get_available_models())
# {'euclidean': ['TransE', 'ComplEx', 'RotatE'],
#  'lorentz': ['RotH', 'RefH', 'AttH', 'LorentzKG']}

# 训练RotH
config = KGEConfig(embedding_dim=32, num_epochs=100)
trainer = KGEmbeddingTrainer(kg, model_type="RotH", config=config)
results = trainer.train()  # 自动使用RiemannianAdam优化器
```

### 数学原理

**Lorentz内积**:
```
<u,v>_L = -u_0*v_0 + Σ(u_i*v_i)
```

**Lorentz距离**:
```
d_L(u,v) = arcosh(-<u,v>_L)
```

**RotH得分函数**:
```
score(h, r, t) = -d_L(Rot_r(h), t)² + b_h + b_t
```

### 参考文献

- **RotH/RefH/AttH**: [Low-Dimensional Hyperbolic Knowledge Graph Embeddings (ACL 2020)](https://aclanthology.org/2020.acl-main.617/)
- **LorentzKG**: [Enhancing Hyperbolic KG Embeddings via Lorentz Transformations (ACL 2024 Findings)](https://arxiv.org/abs/2402.09538)
- **geoopt**: https://github.com/geoopt/geoopt

---

## Route B: 多阶段Qwen3-0.5B训练

### 概述

训练一个专门的因果知识生成器(Qwen3-0.5B)，通过多阶段训练(SFT→GRPO)学习精准的因果推理能力，最终与DeepSeek API结合进行推理。

```
┌─────────────────────────────────────────────────────────────┐
│ 训练流程                                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  训练数据 ──▶ 提取因果三元组 ──▶ SFT数据 ──▶ SFT训练        │
│                      │                           │          │
│                      └───────▶ GRPO数据 ──▶ GRPO训练        │
│                                                  │          │
│                                                  ▼          │
│                                        因果知识生成器        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ 推理流程                                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题 ──▶ Qwen3-0.5B (本地) ──▶ 因果知识 ──▶ DeepSeek API   │
│           生成因果假设              │        最终推理       │
│                                    └──▶ 答案               │
└─────────────────────────────────────────────────────────────┘
```

### 安装依赖

```bash
pip install trl>=0.8.0 peft transformers accelerate bitsandbytes
```

### 训练流程

#### Step 1: 提取因果三元组

```bash
cd baseline3

# 从训练数据提取因果三元组
python data_preparation/extract_causal_triples.py \
    --input ../../train_data/questions.jsonl \
    --output ./data/causal_triples.jsonl \
    --split-output  # 可选: 分别保存正负样本
```

输出统计:
```
Total triples: ~7,500
Positive (correct causes): ~2,000
Negative (incorrect causes): ~5,500
```

#### Step 2: 准备SFT数据

```bash
# 转换为SFT训练格式
python data_preparation/prepare_sft_data.py \
    --input ./data/causal_triples.jsonl \
    --output ./data/causal_sft_data.jsonl \
    --format all  # generation + judgment + mcqa
```

SFT数据格式:
```json
{"messages": [
  {"role": "system", "content": "You are an expert in causal reasoning..."},
  {"role": "user", "content": "Event: Iran launched missile attacks\nWhat are the direct causes?"},
  {"role": "assistant", "content": "The direct cause is: A U.S. drone strike killed Iranian General..."}
]}
```

#### Step 3: SFT训练

```bash
# 单GPU训练
python training/run_sft.py \
    --model Qwen/Qwen3-0.5B-Instruct \
    --data ./data/causal_sft_data.jsonl \
    --output ./output/causal_sft \
    --epochs 3 \
    --batch-size 4 \
    --lora-r 16

# 多GPU训练 (accelerate)
accelerate launch training/run_sft.py \
    --model Qwen/Qwen3-0.5B-Instruct \
    --data ./data/causal_sft_data.jsonl \
    --output ./output/causal_sft
```

#### Step 4: 准备GRPO数据

```bash
python data_preparation/prepare_grpo_data.py \
    --input ./data/causal_triples.jsonl \
    --output ./data/causal_grpo_data.jsonl \
    --format mcqa
```

GRPO数据格式:
```json
{
  "prompt": "Target Event: Iran launched missile attacks...\nOptions:\nA. US conducted airstrikes...\nB. ...",
  "golden_answers": ["D"],
  "golden_causes": ["A U.S. drone strike killed Iranian General..."]
}
```

#### Step 5: GRPO训练

```bash
# GRPO强化学习训练
python training/run_grpo.py \
    --model ./output/causal_sft \
    --data ./data/causal_grpo_data.jsonl \
    --output ./output/causal_grpo \
    --epochs 2 \
    --num-generations 4 \
    --beta 0.1 \
    --reward-type mcqa
```

GRPO配置说明:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-generations` | 4 | 每个prompt生成的回答数 |
| `--beta` | 0.1 | KL惩罚系数 |
| `--loss-type` | grpo | 可选: grpo, dapo, dr_grpo |
| `--scale-rewards` | group | 奖励归一化: group/batch/none |

### 推理使用

```python
from causal_generator import CausalKnowledgeGenerator, TwoStageReasoner

# 初始化因果知识生成器 (本地Qwen3-0.5B)
generator = CausalKnowledgeGenerator(
    model_path="./output/causal_grpo",
    base_model="Qwen/Qwen3-0.5B-Instruct",
    use_lora=True
)

# 初始化两阶段推理器
reasoner = TwoStageReasoner(
    generator=generator,
    deepseek_api_key="your-key"
)

# 推理
result = reasoner.predict(
    target_event="Iran launched ballistic missile attacks...",
    options={
        "A": "US conducted airstrikes...",
        "B": "Military tensions increased...",
        "C": "Diplomatic talks failed...",
        "D": "US drone strike killed Iranian General..."
    }
)

print(result["prediction"])  # ["D"]
print(result["causal_knowledge"])  # 生成的因果分析
```

### 训练资源估算 (RTX 3090/4090)

| Stage | 显存 | 训练时间 |
|-------|------|----------|
| SFT (LoRA) | ~8GB | 1-2小时 |
| GRPO | ~12GB | 2-3小时 |
| **总计** | - | **3-5小时** |

### 奖励函数说明

GRPO使用两种奖励函数:

**1. MCQA奖励 (`--reward-type mcqa`)**
```python
if predicted == golden:
    return 1.0   # 完全匹配
elif predicted.issubset(golden):
    return 0.5   # 部分正确
elif predicted & golden:
    return 0.3   # 有交集
else:
    return -0.5  # 错误
```

**2. 生成奖励 (`--reward-type generation`)**
```python
if cause in completion:
    return 1.0   # 完整匹配
elif cause[:50] in completion:
    return 0.5   # 部分匹配
elif keyword_match >= 3:
    return 0.3   # 关键词匹配
else:
    return 0.0
```

### 文件结构

```
baseline3/
├── data_preparation/
│   ├── extract_causal_triples.py  # 提取因果三元组
│   ├── prepare_sft_data.py        # 准备SFT数据
│   └── prepare_grpo_data.py       # 准备GRPO数据
│
├── training/
│   ├── run_sft.py                 # SFT训练 (TRL SFTTrainer)
│   └── run_grpo.py                # GRPO训练 (TRL GRPOTrainer)
│
├── causal_generator.py            # Qwen3因果知识生成器
│
└── data/
    ├── causal_triples.jsonl       # 因果三元组
    ├── causal_sft_data.jsonl      # SFT训练数据
    └── causal_grpo_data.jsonl     # GRPO训练数据
```

### 参考实现

- **TRL**: https://github.com/huggingface/trl
- **alignment-handbook**: https://github.com/huggingface/alignment-handbook
- **SmolLM**: https://github.com/huggingface/smollm
- **DeepSeek-R1 GRPO**: https://arxiv.org/abs/2401.02954

---

## 两条路线对比

| 特性 | Route A: Lorentz KG | Route B: 多阶段Qwen3 |
|------|---------------------|---------------------|
| **核心思想** | 改进嵌入空间几何 | 训练专用因果生成器 |
| **依赖模型** | DeepSeek (主) | Qwen3-0.5B (辅) + DeepSeek |
| **训练成本** | 低 (仅KG嵌入) | 中 (SFT + GRPO) |
| **推理成本** | 低 | 中 (需本地模型) |
| **可解释性** | 中 (嵌入空间) | 高 (生成显式知识) |
| **适用场景** | 结构化KG | 自由文本因果 |

**建议**:
- 如果有结构化的因果知识图谱 → Route A
- 如果主要依赖自然语言因果描述 → Route B
- 两者可以组合使用，效果可能更好