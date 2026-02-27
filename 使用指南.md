# SemEval 2026 Task 12 – User Guide

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Baseline 1: LLM API Calling](#3-baseline-1-llm-api-calling)
4. [Baseline 2: UnifiedQA / RoBERTa](#4-baseline-2-unifiedqa--roberta)
5. [Baseline 3: Knowledge-Graph–Enhanced QA](#5-baseline-3-knowledge-graph–enhanced-qa)
6. [Multimodal Data Processing](#6-multimodal-data-processing)
7. [FAQ](#7-faq)

---

## 1. Environment Setup

### 1.1 Basic Environment

```bash
# Activate an existing environment
conda activate py310

# Or create a new one
# conda create -n py310 python=3.10
# conda activate py310

# Clone the official dataset repo
git clone https://github.com/sooo66/semeval2026-task12-dataset.git
cd semeval2026-task12-dataset
```

### 1.2 Install Dependencies

```bash
# Dependencies for Baseline 1 & 2
pip install openai anthropic transformers torch tqdm

# Extra dependencies for Baseline 3 (knowledge graph)
pip install numpy scipy

# Multimodal processing
pip install Pillow requests
```

### 1.3 API Key Configuration

```bash
# OpenAI
export OPENAI_API_KEY="sk-xxx"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-xxx"
```

On Windows PowerShell you can use, for example:

```powershell
$env:OPENAI_API_KEY = "sk-xxx"
$env:ANTHROPIC_API_KEY = "sk-ant-xxx"
```

---

## 2. Dataset Preparation

### 2.1 Directory Structure

```text
semeval2026-task12-dataset/
├── sample_data/          # Toy split (200 samples, 10 topics)
│   ├── questions.jsonl
│   ├── docs.json
│   └── docs_updated.json
├── train_data/           # Training split (1,819 samples, 36 topics)
│   ├── questions.jsonl   # Question file
│   ├── docs.json         # Document file
│   └── docs_updated.json # With local image paths
├── dev_data/             # Dev split (400 samples, 36 topics)
│   ├── questions.jsonl
│   ├── docs.json
│   └── docs_updated.json
├── test_data/            # Test split (612 samples) [NEW]
│   ├── questions.jsonl
│   └── docs.json
└── downloaded_images/    # Downloaded images
    ├── sample_data/topic_[1-10]/
    ├── train_data/topic_[1-36]/
    ├── dev_data/topic_[1-36]/
    └── test_data/topic_[1-20]/
```

### 2.2 Data Formats

**questions.jsonl** (one JSON object per line):

```json
{
  "topic_id": 22,
  "id": "uuid-xxx",
  "target_event": "Iran launched missile attacks on U.S. bases ...",
  "option_A": "On Dec 29, U.S. forces carried out airstrikes in Iraq ...",
  "option_B": "After 2006, Muhandis founded ...",
  "option_C": "On Dec 27, Kata'ib Hezbollah attacked ...",
  "option_D": "A U.S. drone strike killed an Iranian general ...",
  "golden_answer": "D"
}
```

**docs.json**:

```json
[
  {
    "topic_id": 22,
    "topic": "Iran missile attack incident",
    "docs": [
      {
        "title": "News headline",
        "content": "Article body ...",
        "source": "News source",
        "imageUrl": "base64 or URL",
        "id": "d-1"
      }
    ]
  }
]
```

**Note:** the newer dataset versions use the key `id` instead of the older `uuid`.

### 2.3 Image Extraction (Optional)

If you want to use the multimodal setting (text + images):

```bash
# Extract images from docs.json and save locally
python process_images.py
```

This will create the `downloaded_images/` directory and `docs_updated.json` files.

---

## 3. Baseline 1: LLM API Calling

### 3.1 Overview

This baseline directly calls large language model (LLM) APIs for abductive causal reasoning, without any task-specific fine-tuning.

**Supported model backends:**

| Type       | Example models                              |
|------------|---------------------------------------------|
| OpenAI     | gpt-4o, gpt-4o-mini, gpt-4-turbo            |
| Anthropic  | claude-3-5-sonnet-20241022                  |
| HuggingFace| meta-llama/Llama-3.1-8B-Instruct            |
| Ollama     | llama3.1:8b, qwen2:7b (local)               |
| vLLM       | High-throughput serving for HF models       |

### 3.2 How to Run

```bash
cd single_modality/baseline

# === OpenAI models ===
# Use GPT-4o-mini (recommended for cost–performance)
python run_baseline.py \
    --model-type openai \
    --model-name gpt-4o-mini \
    --data-split dev

# Use GPT-4o (stronger, more expensive)
python run_baseline.py \
    --model-type openai \
    --model-name gpt-4o \
    --data-split dev \
    --output results_gpt4o.json

# === Anthropic models ===
python run_baseline.py \
    --model-type anthropic \
    --model-name claude-3-5-sonnet-20241022 \
    --data-split dev

# === Local Ollama models ===
# Start Ollama backend first
ollama serve &
ollama pull llama3.1:8b

python run_baseline.py \
    --model-type ollama \
    --model-name llama3.1:8b \
    --data-split dev

# === Run without context documents ===
python run_baseline.py \
    --model-type openai \
    --model-name gpt-4o-mini \
    --no-context

# === Quick sanity check (limit samples) ===
python run_baseline.py \
    --model-type openai \
    --model-name gpt-4o-mini \
    --max-samples 20 \
    --output test_results.json
```

### 3.3 Key Arguments

| Argument        | Description                          | Default        |
|-----------------|--------------------------------------|----------------|
| `--model-type`  | Backend type                         | openai         |
| `--model-name`  | Model identifier                     | gpt-4o-mini    |
| `--data-split`  | Dataset split (train/dev/test)       | dev            |
| `--no-context`  | Disable document context             | False          |
| `--max-samples` | Max number of samples (debug)        | None           |
| `--output`      | Where to save predictions (JSON)     | None           |

### 3.4 Example Evaluation Output

```text
============================================================
Evaluation Results
============================================================
Official score: 0.6250
Exact match rate:   0.5500 (220/400)
Partial match rate: 0.1500 (60/400)
Error rate:         0.3000 (120/400)
```

---

## 4. Baseline 2: UnifiedQA / RoBERTa

### 4.1 Overview

This baseline follows more classical NLP setups:

- **UnifiedQA**: T5-style generative QA used in zero-shot mode.
- **RoBERTa / DeBERTa**: Discriminative multiple-choice QA models that require supervised fine-tuning.

### 4.2 Preprocessing

```bash
cd single_modality/baseline2

# Prepare data (generate UnifiedQA and RoBERTa formats)
python run_baseline2.py preprocess \
    --dataset-dir ../../ \
    --output-dir ./processed_data

# Preprocess without context documents
python run_baseline2.py preprocess \
    --dataset-dir ../../ \
    --output-dir ./processed_data_no_context \
    --no-context
```

Output directory structure:

```text
processed_data/
├── train/
│   ├── unifiedqa.jsonl
│   └── roberta_mcqa.jsonl
└── dev/
    ├── unifiedqa.jsonl
    └── roberta_mcqa.jsonl
```

### 4.3 Running UnifiedQA (Zero-Shot)

```bash
# === Base model ===
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/dev/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-t5-base

# === Stronger UnifiedQA-v2 ===
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/dev/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-v2-t5-large-1363200

# === Save predictions ===
python run_baseline2.py unifiedqa \
    --data-path ./processed_data/dev/unifiedqa.jsonl \
    --model-name allenai/unifiedqa-v2-t5-base-1363200 \
    --output unifiedqa_results.json \
    --batch-size 8
```

**Available UnifiedQA models:**

| Model                                      | Params | Notes                  |
|--------------------------------------------|--------|------------------------|
| `allenai/unifiedqa-t5-small`               | 60M    | Fastest                |
| `allenai/unifiedqa-t5-base`                | 220M   | Good balance           |
| `allenai/unifiedqa-t5-large`               | 770M   | More accurate          |
| `allenai/unifiedqa-v2-t5-base-1363200`     | 220M   | **Recommended**        |
| `allenai/unifiedqa-v2-t5-large-1363200`    | 770M   | Strongest zero-shot    |

### 4.4 Running RoBERTa / DeBERTa (Fine-Tuning)

```bash
# === Train RoBERTa ===
python run_baseline2.py roberta \
    --mode train \
    --train-data ./processed_data/train/roberta_mcqa.jsonl \
    --dev-data ./processed_data/dev/roberta_mcqa.jsonl \
    --model-name roberta-base \
    --output-dir ./roberta_output \
    --epochs 3 \
    --batch-size 4

# === Train DeBERTa (recommended) ===
python run_baseline2.py roberta \
    --mode train \
    --train-data ./processed_data/train/roberta_mcqa.jsonl \
    --dev-data ./processed_data/dev/roberta_mcqa.jsonl \
    --model-name microsoft/deberta-v3-base \
    --output-dir ./deberta_output \
    --epochs 3 \
    --batch-size 4

# === Predict ===
python run_baseline2.py roberta \
    --mode predict \
    --data-path ./processed_data/dev/roberta_mcqa.jsonl \
    --model-name ./roberta_output \
    --output roberta_predictions.json
```

**Available discriminative models:**

| Model                         | Params | Notes                 |
|-------------------------------|--------|-----------------------|
| `roberta-base`                | 125M   | Standard baseline     |
| `roberta-large`               | 355M   | Larger, stronger      |
| `microsoft/deberta-v3-base`   | 184M   | **Recommended**       |
| `microsoft/deberta-v3-large`  | 434M   | Strongest (expensive) |

---

## 5. Baseline 3: Knowledge-Graph–Enhanced QA

### 5.1 Overview

This baseline combines knowledge-graph (KG) embeddings with LLMs to enhance abductive causal reasoning.

**Core components:**

- **KG embedding models**: TransE, ComplEx, RotatE
- **Knowledge generation**: COMET-ATOMIC 2020
- **Fusion strategies**: prompt augmentation / retrieval augmentation

### 5.2 Building the Knowledge Graph

```bash
cd single_modality/baseline3

# === Simple KG (fast) ===
python run_baseline3.py build-kg \
    --data-path ../../train_data \
    --output-dir ./kg_output

# === COMET-based commonsense KG (requires ~3GB GPU) ===
python run_baseline3.py build-kg \
    --data-path ../../train_data \
    --output-dir ./kg_output_comet \
    --use-comet

# === Build + train KG embeddings ===
python run_baseline3.py build-kg \
    --data-path ../../train_data \
    --output-dir ./kg_output_full \
    --use-comet \
    --train-embedding \
    --kg-model TransE \
    --embedding-dim 256 \
    --epochs 100 \
    --batch-size 256
```

Output files:

```text
kg_output/
├── knowledge_graph.json  # KG triples
├── kg_model.pt           # Trained embedding model (optional)
└── embeddings.npz        # Entity / relation embeddings
```

### 5.3 Choosing a KG Embedding Model

| Model     | Intuition / Formula          | When to use                      |
|-----------|-----------------------------|----------------------------------|
| TransE    | `h + r ≈ t`                 | Fast prototyping, simple graphs  |
| ComplEx   | Complex Hermitian product   | Asymmetric / diverse relations   |
| RotatE    | `t = h ◦ r` (rotation)      | Rich relational patterns, strong |

### 5.4 Running KG-Enhanced QA

```bash
# === Prompt augmentation (recommended) ===
# Inject KG triples into the LLM prompt
python run_baseline3.py qa \
    --data-path ../../dev_data \
    --fusion prompt \
    --llm-type openai \
    --llm-model gpt-4o-mini \
    --use-comet

# === Retrieval augmentation ===
# Retrieve relevant triples from the KG
python run_baseline3.py qa \
    --data-path ../../dev_data \
    --fusion retrieval \
    --kg-path ./kg_output_full \
    --llm-model gpt-4o-mini

# === Using Anthropic models ===
python run_baseline3.py qa \
    --data-path ../../dev_data \
    --fusion prompt \
    --llm-type anthropic \
    --llm-model claude-3-5-sonnet-20241022

# === Save results ===
python run_baseline3.py qa \
    --data-path ../../dev_data \
    --fusion prompt \
    --llm-model gpt-4o-mini \
    --output kg_qa_results.json \
    --max-samples 100
```

### 5.5 Suggested Hyperparameters

| Parameter          | Recommended | Description                |
|--------------------|------------|----------------------------|
| `embedding_dim`    | 256        | Embedding dimension        |
| `epochs`           | 100        | Training epochs            |
| `batch_size`       | 256        | Batch size                 |
| `margin`           | 1.0 / 9.0  | Loss margin (TransE/RotatE)|
| `learning_rate`    | 0.001      | Learning rate              |

---

## 6. Multimodal Data Processing

### 6.1 Image Extraction Pipeline

```bash
# Extract images from docs.json
python process_images.py
```

This will:

1. Read base64/URL images from `docs.json`.
2. Save them under `downloaded_images/{split}/topic_{id}/`.
3. Create `docs_updated.json` files that include `local_image_path` fields.

### 6.2 Preparing Data for VaLiK

```bash
# 1. Convert SemEval AER data into VaLiK format
python preparae_dataset/prepare_semeval_for_valik.py

# 2. Generate image captions with Ollama
ollama pull qwen3-vl:8b
ollama serve &

cd VaLiK/src
python Image_to_Text.py \
    --input ../../valik_prepared/train_data/images \
    qwen3 --qwen3_version 8b

# 3. Merge original text and image captions
cd ../..
python merge_texts.py

# 4. Build a multimodal knowledge graph with LightRAG
cd VaLiK/src/LightRAG
python lightrag_ollama_demo_semeval.py
```

### 6.3 Multimodal ML Baseline

```bash
cd baselines

# Quick test (Logistic Regression only)
./run_quick_test.sh

# Full evaluation (all classifiers)
./run_all_splits.sh

# Use image descriptions in the MLP baseline
python ml_mlp_baseline.py \
    --train_questions ../train_data/questions.jsonl \
    --train_docs ../train_data/docs_updated.json \
    --dev_questions ../dev_data/questions.jsonl \
    --dev_docs ../dev_data/docs_updated.json \
    --use_image_descriptions \
    --img2txt_model qwen3 \
    --save_predictions
```

---

## 7. FAQ

### Q1: "API key not found" or similar errors

```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Or set them directly
export OPENAI_API_KEY="sk-xxx"
```

On Windows PowerShell:

```powershell
echo $env:OPENAI_API_KEY
echo $env:ANTHROPIC_API_KEY
```

### Q2: CUDA out of memory

```bash
# Reduce batch size
--batch-size 2

# Use a smaller model
--model-name allenai/unifiedqa-t5-small
```

### Q3: Ollama connection failures

```bash
# Make sure Ollama is running
ollama serve &

# Check that the model is installed
ollama list
ollama pull llama3.1:8b
```

### Q4: Preprocessed files not found

```bash
# Inspect the output directory
ls -la single_modality/baseline2/processed_data/

# Re-run preprocessing
python run_baseline2.py preprocess --dataset-dir ../../
```

### Q5: Knowledge-graph construction fails

```bash
# Check data paths
ls ../../train_data/questions.jsonl

# Disable COMET (avoid GPU issues)
python run_baseline3.py build-kg \
    --data-path ../../train_data \
    --output-dir ./kg_output
```

---

## Evaluation Metric

The official scoring rule is:

- **1.0**: Exact match (prediction set = gold set).
- **0.5**: Partial match (prediction is a non-empty proper subset of the gold set).
- **0.0**: Otherwise (empty prediction, superset, or wrong options).

**Examples:**

| Prediction | Gold  | Score |
|-----------:|:------|:------|
| {A}        | {A}   | 1.0   |
| {A,B}      | {A,B} | 1.0   |
| {A}        | {A,B} | 0.5   |
| {A,B}      | {A}   | 0.0   |
| {C}        | {A}   | 0.0   |

---

## Recommended Workflows

### Quick sanity check

```bash
# 1. LLM API baseline (fastest)
cd single_modality/baseline
python run_baseline.py --model-type openai --model-name gpt-4o-mini --max-samples 50

# 2. UnifiedQA zero-shot
cd ../baseline2
python run_baseline2.py preprocess --dataset-dir ../../
python run_baseline2.py unifiedqa --data-path ./processed_data/dev/unifiedqa.jsonl
```

### Full evaluation

```bash
# 1. Try multiple LLMs
for model in gpt-4o-mini gpt-4o; do
    python run_baseline.py --model-type openai --model-name $model --output results_$model.json
done

# 2. Fine-tune DeBERTa
python run_baseline2.py roberta --mode train \
    --train-data ./processed_data/train/roberta_mcqa.jsonl \
    --dev-data ./processed_data/dev/roberta_mcqa.jsonl \
    --model-name microsoft/deberta-v3-base

# 3. Knowledge-graph–enhanced QA
cd ../baseline3
python run_baseline3.py build-kg --data-path ../../train_data --use-comet --train-embedding
python run_baseline3.py qa --data-path ../../dev_data --fusion prompt
```

---

## Useful References

- [Official dataset](https://github.com/sooo66/semeval2026-task12-dataset)
- [Competition page (Codabench)](https://www.codabench.org/competitions/12440/)
- [SemEval 2026](https://semeval.github.io/SemEval2026/)
- [UnifiedQA paper](https://arxiv.org/abs/2005.00700)
- [COMET-ATOMIC 2020](https://github.com/allenai/comet-atomic-2020)

---

*Last updated: 2025-01-22 (added test_data, id field replaces uuid)*
