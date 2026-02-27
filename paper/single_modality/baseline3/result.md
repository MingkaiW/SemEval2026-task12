# Baseline 3: KG-Enhanced LLM Results

## Experiment Overview

- **Task**: SemEval 2026 Task 12 - Abductive Event Reasoning
- **Dataset**: dev_data (400 samples), test_data (612 samples)
- **Date**: 2026-01-23

---

## Results Summary

| Method | Model | Dev Score | Exact Match | Partial Match | Wrong Rate |
|--------|-------|-----------|-------------|---------------|------------|
| Baseline 1 (LLM) | Qwen3:1.7b | 0.1350 | 11.25% | 4.50% | - |
| Baseline 2 (UnifiedQA) | UnifiedQA-t5-small | 0.0625 | 3.00% | 6.50% | - |
| Baseline 3 (Simple KG) | DeepSeek | 0.5713 | 40.75% | 32.75% | 26.50% |
| Baseline 3 (RotatE) | DeepSeek | 0.5787 | 42.00% | 31.75% | 26.25% |
| **Baseline 3 (COMET)** | **DeepSeek** | **0.6088** | **46.25%** | **29.25%** | **24.50%** |

---

## Best Result: KG + COMET Knowledge Generation

### Configuration

| Parameter | Value |
|-----------|-------|
| KG Model | RotatE |
| Embedding Dimension | 256 |
| Training Epochs | 100 |
| Knowledge Source | **COMET (comet-atomic_2020_BART)** |
| LLM | DeepSeek-chat |
| Fusion Method | prompt |
| API | https://api.deepseek.com |

### Knowledge Graph Statistics (COMET-Enhanced)

| Metric | Simple KG | COMET KG | Improvement |
|--------|-----------|----------|-------------|
| Entities | 588 | 3,988 | +6.8x |
| Relations | 8 | 8 | - |
| Triples | 218 | 19,396 | +89x |

### Dev Set Results (COMET)

| Metric | Value |
|--------|-------|
| **Score** | **0.6088** |
| Exact Match | 46.25% (185/400) |
| Partial Match | 29.25% (117/400) |
| Wrong Rate | 24.50% (98/400) |

### Test Set Results (COMET)

- **Samples**: 612
- **Submission File**: `submission_kg_comet_full_test.jsonl`
- **Note**: No labels available (submit to official evaluation)

---

## Experiment History

### Experiment 1: Simple KG (Baseline)

| Parameter | Value |
|-----------|-------|
| Knowledge Source | SimpleCausalKnowledgeBase |
| KG Stats | 588 entities, 218 triples |
| Dev Score | 0.5713 |
| Exact Match | 40.75% |

### Experiment 2: RotatE Embedding (No COMET)

| Parameter | Value |
|-----------|-------|
| KG Model | RotatE |
| Embedding Dimension | 256 |
| Training Epochs | 100 |
| Final Loss | 0.0767 |
| Knowledge Source | SimpleCausalKnowledgeBase (COMET fallback) |
| Dev Score | 0.5787 |
| Exact Match | 42.00% |

**Note**: COMET was unavailable due to PyTorch 2.5.1 security restrictions (CVE-2025-32434).

### Experiment 3: Full COMET Knowledge Generation

| Parameter | Value |
|-----------|-------|
| KG Model | RotatE |
| Embedding Dimension | 256 |
| Training Epochs | 100 |
| Knowledge Source | **COMET (comet-atomic_2020_BART)** |
| Dev Score | **0.6088** |
| Exact Match | **46.25%** |

**Fix Applied**: Used `use_safetensors=True` in model loading to bypass PyTorch security check.

---

## Comparison: Simple KG vs COMET KG

| Metric | Simple KG | RotatE (Simple) | COMET KG | Change (vs Simple) |
|--------|-----------|-----------------|----------|-------------------|
| Score | 0.5713 | 0.5787 | **0.6088** | **+6.6%** |
| Exact Match | 40.75% | 42.00% | **46.25%** | **+5.50%** |
| Partial Match | 32.75% | 31.75% | 29.25% | -3.50% |
| Wrong Rate | 26.50% | 26.25% | **24.50%** | **-2.00%** |

### Key Findings

1. **COMET Knowledge Dramatically Improves Results**:
   - Score increased from 0.5787 to 0.6088 (+5.2% relative improvement)
   - Exact match improved from 42.00% to 46.25% (+4.25% absolute)
   - Wrong rate decreased from 26.25% to 24.50%

2. **Knowledge Graph Density Matters**:
   - COMET KG has 89x more triples (19,396 vs 218)
   - COMET KG has 6.8x more entities (3,988 vs 588)
   - Richer commonsense knowledge leads to better reasoning

3. **Overall Performance**:
   - KG+COMET achieves ~4.5x better score than Baseline 1 (Qwen3)
   - KG+COMET achieves ~9.7x better score than Baseline 2 (UnifiedQA)

### Technical Fix: COMET Model Loading

The COMET model loading issue (PyTorch 2.5.1 security restrictions) was resolved by using safetensors format:

```python
# In comet_knowledge.py
self.model = AutoModelForSeq2SeqLM.from_pretrained(
    self.config.model_name,
    use_safetensors=True  # Bypasses torch.load security check
).to(self.config.device)
```

### Future Improvements

1. Try larger LLM models (e.g., DeepSeek-67B, GPT-4)
2. Experiment with retrieval fusion method instead of prompt fusion
3. Fine-tune COMET on domain-specific causal knowledge
4. Combine multiple KG embedding models (ensemble)

---

## Generated Files

### COMET-Enhanced KG (Best Results)

| File | Description |
|------|-------------|
| `kg_output_comet_full/knowledge_graph.json` | COMET knowledge graph (3,988 entities, 19,396 triples) |
| `kg_output_comet_full/kg_model.pt` | Trained RotatE model on COMET KG |
| `kg_output_comet_full/embeddings.npz` | Entity embeddings from COMET KG |
| `results_kg_comet_full_dev.json` | Dev set evaluation results (COMET) |
| `results_kg_comet_full_test.json` | Test set predictions (COMET) |
| `submission_kg_comet_full_dev.jsonl` | Dev set submission file (COMET) |
| `submission_kg_comet_full_test.jsonl` | Test set submission file (COMET) |

### Simple KG (Previous Experiments)

| File | Description |
|------|-------------|
| `kg_output_comet/knowledge_graph.json` | Simple knowledge graph (588 entities, 218 triples) |
| `kg_output_comet/kg_model.pt` | Trained RotatE model |
| `kg_output_comet/embeddings.npz` | Entity embeddings |
| `results_kg_rotateE_dev.json` | Dev set evaluation results |
| `results_kg_rotateE_test.json` | Test set predictions |
| `submission_kg_rotateE_dev.jsonl` | Dev set submission file |
| `submission_kg_rotateE_test.jsonl` | Test set submission file |

---

## Run Commands

### Build KG with COMET + Train Embedding (Recommended)

```bash
python run_baseline3.py build-kg \
    --data-path ../../train_data \
    --output-dir ./kg_output_comet_full \
    --use-comet \
    --train-embedding \
    --kg-model RotatE \
    --embedding-dim 256 \
    --epochs 100
```

### Run QA on Dev Set (with COMET KG)

```bash
python run_baseline3.py qa \
    --data-path ../../dev_data \
    --fusion prompt \
    --use-comet \
    --kg-path ./kg_output_comet_full \
    --llm-type openai \
    --llm-model deepseek-chat \
    --api-base https://api.deepseek.com \
    --api-key YOUR_API_KEY \
    --output results_kg_comet_full_dev.json \
    --submission-file submission_kg_comet_full_dev.jsonl
```

### Run QA on Test Set (with COMET KG)

```bash
python run_baseline3.py qa \
    --data-path ../../test_data \
    --fusion prompt \
    --use-comet \
    --kg-path ./kg_output_comet_full \
    --llm-type openai \
    --llm-model deepseek-chat \
    --api-base https://api.deepseek.com \
    --api-key YOUR_API_KEY \
    --output results_kg_comet_full_test.json \
    --submission-file submission_kg_comet_full_test.jsonl
```

---

*Last Updated: 2026-01-23*
