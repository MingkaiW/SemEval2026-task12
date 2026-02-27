# Baseline 4: kg_rotateE

## Submission Info

- **Submission Time**: 2026-01-23 11:12:44
- **Number of Entries**: 612
- **Source File**: `single_modality/baseline3/results_kg_rotateE_test.json`

## Approach: Knowledge Graph + RotatE Baseline

**Model**: DeepSeek Chat + Knowledge Graph Embeddings
**Method**: Prompt-based fusion with KG enhancement
**Configuration**:
- LLM model: deepseek-chat
- Fusion method: prompt
- KG embedding: RotatE

**Description**:
This baseline combines DeepSeek Chat with knowledge graph embeddings using the RotatE
model. The approach leverages structured knowledge from knowledge graphs to enhance
causal relation identification. RotatE embeddings provide relational reasoning
capabilities that complement the LLM's language understanding.


## Files

- `submission.zip` - Submission package for Codabench
  - Contains `submission/submission.jsonl`

## How to Submit

1. Upload `submission.zip` to Codabench: https://www.codabench.org/competitions/12440/
