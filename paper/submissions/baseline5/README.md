# Baseline 5: kg_comet_full

## Submission Info

- **Submission Time**: 2026-01-23 14:04:00
- **Number of Entries**: 612
- **Source File**: `single_modality/baseline3/submission_kg_comet_full_test.jsonl`

## Approach: Knowledge Graph + COMET (Full) Baseline

**Model**: DeepSeek Chat + Knowledge Graph + COMET
**Method**: Prompt-based fusion with KG and commonsense enhancement
**Configuration**:
- LLM model: deepseek-chat
- Fusion method: prompt
- Use COMET: Yes (Full)
- KG embeddings: Enhanced with commonsense knowledge

**Description**:
This baseline combines DeepSeek Chat with both knowledge graph embeddings and COMET
(Commonsense Transformers) for enhanced causal reasoning. COMET provides commonsense
knowledge about cause-effect relationships, complementing the structured knowledge from
knowledge graphs. This full configuration leverages multiple knowledge sources:
1. Pre-trained language model (DeepSeek Chat)
2. Knowledge graph embeddings for structured relational knowledge
3. COMET for commonsense causal reasoning

This approach aims to improve causal relation identification by integrating both
explicit (KG) and implicit (COMET) knowledge sources alongside the LLM's
reasoning capabilities.

## Files

- `submission.zip` - Submission package for Codabench
  - Contains `submission/submission.jsonl`
- `submission.jsonl` - Direct JSONL file for inspection

## How to Submit

1. Upload `submission.zip` to Codabench: https://www.codabench.org/competitions/12440/
