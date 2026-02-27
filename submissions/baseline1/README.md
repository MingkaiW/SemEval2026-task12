# Baseline 1: qwen3

## Submission Info

- **Submission Time**: 2026-01-23 11:12:43
- **Number of Entries**: 612
- **Source File**: `single_modality/baseline/results_qwen3_test.json`

## Approach: Qwen3 LLM Baseline

**Model**: Qwen3 1.7B (via Ollama)
**Method**: Direct prompting with context
**Configuration**:
- Model type: Ollama local inference
- Model name: qwen3:1.7b
- Use context: Yes
- Task: Multi-label causal relation identification

**Description**:
This baseline uses the Qwen3 1.7B language model to directly predict causal relations
between events. The model receives the target event, candidate options (A-D), and
relevant context documents, then outputs which options are valid causes of the target event.


## Files

- `submission.zip` - Submission package for Codabench
  - Contains `submission/submission.jsonl`

## How to Submit

1. Upload `submission.zip` to Codabench: https://www.codabench.org/competitions/12440/
