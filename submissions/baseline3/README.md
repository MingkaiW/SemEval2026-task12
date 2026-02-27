# Baseline 3: deepseek

## Submission Info

- **Submission Time**: 2026-01-23 11:12:43
- **Number of Entries**: 612
- **Source File**: `single_modality/baseline3/results_deepseek_test.json`

## Approach: LLM Baseline

**Model**: DeepSeek Chat
**Method**: Prompt-based fusion
**Configuration**:
- LLM model: deepseek-chat
- Fusion method: prompt
- Use COMET: No

**Description**:
This baseline uses the DeepSeek Chat model with prompt-based fusion. The model receives
the target event and candidate options, then predicts causal relations through
natural language reasoning without external commonsense knowledge (COMET disabled).


## Files

- `submission.zip` - Submission package for Codabench
  - Contains `submission/submission.jsonl`

## How to Submit

1. Upload `submission.zip` to Codabench: https://www.codabench.org/competitions/12440/
