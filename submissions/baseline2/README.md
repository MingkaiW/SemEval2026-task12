# Baseline 2: unifiedqa

## Submission Info

- **Submission Time**: 2026-01-23 11:12:43
- **Number of Entries**: 612
- **Source File**: `single_modality/baseline2/results_unifiedqa_test.json`

## Approach: UnifiedQA Baseline

**Model**: allenai/unifiedqa-t5-small
**Method**: Question-answering format
**Configuration**:
- Model name: allenai/unifiedqa-t5-small
- Input format: QA-style with context

**Description**:
This baseline uses the UnifiedQA model (T5-small variant) which is trained on multiple
QA datasets. The causal identification task is reformatted as a question-answering problem
where the model must identify which events caused the target event.


## Files

- `submission.zip` - Submission package for Codabench
  - Contains `submission/submission.jsonl`

## How to Submit

1. Upload `submission.zip` to Codabench: https://www.codabench.org/competitions/12440/
