# SemEval 2026 Task 12 Baseline Results Comparison and Detailed Analysis

## 1. Overall Metrics Comparison

| Baseline Folder           | Model                     | Score   | Exact Match Rate | Partial Match Rate | Error Rate | Sample Size |
|--------------------------|----------------------------|---------|------------------|--------------------|------------|-------------|
| baseline1_final_results  | claude-3-haiku-20240307    | 0.3925  | 31.5%            | 15.5%              | 53.0%      | 400         |
| baseline2_final_results  | DeBERTa v3 base            | 0.725   | 49.5%            | 46.0%              | 4.5%       | 400         |
| baseline3_final_results  | Haiku + KG (no COMET)      | 0.4625  | 33.25%           | 26.0%              | 40.75%     | 400         |
| baseline3_final_results  | DeepSeek-chat + KG + COMET | 0.6088  | 46.25%           | 29.25%             | 24.5%      | 400         |

## 2. Example Predictions

### baseline1_final_results
- q-2020: golden=C, prediction=A
- q-2021: golden=C, prediction=A/C

### baseline2_final_results
- q-2020: golden=C, prediction=C, probability C=0.9995
- q-2021: golden=C, prediction=C, probability C=0.9982

### baseline3_final_results
- Haiku + KG (no COMET):
  - q-2020: golden=C, prediction=A
  - q-2021: golden=C, prediction=C
- DeepSeek-chat + KG + COMET:
  - See result.md for details, this is the historical best configuration, not rerun in this round.

## 3. Chart Description

```mermaid
barChart
    title Baseline Score Comparison
    x-axis Baseline
    y-axis Score
    bar baseline1_final_results: 0.3925
    bar baseline2_final_results: 0.725
    bar baseline3_final_results_haiku: 0.4625
    bar baseline3_final_results_deepseek: 0.6088
```

```mermaid
barChart
    title Exact Match Rate Comparison
    x-axis Baseline
    y-axis Exact Match Rate
    bar baseline1_final_results: 31.5
    bar baseline2_final_results: 49.5
    bar baseline3_final_results_haiku: 33.25
    bar baseline3_final_results_deepseek: 46.25
```

```mermaid
barChart
    title Error Rate Comparison
    x-axis Baseline
    y-axis Error Rate
    bar baseline1_final_results: 53.0
    bar baseline2_final_results: 4.5
    bar baseline3_final_results_haiku: 40.75
    bar baseline3_final_results_deepseek: 24.5
```

---

> Note: The DeepSeek-chat + KG + COMET configuration for Baseline3 is the historical best, not rerun in this round, and the result is summarized from result.md.

---

**Summary of Baseline3 Selection Process and Rationale:**

1. Three KG schemes were tried in sequence:
   - Simple KG (basic causal KB, score 0.5713)
   - RotatE Embedding (no COMET, score 0.5787)
   - COMET-enhanced KG (COMET-atomic_2020_BART, score 0.6088)

2. Due to PyTorch 2.5.1 security restrictions, the COMET model was initially unavailable, but was later enabled by using use_safetensors=True.

3. Results show that COMET-enhanced KG significantly improved the score and exact match rate, and further reduced the error rate:
   - Score increased from 0.5787 to 0.6088 (+5.2%)
   - Exact match rate increased from 42.00% to 46.25%
   - Error rate decreased from 26.25% to 24.50%
   - KG triples increased by 89x, entities by 6.8x, much denser knowledge

4. Conclusion: Richer commonsense knowledge (like COMET) and denser KG structure are key to better reasoning. DeepSeek-chat + KG + COMET is recommended as the best Baseline3 configuration.

5. Future suggestions: Try larger LLMs, retrieval fusion, domain-specific COMET finetuning, and model ensemble.
