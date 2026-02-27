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
    - 详见 result.md，结果为历史最佳配置，非本轮主动运行。

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


---

**Baseline3 配置选择过程与理由总结：**

1. 实验先后尝试了三种知识图谱（KG）方案：
    - Simple KG（基础因果知识库，得分0.5713）
    - RotatE Embedding（无COMET，得分0.5787）
    - COMET增强KG（COMET-atomic_2020_BART，得分0.6088）

2. 由于 PyTorch 2.5.1 安全限制，COMET模型一度无法加载，后通过 use_safetensors=True 技术修复，成功启用COMET知识增强。

3. 结果对比显示，COMET增强KG显著提升了分数和完全匹配率，错误率也进一步降低：
    - Score从0.5787提升到0.6088（+5.2%）
    - 完全匹配率从42.00%提升到46.25%
    - 错误率从26.25%降至24.50%
    - KG三元组数量提升89倍，实体数提升6.8倍，知识密度大幅提升

4. 结论：更丰富的常识知识（如COMET）和更密集的知识图谱结构，是提升推理能力的关键。最终推荐 DeepSeek-chat + KG + COMET 作为 Baseline3 的最佳配置。

5. 未来建议：尝试更大规模LLM、检索式融合、领域微调COMET、多模型融合等方向。

If you need more detailed prediction statistics or other charts, please specify.
