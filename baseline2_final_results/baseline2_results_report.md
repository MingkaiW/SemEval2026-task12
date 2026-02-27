
# Baseline2 最终结果与最佳模型

## 最终采用的最佳模型
- 模型：DeBERTa v3 base（判别式 MCQA）
- 模型路径：single_modality/baseline2/models/deberta_v3_base

## Dev 集评估结果
- 结果文件：single_modality/baseline2/results_deberta_v3_base_dev.json
- 指标（400样本）：
  - Score：0.725
  - 完全匹配率：49.5%（198/400）
  - 部分匹配率：46.0%（184/400）
  - 错误率：4.5%（18/400）

## 最终保留的 Test 结果
- Test 结果：single_modality/baseline2/results_deberta_v3_base_test.json
- Test提交文件：submissions/baseline2/deberta_v3_base_test.jsonl
- 说明：测试集无标签，无法评估分数，仅用于生成提交文件。

## 选据依据（简述）
- 在 dev 集评估中，DeBERTa v3 base 为当前最佳表现的判别式模型，因此用于最终 Test 预测。

## 最终命令（摘要）
- Test预测：
  python single_modality/baseline2/run_baseline2.py roberta --mode predict --data-path single_modality/baseline2/processed_data/test/roberta_mcqa.jsonl --model-name single_modality/baseline2/models/deberta_v3_base --output single_modality/baseline2/results_deberta_v3_base_test.json --submission-file submissions/baseline2/deberta_v3_base_test.jsonl
