# Baseline2 最终结果与最佳模型

## 最终采用的最佳模型
- 模型：DeBERTa v3 base（判别式 MCQA）
- 模型路径：single_modality/baseline2/models/deberta_v3_base

## 最终保留的结果
- Test 结果：single_modality/baseline2/results_deberta_v3_base_test.json
- Test 提交文件：submissions/baseline2/deberta_v3_base_test.jsonl

## 选择依据（简述）
- 在 dev 集评估中，DeBERTa v3 base 为当前最佳表现的判别式模型，因此用于最终 Test 预测。

## 最终命令（摘要）
- Test 预测：
  - python single_modality/baseline2/run_baseline2.py roberta --mode predict --data-path single_modality/baseline2/processed_data/test/roberta_mcqa.jsonl --model-name single_modality/baseline2/models/deberta_v3_base --output single_modality/baseline2/results_deberta_v3_base_test.json --submission-file submissions/baseline2/deberta_v3_base_test.jsonl
