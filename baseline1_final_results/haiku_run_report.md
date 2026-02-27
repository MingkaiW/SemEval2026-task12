# Haiku 运行结果汇总

## 概览
本报告整理所有已完成的 Claude Haiku 相关运行结果（Baseline1 与 Baseline3）。

## Baseline1（single_modality/baseline）
### Dev 结果（claude-3-haiku-20240307）
- 结果文件：single_modality/baseline/results_claude_haiku_dev.json
- 指标（400 样本）：
  - Score：0.3925
  - 完全匹配率：0.3150（126/400）
  - 部分匹配率：0.1550（62/400）
  - 错误率：0.5300（212/400）

### Test 结果（claude-3-haiku-20240307）
- 结果文件：single_modality/baseline/results_claude_haiku_test.json
- 提交文件：submissions/baseline1/anthropic_haiku_test.jsonl
- 说明：测试集无标注，评估分数为 0（仅用于生成提交文件）。

## Baseline3（single_modality/baseline3）
### Dev 结果（KG + LLM，claude-3-haiku-20240307）
- 结果文件：single_modality/baseline3/results/kg_llm_dev_anthropic_haiku.json
- 配置：fusion=prompt，use_comet=false
- 指标（400 样本）：
  - Score：0.4625
  - 完全匹配率：0.3325（133/400）
  - 部分匹配率：0.2600（104/400）
  - 错误率：0.4075（163/400）

## 备注
- Baseline3 的 test 未运行（此前被取消/未执行）。
- 若需继续：可用 Haiku 运行 Baseline3 test 或重新跑 dev/test。