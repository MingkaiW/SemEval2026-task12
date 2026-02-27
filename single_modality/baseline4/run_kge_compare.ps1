param(
    [string]$PythonExe = "D:/Anaconda/envs/dl310/python.exe",
    [string]$LlmModel = "deepseek-chat"
)

# 保证在 baseline4 目录中运行
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[Baseline4] Step 1: Train RotatE (Euclidean) embeddings on existing COMET KG..." -ForegroundColor Cyan
# 默认使用 kg_output_hyper -> kg_output_euclid；如需调参可直接修改下行命令
& $PythonExe train_rotate_on_hyper_kg.py `
  --kg-dir ./kg_output_hyper `
  --out-dir ./kg_output_euclid `
  --embedding-dim 32 `
  --epochs 30 `
  --batch-size 256
if ($LASTEXITCODE -ne 0) {
  Write-Host "[Baseline4] RotatE training failed with exit code $LASTEXITCODE" -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "[Baseline4] Step 2: Run KGE-QA with RotatE (Euclidean) embeddings..." -ForegroundColor Cyan
& $PythonExe run_baseline4.py kge-qa `
  --data-path ../../dev_data `
  --kg-dir ./kg_output_euclid `
  --kg-model-type RotatE `
  --llm-model $LlmModel `
  --api-base https://api.deepseek.com `
  --output ./results_kge_rotate_dev.json `
  --submission-file ./submission_kge_rotate_dev.jsonl
if ($LASTEXITCODE -ne 0) {
  Write-Host "[Baseline4] RotatE KGE-QA failed with exit code $LASTEXITCODE" -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "[Baseline4] Step 3: Run KGE-QA with RotH (Hyperbolic) embeddings..." -ForegroundColor Cyan
& $PythonExe run_baseline4.py kge-qa `
  --data-path ../../dev_data `
  --kg-dir ./kg_output_hyper `
  --kg-model-type RotH `
  --llm-model $LlmModel `
  --api-base https://api.deepseek.com `
  --output ./results_kge_roth_dev.json `
  --submission-file ./submission_kge_roth_dev.jsonl
if ($LASTEXITCODE -ne 0) {
  Write-Host "[Baseline4] RotH KGE-QA failed with exit code $LASTEXITCODE" -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "[Baseline4] Euclidean vs Hyperbolic KGE comparison on dev_data finished." -ForegroundColor Green
