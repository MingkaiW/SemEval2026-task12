param(
    [string]$PythonExe = "D:/Anaconda/envs/dl310/python.exe",
    [string]$LlmModel = "gpt-4o-mini"
)

# 脚本所在目录 = baseline4 目录
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[Baseline4] Running QA on dev_data..." -ForegroundColor Cyan
& $PythonExe run_baseline4.py qa `
  --data-path ../../dev_data `
  --fusion prompt `
  --llm-type openai `
  --llm-model $LlmModel `
  --use-comet `
  --output ./results_baseline4_dev.json `
  --submission-file ./submission_baseline4_dev.jsonl

if ($LASTEXITCODE -ne 0) {
  Write-Host "[Baseline4] dev QA failed with exit code $LASTEXITCODE" -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "[Baseline4] Running QA on test_data..." -ForegroundColor Cyan
& $PythonExe run_baseline4.py qa `
  --data-path ../../test_data `
  --fusion prompt `
  --llm-type openai `
  --llm-model $LlmModel `
  --use-comet `
  --output ./results_baseline4_test.json `
  --submission-file ./submission_baseline4_test.jsonl

if ($LASTEXITCODE -ne 0) {
  Write-Host "[Baseline4] test QA failed with exit code $LASTEXITCODE" -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "[Baseline4] All QA runs finished." -ForegroundColor Green
