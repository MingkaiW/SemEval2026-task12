param(
    [string]$EnvVarName = "OPENAI_API_KEY"
)

Write-Host "将通过交互方式设置 $EnvVarName，不会在历史中记录明文 key。" -ForegroundColor Cyan

try {
    # 尽量关闭本会话的历史持久化（命令本身仍可能被记录，但不包含 key 内容）
    Set-PSReadLineOption -HistorySaveStyle SaveNothing -ErrorAction SilentlyContinue
} catch {
    # 在某些环境下 PSReadLine 不可用，忽略错误
}

# 安全读取，不回显明文
$secure = Read-Host "请输入 $EnvVarName：" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)

try {
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr)
} finally {
    # 清理非托管内存，减少明文驻留时间
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
}

Set-Item -Path "Env:$EnvVarName" -Value $plain

Write-Host "$EnvVarName has been set for this PowerShell session (not persisted to disk)." -ForegroundColor Green
