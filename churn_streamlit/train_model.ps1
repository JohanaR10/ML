param(
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,

    [string]$Sep = "|",
    [string]$Encoding = "utf-16"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\johan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot ".churn-colab-pydeps"

Set-Location $PSScriptRoot
& $PythonExe train_churn_model.py --csv $CsvPath --sep $Sep --encoding $Encoding
