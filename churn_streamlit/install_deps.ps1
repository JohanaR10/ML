$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\johan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$DepsPath = Join-Path $ProjectRoot ".churn-colab-pydeps"

& $PythonExe -m pip install -r (Join-Path $PSScriptRoot "requirements.txt") --target $DepsPath
