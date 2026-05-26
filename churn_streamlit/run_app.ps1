$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\johan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot ".churn-colab-pydeps"

Set-Location $PSScriptRoot
& $PythonExe -m streamlit run app_streamlit.py --global.developmentMode false --server.port 8501 --server.headless true
