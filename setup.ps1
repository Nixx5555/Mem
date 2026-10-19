# One-time setup on Windows: a virtualenv, the packages, the models.
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup.ps1 --no-chat
# At the end it offers to save an Anthropic key in .env (optional, private).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "memgate")
$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
& $py -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { Write-Error "MemGate needs Python 3.10 or newer." }
if (-not (Test-Path ".venv")) { & $py -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip -q
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed." }
& .\.venv\Scripts\python.exe setup_models.py @args
if ($LASTEXITCODE -ne 0) { Write-Error "Model download failed; run setup.ps1 again." }
& .\.venv\Scripts\python.exe set_key.py --if-missing
Write-Host ""
Write-Host "Setup finished. Start MemGate with start.bat"
