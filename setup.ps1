# One-time setup on Windows: a virtualenv, the packages, the models.
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup.ps1 --no-chat
# At the end it offers to save an Anthropic or Gemini key in .env (optional, private).
$ErrorActionPreference = "Stop"
function Stop-Setup($msg) {
    Write-Host ""
    Write-Host "  Setup stopped: $msg" -ForegroundColor Red
    Write-Host "  Fix that, then run the same command again; finished steps are skipped."
    exit 1
}
Set-Location (Join-Path $PSScriptRoot "memgate")
$py = $null
if (Get-Command py -ErrorAction SilentlyContinue) { $py = "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" }
if (-not $py) { Stop-Setup "Python is not installed. Install Python 3.10 or newer from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then open a new terminal." }
& $py -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { Stop-Setup "MemGate needs Python 3.10 or newer (python --version shows yours)." }
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  creating the virtual environment (.venv) ..."
    & $py -m venv .venv
    if ($LASTEXITCODE -ne 0) { Stop-Setup "could not create memgate\.venv." }
}
$vpy = ".\.venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip -q
Write-Host "  installing packages (the first time takes a few minutes) ..."
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Stop-Setup "pip install failed. Check the internet connection; the error is printed above." }
& $vpy setup_models.py @args
if ($LASTEXITCODE -ne 0) { Stop-Setup "a model download failed (see above). Check the internet connection." }
& $vpy set_key.py --if-missing
Write-Host ""
Write-Host "  Setup finished. Start MemGate with:  .\start.bat" -ForegroundColor Green
