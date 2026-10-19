@echo off
rem Start MemGate and open it in the browser. Extra options go to app.py,
rem e.g.  start.bat --llm claude   or   start.bat --port 9000
cd /d "%~dp0memgate"
if not exist .venv\Scripts\python.exe (
  echo Run setup.ps1 first.
  exit /b 1
)
.venv\Scripts\python.exe app.py %*
