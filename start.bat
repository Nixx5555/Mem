@echo off
rem Start MemGate and open it in the browser. Extra options go to app.py,
rem e.g.  start.bat --llm gemini   or   start.bat --port 9000
cd /d "%~dp0memgate"
if not exist .venv\Scripts\python.exe (
  echo.
  echo   MemGate is not set up yet. Run this once, then start again:
  echo     powershell -ExecutionPolicy Bypass -File setup.ps1
  exit /b 1
)
.venv\Scripts\python.exe app.py %*
if errorlevel 1 (
  echo.
  echo   MemGate stopped with an error, printed above.
  echo   If it says the port is in use, close the other MemGate window or run:  start.bat --port 9000
)
