@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run INSTALL.bat first.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8502 in your browser. Keep this window open.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --server.headless false --server.fileWatcherType none
if errorlevel 1 pause
