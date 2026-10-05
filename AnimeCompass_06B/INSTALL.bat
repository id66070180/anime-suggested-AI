@echo off
setlocal
cd /d "%~dp0"
echo Anime Compass - first-time setup (Internet required)
if exist ".venv\Scripts\python.exe" goto dependencies
py -3.11 --version >nul 2>&1
if errorlevel 1 goto fallback
py -3.11 -m venv .venv
if errorlevel 1 goto fail
goto dependencies
:fallback
python -c "import sys; assert sys.version_info[:2] == (3,11), 'Install Python 3.11 (64-bit) first'"
if errorlevel 1 goto fail
python -m venv .venv
if errorlevel 1 goto fail
:dependencies
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements-locked.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" VERIFY.py
if errorlevel 1 goto fail
echo Setup complete. Double-click START.bat to open the website.
pause
exit /b 0
:fail
echo Setup failed. Read the error above and README_TH.md.
pause
exit /b 1
