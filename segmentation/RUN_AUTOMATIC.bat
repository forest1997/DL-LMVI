@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
"%~dp0Environment\python.exe" "%~dp0run_automatic.py" %*
pause
