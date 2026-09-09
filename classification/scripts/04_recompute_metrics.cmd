@echo off
setlocal
set "PACKAGE_ROOT=%~dp0.."
set "PYTHON_EXE=python"
if not "%~1"=="" set "PYTHON_EXE=%~1"
pushd "%PACKAGE_ROOT%"
"%PYTHON_EXE%" "src\evaluate_results.py"
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
