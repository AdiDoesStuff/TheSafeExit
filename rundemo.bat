@echo off
setlocal

echo =======================================================================
echo Starting TheSafeExit Demo Suite
echo =======================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Could not find .venv\Scripts\python.exe
    echo Please create the virtual environment and install requirements first:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

set "PYTHON=.venv\Scripts\python.exe"
set "MPLBACKEND=Agg"

echo Running pytest test suite...
"%PYTHON%" -m pytest -v --basetemp .pytest_tmp
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Unit tests failed. Please check the output above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================================
echo Running Mid-Sem Demo
echo =======================================================================
"%PYTHON%" demo\mid_sem_demo.py
if %ERRORLEVEL% NEQ 0 goto demo_failed

echo.
echo =======================================================================
echo Running Phase 2 Demo
echo =======================================================================
"%PYTHON%" demo\demo_phase2.py
if %ERRORLEVEL% NEQ 0 goto demo_failed

echo.
echo =======================================================================
echo Running Phase 3 Demo
echo =======================================================================
"%PYTHON%" demo\demo_phase3.py
if %ERRORLEVEL% NEQ 0 goto demo_failed

echo.
echo =======================================================================
echo Running Phase 3b Geometry Demo
echo =======================================================================
"%PYTHON%" demo\demo_phase3b.py
if %ERRORLEVEL% NEQ 0 goto demo_failed

echo.
echo =======================================================================
echo Demo suite completed successfully.
echo Figures are saved in demo\figures\
echo =======================================================================
pause
exit /b 0

:demo_failed
echo.
echo [ERROR] A demo command failed. Please check the output above.
pause
exit /b %ERRORLEVEL%
