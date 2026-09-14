@echo off
echo =======================================================================
echo Starting TheSafeExit Mid-Sem Demo Execution
echo =======================================================================

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Run the test suite first to verify invariants
echo Running pytest test suite...
pytest -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Unit tests failed! Please check code issues.
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: Run the live demo script
echo Running mid-semester live demo script...
python demo/mid_sem_demo.py

echo.
echo =======================================================================
echo Execution Completed. Figures are saved in "demo/figures/"
echo =======================================================================
pause
