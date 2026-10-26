@echo off
setlocal EnableExtensions EnableDelayedExpansion
title TheSafeExit - Demo Launcher

REM Run from the repository root (the folder containing src\, experiments\, demo\).
cd /d "%~dp0.."

set "PY=%CD%\.venv\Scripts\python.exe"
set "PORT=5050"
set "URL=http://127.0.0.1:%PORT%"
set "MPLBACKEND=Agg"

echo.
echo  ==============================================================
echo    TheSafeExit  -  demo launcher
echo  ==============================================================
echo.

if not exist "%PY%" (
  echo  [ERROR] Python virtual environment not found:
  echo          %PY%
  echo.
  echo  Create it once with:
  echo    python -m venv .venv
  echo    .venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-web.txt
  echo.
  pause
  exit /b 1
)

echo   Editor      %URL%
echo   Showcase    .venv\Scripts\python.exe demo\showcase.py   (under a minute)
echo   Evidence    experiments\out\  +  docs\RESULTS.md
echo.

REM ---------------------------------------------------------------------------
REM Optional: regenerate the experiment evidence. Skipped by default so the
REM demo starts immediately; press Y within 5 seconds to run the quick suite.
REM ---------------------------------------------------------------------------
if not exist "experiments\out\results.csv" (
  echo  [WARN] experiments\out\results.csv is missing.
  echo         Run: .venv\Scripts\python.exe experiments\run_all.py --quick
  echo.
  goto start_editor
)

choice /C YN /N /T 5 /D N /M " Regenerate experiment evidence now (run_all.py --quick, about 6 min)? [Y/N, default N in 5 s] "
if errorlevel 2 (
  echo  Skipped. Using the existing evidence in experiments\out\.
  goto start_editor
)

echo.
echo  Running experiments\run_all.py --quick ...
"%PY%" experiments\run_all.py --quick
if errorlevel 1 (
  echo.
  echo  [ERROR] The quick experiment run failed. The editor will still start.
  echo          See experiments\out\ for the partial results.
  echo.
)

:start_editor
REM ---------------------------------------------------------------------------
REM Start the Flask editor unless something is already listening on the port.
REM ---------------------------------------------------------------------------
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo  A server is already listening on port %PORT% - reusing it.
  goto open_browser
)

echo.
echo  Starting the web editor ...
start "TheSafeExit Web Editor" "%PY%" demo\web_editor\app.py

REM Wait until the server answers, at most ~20 seconds.
set /a tries=0
:wait_loop
set /a tries+=1
"%PY%" -c "import urllib.request,sys; urllib.request.urlopen('%URL%/', timeout=1)" >nul 2>&1
if not errorlevel 1 goto open_browser
if %tries% GEQ 20 (
  echo  [WARN] The editor did not answer within 20 seconds.
  echo         Check the "TheSafeExit Web Editor" window for an error.
  goto open_browser
)
ping -n 2 127.0.0.1 -w 1000 >nul 2>&1
goto wait_loop

:open_browser
echo  Opening %URL% ...
start "" "%URL%"
echo.
echo  ==============================================================
echo    Demo running. Close the "TheSafeExit Web Editor" window to stop.
echo  ==============================================================
echo.
pause
endlocal
