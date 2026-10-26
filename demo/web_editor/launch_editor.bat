@echo off
REM TheSafeExit — Web Editor Launcher
REM Starts the Flask backend and opens the browser.

title TheSafeExit Web Editor

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║    TheSafeExit  —  Web Floor Plan Editor ║
echo  ╚══════════════════════════════════════════╝
echo.
echo  Starting Flask server...
echo  Open your browser at:  http://127.0.0.1:5050
echo.
echo  Press Ctrl+C to stop the server.
echo.

REM Navigate to project root so `src` is resolvable
cd /d "%~dp0..\.."

REM Launch the server
.venv\Scripts\python.exe demo\web_editor\app.py

pause
