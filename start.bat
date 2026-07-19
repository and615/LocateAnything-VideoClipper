@echo off
cd /d "%~dp0"

echo ============================================
echo   LocateAnything-VideoClipper
echo ============================================
echo.

start /b embedded_python\python.exe -u app.py

ping 127.0.0.1 -n 6 >nul
start http://localhost:7860

echo  Server running at http://localhost:7860
echo  Press Ctrl+C to stop
echo.
pause
