@echo off
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Falta .venv. Crea el entorno Python 3.11.9 e instala requirements.txt antes de iniciar.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0web.py" --puerto 8765
pause
