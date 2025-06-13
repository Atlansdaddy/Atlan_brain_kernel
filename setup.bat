@echo off
echo ===================================
echo Atlan Brain Kernel - Windows Setup
echo ===================================
echo.

REM Create directories
echo Creating project structure...
mkdir experiments 2>nul
mkdir visualizations 2>nul
mkdir docs\tutorials 2>nul

REM Create requirements.txt
echo Creating requirements.txt...
(
echo numpy^>=1.19.0
echo matplotlib^>=3.3.0
echo pytest^>=6.0
) > requirements.txt

REM Create __init__.py
echo. > experiments\__init__.py

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Run: pip install -r requirements.txt
echo 2. Copy the Python files into this directory
echo 3. Run: python atlan_brain_kernel.py
echo.
pause