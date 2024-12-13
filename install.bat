@echo off

REM Define paths
SET REQUIREMENTS=requirements.txt
SET PYTHON_SCRIPT=yolov11.py
SET DESKTOP=%USERPROFILE%\OneDrive\Desktop
SET SHORTCUT_NAME=Object Detector for Robotic Bin-Picking

REM Install requirements
echo Installing dependencies from %REQUIREMENTS%...
pip install -r %REQUIREMENTS%

REM Check if pip install succeeded
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies. Exiting...
    exit /b 1
)
