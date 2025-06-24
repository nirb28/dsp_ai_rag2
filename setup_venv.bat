@echo off
echo Setting up RAG as a Service virtual environment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo Please edit .env file and add your Groq API key
    echo.
)

echo.
echo Setup complete!
echo.
echo To activate the virtual environment in the future, run:
echo .venv\Scripts\activate.bat
echo.
echo To start the server, run:
echo python -m app.main
echo.
echo To run tests, run:
echo pytest
echo.
pause
