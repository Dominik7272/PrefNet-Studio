@echo off
title PrefNet Studio

if not exist .venv (
    echo [PrefNet] Setting up virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment. Make sure Python is installed and in your PATH.
        pause
        exit /b %errorlevel%
    )
    call .venv\Scripts\activate
    echo [PrefNet] Installing dependencies...
    pip install -r requirements.txt --quiet
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b %errorlevel%
    )
) else (
    call .venv\Scripts\activate
)

python server.py
if %errorlevel% neq 0 (
    echo.
    echo PrefNet Studio exited with error code %errorlevel%.
    pause
)
