@echo off
title PrefNet Studio
python server.py
if %errorlevel% neq 0 (
    echo.
    echo PrefNet Studio exited with error code %errorlevel%.
    pause
)
