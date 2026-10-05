@echo off
title FLUX Market API
echo Starting FLUX backend on http://localhost:8000 ...
echo Press Ctrl+C to stop.
echo.
cd /d "%~dp0.."
uvicorn backend.main:app --reload --port 8000
