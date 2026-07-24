@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
title Denote L0 WATCHDOG
set RUST_LOG=info
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

if not exist "target\release\denote-train.exe" (
  echo ERROR: target\release\denote-train.exe not found.
  echo Current dir: %CD%
  pause
  exit /b 1
)

if not exist "runs\phase3_logs" mkdir "runs\phase3_logs"

echo.
echo === L0 WATCHDOG auto-restart every 15 min ===
echo Target: about 15504 latents + manifest.jsonl
echo Log: runs\phase3_logs\watch_l0_d1024.log
echo Close window = stop watchdog. Training process may keep running.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "runs\phase3_logs\watch_l0_d1024.ps1"
echo.
echo Watchdog exited.
pause
