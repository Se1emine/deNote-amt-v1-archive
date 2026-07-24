@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
title Denote L0 cache
set RUST_LOG=info
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

if not exist "target\release\denote-train.exe" (
  echo ERROR: target\release\denote-train.exe not found.
  echo Build first: cargo build -p denote-train --release --features cuda
  echo Current dir: %CD%
  pause
  exit /b 1
)

if not exist "runs\phase3_logs" mkdir "runs\phase3_logs"

echo.
echo === L0: cache latents d_model=1024 ===
echo Config: configs\phase3_cache.toml
echo Skip existing lat.bin = yes
echo Do not close this window.
echo.

set TS=%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TS=%TS: =0%

.\target\release\denote-train.exe --config configs/phase3_cache.toml --job cache-l0 --device cuda 1>> "runs\phase3_logs\l0_click_%TS%.log" 2>> "runs\phase3_logs\l0_click_%TS%.err"

echo.
echo Exit code: %ERRORLEVEL%
echo Check: 00_STATUS.bat
echo Log: runs\phase3_logs\l0_click_%TS%.err
echo.
pause
