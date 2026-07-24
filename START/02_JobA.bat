@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
title Denote Job A
set RUST_LOG=info
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

if not exist "target\release\denote-train.exe" (
  echo ERROR: target\release\denote-train.exe not found.
  echo Build: cargo build -p denote-train --release --features cuda
  echo Current dir: %CD%
  pause
  exit /b 1
)

if not exist "data\cache\phase3\manifest.jsonl" (
  echo ERROR: L0 not finished - missing data\cache\phase3\manifest.jsonl
  pause
  exit /b 1
)

set "RESUME_ARGS="
if exist "runs\jobA_phase3\checkpoints\last\adapter.mpk" (
  set "RESUME_ARGS=--resume runs/jobA_phase3/checkpoints/last"
  echo Resume: runs\jobA_phase3\checkpoints\last
) else if exist "runs\jobA_phase3\checkpoints\state.json" (
  set "RESUME_ARGS=--resume runs/jobA_phase3/checkpoints/last"
  echo Resume: last from state.json
) else (
  echo Fresh start Job A
)

echo === Job A d_model=1024 max_steps from config ===
echo Config: configs\phase3_jobA_ft.toml
echo Run dir: runs\jobA_phase3
echo Do not close this window.
echo.

.\target\release\denote-train.exe --config configs/phase3_jobA_ft.toml --job job-a --run-dir runs/jobA_phase3 --device cuda %RESUME_ARGS%

echo.
echo Exit code: %ERRORLEVEL%
echo Checkpoints: runs\jobA_phase3\checkpoints\
echo.
pause
