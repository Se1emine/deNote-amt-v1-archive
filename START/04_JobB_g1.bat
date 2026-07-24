@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
title Denote Job B g1
set RUST_LOG=info
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

if not exist "target\release\denote-train.exe" (
  echo ERROR: denote-train.exe not found.
  pause
  exit /b 1
)
if not exist "runs\jobA_phase3\checkpoints\last" (
  echo ERROR: need Job A checkpoint first.
  pause
  exit /b 1
)

set "RESUME_ARGS="
if exist "runs\jobB_g1_phase3\checkpoints\last" (
  set "RESUME_ARGS=--resume runs/jobB_g1_phase3/checkpoints/last"
  echo Resume: runs\jobB_g1_phase3\checkpoints\last
) else (
  echo Fresh start Job B g1
)

echo === Job B g1 === configs\phase3_jobB_g1.toml
.\target\release\denote-train.exe --config configs/phase3_jobB_g1.toml --job job-b --run-dir runs/jobB_g1_phase3 --device cuda %RESUME_ARGS%
echo Exit code: %ERRORLEVEL%
pause
