@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
title Denote Job A RESUME
set RUST_LOG=info
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

if not exist "target\release\denote-train.exe" (
  echo ERROR: denote-train.exe not found.
  pause
  exit /b 1
)

REM Optional: pass step folder as first arg, e.g. 02b_JobA_resume.bat step_000050
set "CKPT=runs\jobA_phase3\checkpoints\last"
if not "%~1"=="" set "CKPT=runs\jobA_phase3\checkpoints\%~1"

if not exist "%CKPT%\adapter.mpk" if not exist "%CKPT%.mpk" (
  if not exist "%CKPT%\adapter.mpk" (
    echo ERROR: checkpoint not found: %CKPT%
    echo Examples:
    echo   02b_JobA_resume.bat
    echo   02b_JobA_resume.bat last
    echo   02b_JobA_resume.bat step_000050
    dir /b "runs\jobA_phase3\checkpoints" 2>nul
    pause
    exit /b 1
  )
)

echo === Job A RESUME from %CKPT% ===
echo Note: training will run max_steps MORE from saved step.
echo To only finish remaining budget, set max_steps in toml or:
echo   denote-train ... --resume ... --max-steps 4900
echo.

.\target\release\denote-train.exe --config configs/phase3_jobA_ft.toml --job job-a --run-dir runs/jobA_phase3 --device cuda --resume "%CKPT:\=/%"

echo Exit code: %ERRORLEVEL%
pause
