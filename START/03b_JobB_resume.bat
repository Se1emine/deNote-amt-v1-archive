@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
title Denote Job B RESUME
set RUST_LOG=info
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

REM Usage:
REM   03b_JobB_resume.bat g0
REM   03b_JobB_resume.bat g0 last
REM   03b_JobB_resume.bat g0 step_000500
REM   03b_JobB_resume.bat single

set "G=%~1"
if "%G%"=="" set "G=g0"
set "STEP=%~2"
if "%STEP%"=="" set "STEP=last"

if /I "%G%"=="single" (
  set "CFG=configs\phase3_jobB_single.toml"
  set "RUN=runs\jobB_single_phase3"
) else if /I "%G%"=="g0" (
  set "CFG=configs\phase3_jobB_g0.toml"
  set "RUN=runs\jobB_g0_phase3"
) else if /I "%G%"=="g1" (
  set "CFG=configs\phase3_jobB_g1.toml"
  set "RUN=runs\jobB_g1_phase3"
) else if /I "%G%"=="g2" (
  set "CFG=configs\phase3_jobB_g2.toml"
  set "RUN=runs\jobB_g2_phase3"
) else if /I "%G%"=="g3" (
  set "CFG=configs\phase3_jobB_g3.toml"
  set "RUN=runs\jobB_g3_phase3"
) else (
  echo Unknown group: %G%  use g0 g1 g2 g3 single
  pause
  exit /b 1
)

set "CKPT=%RUN%\checkpoints\%STEP%"
if not exist "%CKPT%" (
  echo ERROR: %CKPT% not found
  dir /b "%RUN%\checkpoints" 2>nul
  pause
  exit /b 1
)

echo === Job B RESUME group=%G% ckpt=%CKPT% ===
.\target\release\denote-train.exe --config %CFG% --job job-b --run-dir %RUN% --device cuda --resume "%CKPT:\=/%"
echo Exit code: %ERRORLEVEL%
pause
