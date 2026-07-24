@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
title Denote STATUS
echo.
echo === Denote STATUS ===
echo Project: %CD%
echo.

set ALIVE=NO
tasklist /FI "IMAGENAME eq denote-train.exe" 2>nul | find /I "denote-train.exe" >nul
if %ERRORLEVEL%==0 set ALIVE=YES
echo denote-train process: %ALIVE%

set N=0
for /f %%A in ('dir /b "data\cache\phase3\latents\*.lat.bin" 2^>nul ^| find /c /v ""') do set N=%%A
echo L0 latents: %N% / 15504

if exist "data\cache\phase3\manifest.jsonl" (
  echo L0 manifest: YES
) else (
  echo L0 manifest: NO
)

if exist "runs\jobA_phase3\checkpoints\last" (
  echo Job A last ckpt: YES
) else (
  echo Job A last ckpt: NO
)

if exist "target\release\denote-train.exe" (
  echo denote-train.exe: YES
) else (
  echo denote-train.exe: NO - build with cargo
)

echo.
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
  echo --- GPU ---
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
) else (
  echo nvidia-smi not found
)

echo.
echo Last L0 err log tail if any:
for /f "delims=" %%F in ('dir /b /o-d "runs\phase3_logs\l0_d1024_*.err" 2^>nul') do (
  echo --- runs\phase3_logs\%%F ---
  powershell -NoProfile -Command "Get-Content 'runs\phase3_logs\%%F' -Tail 8 -ErrorAction SilentlyContinue"
  goto done_log
)
:done_log
echo.
pause
