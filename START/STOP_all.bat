@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
title Denote STOP
echo Stopping denote-train and L0 watchdog...
taskkill /IM denote-train.exe /F 2>nul
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='powershell.exe'\" | Where-Object { $_.CommandLine -match 'watch_l0' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output ('killed watchdog pid ' + $_.ProcessId) }"
echo Done.
pause
