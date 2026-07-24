============================================================
  DENOTE — двойной щелчок = запуск этапа (d_model=1024)
============================================================

Порядок (сверху вниз):

  00_STATUS.bat          — проверить: жив ли train, сколько latents
  01_L0_cache.bat        — кэш latents (нужен data/raw/phase3)
  01b_L0_watchdog.bat    — L0 + авто-перезапуск каждые 15 мин
  02_JobA.bat            — Job A (presence+adapter), после L0
  03_JobB_g0.bat         — Job B группа 0
  04_JobB_g1.bat         — Job B группа 1
  05_JobB_g2.bat         — Job B группа 2
  06_JobB_g3.bat         — Job B группа 3
  07_JobB_single.bat     — Job B single (альтернатива curriculum g0–g3)
  STOP_all.bat           — убить denote-train и L0-watchdog

Не закрывай чёрное окно, пока этап не закончится.
В конце окно ждёт Enter — можно прочитать ошибки.

Готовность L0: latents ~15504 + файл
  data\cache\phase3\manifest.jsonl

Job A val: val_every=100 остаётся; eval = no-grad + max_val_batches=32
  (configs\phase3_jobA_ft.toml). После фикса кода: cargo build -p denote-train --release --features cuda

Логи:
  runs\phase3_logs\
  runs\jobA_phase3\
  runs\jobB_*_phase3\

Сырые данные НЕ трогать: data\raw\phase3
============================================================
