# denote-ai-model

Локальная нейросеть для автоматической транскрипции нот (**pitch / start / end / duration**)  
на **Rust + Burn**.

Совместима по контракту с deNote (`NoteEvent`: start/end, midi pitch, velocity).

## Архитектура: Denote-MIROS-v1 (primary)

Ближе к **MIROS / YourMT3 / MT3**, **не** к Basic Pitch:

| | |
|--|--|
| Вход | wav **24 kHz** → log-mel 128 bands (MusicFM frontend) |
| Encoder | Conv frontend + Conformer/Transformer stack → latents ~**25 Hz** |
| Adapter ★ | program conditioning + recurrent adapter + **aux pitch/onset** |
| Decoder | T5-style **seq2seq** → **MIDI-event tokens** (не piano-roll grid) |
| Decode ★ | FSM-constrained AR + detokenize → `NoteEvent` |

★ улучшения относительно «чистого» MIROS (anti-leakage FSM, aux multi-task, stem/domain later).

```text
wav → musicfm mel → MusicFmEncoder → InstrumentAdapter → TokenDecoder
                                                      ↓
                                              token ids → NoteEvent[]
```

Токены: `BOS → [Program] → (TimeShift | Velocity+NoteOn | NoteOff)* → EOS`.

### Legacy: Denote-CRNN-v1

Onsets-and-Frames CRNN (onset/frame/offset) остаётся за feature `legacy-crnn` для ablations.  
Не основной путь.

## Workspace

```text
crates/
  denote-audio/     # load, resample, STFT, mel (+ MusicFM 24 kHz path)
  denote-tokens/    # vocab, encode/decode notes↔tokens, FSM
  denote-encoder/   # MusicFM-compat encoder (conv + transformer stack)
  denote-data/      # token cache + legacy piano-roll cache
  denote-model/     # DenoteMiros (+ optional legacy CRNN)
  denote-decode/    # tokens→notes (+ legacy frame decode)
  denote-eval/      # note F1 (mir_eval-style)
  denote-train/     # training binary
  denote-infer/     # wav → JSON/MIDI
configs/
  miros_v1_smoke.toml
  v1_synth.toml           # legacy CRNN
  v1_guitarset_ft.toml    # legacy CRNN
tools/
  convert_musicfm.py      # one-shot PT→numpy dump (not runtime)
```

## Планы

| | |
|--|--|
| Master | [`PLAN_Denote_Free_Modular.md`](PLAN_Denote_Free_Modular.md) |
| **Phase 0** (platform proof) | [`PLAN_Phase0_Platform_Proof.md`](PLAN_Phase0_Platform_Proof.md) |
| **Phase 1** (cache + Job A) | [`PLAN_Phase1_Cache_JobA.md`](PLAN_Phase1_Cache_JobA.md) · [`scripts/job_L0_cache.md`](scripts/job_L0_cache.md) · [`scripts/job_A.md`](scripts/job_A.md) |
| **Phase 2** (modular decoders B_g) | [`PLAN_Phase2_Modular_Decoders.md`](PLAN_Phase2_Modular_Decoders.md) · [`scripts/job_B.md`](scripts/job_B.md) |
| **Phase 3** (data scale + F1 vs MIROS) | [`PLAN_Phase3_Data_F1_vs_MIROS.md`](PLAN_Phase3_Data_F1_vs_MIROS.md) · [`scripts/data_scale.md`](scripts/data_scale.md) · [`scripts/eval_challenge_f1.md`](scripts/eval_challenge_f1.md) |
| 1070 runbook | [`scripts/train_1070.md`](scripts/train_1070.md) |
| Later | Phase 4–5 plan files at repo root |

## Быстрый старт (CPU smoke + Phase 0 mini)

```powershell
# из корня denote-ai-model
$env:Path = "$env:USERPROFILE\.cargo\bin;" + $env:Path
cargo test --workspace --lib

# Denote-MIROS seq2seq smoke (primary)
cargo run -p denote-train --release -- --config configs/miros_v1_smoke.toml --smoke --device cpu

# Phase 0 mini-train: 100 TF steps + JSONL + checkpoints
cargo run -p denote-train --release -- `
  --config configs/phase0_mini.toml --mini-steps 100 --device cpu `
  --run-dir runs/phase0_mini_cpu

# TensorBoard from JSONL (Windows: py)
py tools/jsonl_to_tb.py --jsonl runs/phase0_mini_cpu/metrics.jsonl --out runs/phase0_mini_cpu/tb
# py -m pip install tensorboardX tensorboard; py -m tensorboard --logdir runs --port 6006

# Legacy CRNN smoke
cargo run -p denote-train -- --config configs/v1_synth.toml --smoke

# Phase 1: mini cache + Job A (adapter+presence, no encoder)
py -3 tools/build_phase1_mini.py --out data/raw/phase1_mini --n 50
cargo run -p denote-train --release -- --config configs/free_1070_cache.toml --job cache-l0 --device cpu
cargo run -p denote-train --release -- --config configs/free_1070_jobA.toml --job job-a `
  --run-dir runs/jobA_smoke --device cpu --max-steps 200

# Phase 3: corpus (Rust + FluidSynth CLI) → L0 → Job A/B curriculum → Job E F1
# See scripts/data_scale.md — no Python data pipeline.
cargo run -p denote-train --release -- --config configs/phase3_corpus.toml --job phase3-data
cargo run -p denote-train --release --features cuda -- --config configs/phase3_cache.toml --job cache-l0 --device cuda
cargo run -p denote-train --release --features cuda -- --config configs/phase3_jobA_ft.toml --job job-a --device cuda
# then phase3_jobB_g0..g3.toml; Job E:
cargo run -p denote-eval --release -- `
  --manifest data/cache/phase3/manifest.jsonl --split val `
  --job-a runs/jobA_phase3/checkpoints/last `
  --decoders runs/jobB_g0_phase3/checkpoints/last `
  --out runs/eval_phase3
```

MIROS smoke:
1. синтетический sine 1 s @ 24 kHz + одна MIDI-нота
2. teacher-forcing CE + AdamW step

GPU (optional, see `scripts/train_1070.md`):

```powershell
cargo run -p denote-train --release --features cuda -- --mini-steps 100 --device cuda --config configs/phase0_mini.toml
# or: --features wgpu --device wgpu
```

## Обучение (план)

1. **Phase 0** — platform proof (this repo stage): mini-train, RunLogger, resume, TB  
2. Synth / MIDI-render → token cache (`data/cache/tokens/`) + latent cache  
3. Freeze encoder, train adapter+decoder (modular jobs)  
4. Optional LoRA / full FT на большой GPU  
5. GuitarSet / stem domain finetune  

```powershell
# (после data pipeline — Phase 1+)
cargo run -p denote-train --release -- --config configs/miros_v1_synth.toml
```

**VRAM:** default = freeze MusicFM-scale encoder; full FT needs 24 GB+. Smoke ~1.3M params (tiny config).

## MusicFM weights

Официальные веса: [minzwon/musicfm](https://github.com/minzwon/musicfm) (PyTorch).  
Конвертация (offline, не runtime):

```powershell
python tools/convert_musicfm.py --ckpt pretrained_msd.pt --stats msd_stats.json --out assets/musicfm_msd
```

Bit-exact Conformer load — следующий этап (сейчас shape-compatible Burn stack + random init / smoke).

## Данные

См. `DATA_TRAINING_GUIDE.md`. Нужны **размеченные** пары audio+notes (GuitarSet, synth MIDI, …).  
Demucs-stem finetune — 2-й этап с **теми же** лейблами.

## Метрики

`denote-eval`: note-onset F1 @ 50 ms, note-with-offset F1.

## Интеграция с deNote

Позже: checkpoint / ONNX → Tauri рядом с pipeline Demucs; тот же `NoteEvent`.

