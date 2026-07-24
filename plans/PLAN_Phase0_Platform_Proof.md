# Phase 0 — Platform proof (GTX 1070)

**Master-план:** [`PLAN_Denote_Free_Modular.md`](PLAN_Denote_Free_Modular.md) §7 Phase 0  
**Цель фазы:** доказать, что train loop **реально крутится** на твоей машине (в идеале GTX 1070), логирует метрики, переживает interrupt/resume, и есть понятные knobs против OOM.  
**Не цель фазы:** MusicFM pretrained weights, latent cache, presence, multi-decoder, F1 vs MIROS (это Phase 1+).

---

## 1. Контекст и baseline в репозитории

Уже есть:

| Компонент | Где | Заметка |
|-----------|-----|---------|
| MIROS smoke | `crates/denote-train/src/miros_smoke.rs` | synthetic 1s, CE + AdamW step, CPU/ndarray |
| Config smoke | `configs/miros_v1_smoke.toml` | tiny model |
| Features | `denote-train`: `cuda`, `wgpu`, `ndarray-backend` | CUDA/wgpu **не** в default |
| CLI | `denote-train --config … --smoke --device auto` | device flag есть; smoke ещё не multi-step job |
| Legacy CRNN smoke | `train_loop.rs` | не primary |

Пробелы Phase 0:

- нет `RunLogger` / `runs/` layout  
- нет JSONL metrics + TensorBoard bridge  
- нет multi-step mini-train (100 steps) с checkpoint/resume  
- нет `scripts/train_1070.md`  
- нет зафиксированного GPU backend matrix (cuda vs wgpu vs cpu) на 1070  
- VRAM/OOM knobs не задокументированы  

---

## 2. Exit criteria (Definition of Done Phase 0)

Фаза **закрыта**, когда выполнено **всё**:

| ID | Критерий | Как проверить |
|----|----------|---------------|
| E0.1 | Workspace собирается `cargo test --workspace --lib` (default features) | local green |
| E0.2 | `cargo run -p denote-train --release -- --config configs/miros_v1_smoke.toml --smoke` OK | log `MIROS Smoke OK` |
| E0.3 | Зафиксирован **working backend** для 1070: `cuda` **или** `wgpu` **или** честный CPU + note | таблица в `scripts/train_1070.md` |
| E0.4 | **≥100 training steps** mini-loop (teacher-forcing CE) с finite loss | `runs/phase0_*/metrics.jsonl` ≥ 100 step rows |
| E0.5 | Checkpoint save + **`--resume`** продолжает step counter | 50 steps → stop → resume ещё 50 |
| E0.6 | JSONL metrics: `loss`, `lr`, `step_time_ms` (минимум) | schema §5 |
| E0.7 | TensorBoard: кривая `train/loss` (`tensorboard --logdir runs` или после `tools/jsonl_to_tb.py`) | reproducible commands |
| E0.8 | Документ `scripts/train_1070.md`: commands, OOM knobs, backend, troubleshooting | exists |

**Не блокирует Phase 0:** MusicFM weights, F1>0.6, multi-GPU.

**Soft success:** 100 steps на **CUDA 1070**.  
**Hard success (минимум):** 100 steps на **любом** backend + documented GPU path + JSONL+TB pipeline.

---

## 3. Scope / Non-scope

### In scope

1. Backend spike (Windows + GTX 1070 CC 6.1)  
2. Mini training job loop (100+ steps) на tiny MIROS-config  
3. `RunLogger`: console + JSONL  
4. TensorBoard path (P0: `tools/jsonl_to_tb.py`; optional native writer)  
5. Checkpoint + resume CLI  
6. Run directory layout `runs/{exp_id}/`  
7. Docs: `scripts/train_1070.md`  
8. Config `configs/phase0_mini.toml`  

### Out of scope (Phase 1+)

- MusicFM production load + golden latents  
- Latent cache Job L0  
- Presence / modular multi-decoder  
- FluidR3 dataset scale  
- Kaggle production notebooks (только note overflow)  

---

## 4. Work packages

### WP0.1 — Inventory & smoke baseline (0.5 day)

1. Зафиксировать: Rust, Burn 0.21, Windows, driver, `nvidia-smi` → `scripts/train_1070.md` §Environment  
2. Прогнать:
   ```powershell
   cargo test --workspace --lib
   cargo run -p denote-train --release -- --config configs/miros_v1_smoke.toml --smoke
   ```
3. Wall time smoke, backend used.

**Acceptance:** baseline commands documented.

---

### WP0.2 — GPU backend spike (1–2 days, critical path)

| Order | Feature | Command sketch | 1070 |
|-------|---------|----------------|------|
| 1 | `cuda` | `--features cuda --device cuda` | ideal; risk CC 6.1 |
| 2 | `wgpu` | `--features wgpu --device wgpu` | Vulkan/DX12 fallback |
| 3 | CPU `ndarray` | default | always |

**Deliverables:**

1. `resolve_device(arg)` for smoke + mini-loop  
2. Log: `backend=… device=…`  
3. Matrix in `train_1070.md`  
4. If CUDA fails: note Kaggle overflow (не блок hard success)

**Touchpoints:** `miros_smoke.rs`, `main.rs`, features in Cargo.toml.

---

### WP0.3 — Run layout + CLI flags (0.5 day)

```text
--config PATH
--smoke
--mini-steps N          # NEW
--run-dir PATH          # NEW default runs/<timestamp>_<name>
--resume PATH           # NEW
--device auto|cpu|cuda|wgpu
--log-every N
--ckpt-every N
```

```text
runs/phase0_mini_YYYYmmdd_HHMMSS/
  config.toml.copy
  hparams.json
  metrics.jsonl
  tb/
  checkpoints/
    step_000050.*
    last.*
    state.json          # step, optional optim note
```

---

### WP0.4 — RunLogger JSONL (0.5–1 day)

**Module:** `crates/denote-train/src/metrics_log.rs`

```rust
pub struct RunLogger { /* ... */ }
impl RunLogger {
    pub fn new(run_dir: &Path) -> Result<Self>;
    pub fn log_scalars(&mut self, step: u64, split: &str, kv: &[(&str, f32)]) -> Result<()>;
    pub fn flush(&mut self) -> Result<()>;
}
```

**JSONL line example:**

```json
{"ts":"2026-07-20T12:00:00Z","step":10,"split":"train","loss":6.12,"lr":0.0003,"step_time_ms":42.5}
```

**Deps:** `serde_json` only. No `tensorboard-rs` required.

---

### WP0.5 — Mini train loop 100 steps (1–2 days)

**Module:** `crates/denote-train/src/mini_train.rs`

```text
build tiny DenoteMiros from config
for step in start..start+mini_steps:
    synthetic batch (reuse smoke path)
    TF forward → CE → AdamW
    log; ckpt every ckpt_every
```

**Config:** `configs/phase0_mini.toml` (from `miros_v1_smoke`, tiny).  
**Data:** synthetic only — no downloads.

**OOM knobs:** batch_size=1, tiny width, mini_steps, device.

---

### WP0.6 — Checkpoint + resume (0.5–1 day)

```powershell
cargo run -p denote-train --release --features <backend> -- `
  --config configs/phase0_mini.toml --mini-steps 50 --run-dir runs/phase0_resume_test

cargo run -p denote-train --release --features <backend> -- `
  --config configs/phase0_mini.toml --mini-steps 50 `
  --resume runs/phase0_resume_test/checkpoints/last --run-dir runs/phase0_resume_test
```

If optim resume hard: model weights + `state.json` step only — document.

---

### WP0.7 — TensorBoard bridge (0.5 day)

**P0:** `tools/jsonl_to_tb.py`

```powershell
python tools/jsonl_to_tb.py --jsonl runs/phase0_.../metrics.jsonl --out runs/phase0_.../tb
pip install tensorboard
tensorboard --logdir runs --port 6006
```

**P1 stretch:** feature-gated `tensorboard-rs` only if builds on Windows.

---

### WP0.8 — Documentation (0.5 day)

`scripts/train_1070.md`:

1. Environment  
2. Backend matrix  
3. Quickstart (smoke, mini-100, resume, TB)  
4. OOM troubleshooting  
5. Kaggle overflow policy  
6. Exit checklist  

Minimal README links to PLAN + Phase 0.

---

## 5. File change map

| Path | Action |
|------|--------|
| `scripts/train_1070.md` | create |
| `tools/jsonl_to_tb.py` | create |
| `configs/phase0_mini.toml` | create |
| `crates/denote-train/src/metrics_log.rs` | create |
| `crates/denote-train/src/mini_train.rs` | create |
| `crates/denote-train/src/main.rs` | wire CLI |
| `crates/denote-train/src/miros_smoke.rs` | device param |
| `crates/denote-train/src/device.rs` | optional resolve helper |
| `runs/.gitkeep` + gitignore `runs/**` | create |
| `README.md` | link PLAN + phase0 |
| `docs/plans/phase0_platform_proof.md` | this file |
| `docs/plans/README.md` | index of phases |

---

## 6. Implementation order

```text
Day 1: WP0.1 → WP0.2 backend spike
Day 2: WP0.3 CLI → WP0.4 logger → WP0.5 mini on CPU
Day 3: WP0.5 GPU → WP0.6 resume → WP0.7 TB → WP0.8 docs
```

---

## 7. Test plan

| Test | Command |
|------|---------|
| lib tests | `cargo test --workspace --lib` |
| smoke | `--smoke` |
| mini CPU | `--mini-steps 100 --device cpu` |
| mini GPU | `--features cuda|wgpu --device …` |
| resume | WP0.6 |
| TB | `tensorboard --logdir runs` |

---

## 8. Risks

| Risk | Mitigation |
|------|------------|
| Burn cuda fails Win+1070 | wgpu → CPU; Kaggle later |
| Slow CUDA compile | develop logger on CPU |
| Checkpoint format | state.json + model only |
| Scope creep MusicFM | strict non-scope |

---

## 9. Handoff → Phase 1

После E0.* green Phase 1 использует: RunLogger, runs/, resume, backend matrix, mini-loop patterns → Job L0 cache + Job A.

---

## 10. Checklist

- [x] WP0.1 baseline  
- [x] WP0.2 backend matrix (`cpu` + `cuda` on GTX 1070)  
- [x] WP0.3 CLI + run dir  
- [x] WP0.4 JSONL  
- [x] WP0.5 mini ≥100 steps  
- [x] WP0.6 resume  
- [x] WP0.7 TensorBoard (`tools/jsonl_to_tb.py`)  
- [x] WP0.8 docs (`scripts/train_1070.md`)  
- [x] E0.1–E0.8  

---

*Phase 0 = platform proof only. Architecture/MIROS-beating starts Phase 1+.*
