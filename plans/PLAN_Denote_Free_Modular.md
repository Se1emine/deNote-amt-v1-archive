# План (технический): Denote Free Modular → scale to ~1B — GTX 1070 first, Kaggle overflow, observability

## North Star (ответ на «план всё ещё про это?»)

| Вопрос | Ответ |
|--------|--------|
| **1. Архитектурно лучше MIROS?** | **Да — это главная ось v1.** Не «просто больше весов», а presence-gate, modular multi-decoder, cross-stem, FluidR3 domain, FSM anti-leakage, aux heads. Ablation обязателен: arch @ small/medium budget vs MIROS-style baseline. |
| **2. В итоге ~1B параметров?** | **Да как цель endgame (Phase 5), не как первый train job.** v1 = modular free на 1070 (~150–350M system, trainable куски ≤60M). Потом progressive scale (шире/глубже декодеры, больше groups/experts, optional LoRA/partial enc FT, MoE-FFN) → **system ~0.9–1.1B**, обучая **по-прежнему кусками** (на 1070/Kaggle — только то, что влезает; полный 1B dense FT — только если появится >24GB, иначе 1B = сумма модулей + frozen backbone). |
| **3. Practically fully Rust?** | **Да.** Train + infer + tokens + model + eval primary = **Rust/Burn 0.21**. Python **только** offline tools: `convert_musicfm.py`, FluidR3 render, optional `jsonl_to_tb.py`. Не возвращаем PyTorch competition stack как primary. |

**Иерархия приоритетов (если конфликтуют):**

```text
1) Работает на 1070 + free overflow Kaggle
2) Arch advantages vs MIROS + честные ablations
3) Practically full Rust
4) Scale system → ~1B (после 1–3, modular path)
5) Multi Onset F1 > 0.5998 (целевая метрика, best-effort on free wall-clock)
```

---

## 0. Constraints (жёсткие)

| ID | Constraint |
|----|------------|
| C1 | **Primary compute = GTX 1070 8 GB (local Windows)** — всё, что физически влезает и сходится за разумное время |
| C2 | **Kaggle free** (T4/P100, ~30 h/week) — **только overflow**: jobs, которые на 1070 OOM / неприемлемо медленные / нужен Linux CUDA sanity |
| C3 | **No paid GPU / no university required** for v1–v2 modular; optional later only for dense FT large single module |
| C4 | **Rust + Burn 0.21** primary (`denote-train` already `burn` features `train`) — practically full Rust |
| C5 | Modular train: **one trainable submodule per job**; freeze rest; **scale-to-1B = more/wider modules, not one 1B job on 8GB** |
| C6 | Target metric: multi-instrument **Note Onset F1** (program+pitch+onset ±50 ms) > **MIROS 0.5998** (best-effort on free) |
| C7 | **Architecture > scale** for claiming “better than MIROS”: matched-budget ablation before “we have 1B” claim |

---

## 1. Compute policy: 1070 vs Kaggle

### 1.1 Decision matrix

| Workload | Where | Why |
|----------|-------|-----|
| `cargo test`, unit tests, FSM, token encode | **1070 / CPU local** | no GPU need |
| FluidR3 MIDI→wav render | **CPU local** | saves GPU quota |
| Token cache + **MusicFM latent cache** write | **1070** if Burn+CUDA OK; else CPU/ndarray slow; Kaggle if OOM on long tracks | one-time |
| Train Job A: adapter + presence (≤20M trainable) | **1070** | default |
| Train Job B..G: single group decoder (15–40M) | **1070** | default; on latent cache almost always fits |
| Full MusicFM forward **without** cache, long seq, large batch | **Kaggle** if 1070 OOM | overflow |
| Multi-day uninterrupted long run needing Linux | **Kaggle** optional | 1070 can still multi-day with resume |
| Final multi-checkpoint eval sweep | **1070** or Kaggle | whichever free |
| Dense end-to-end FT of encoder | **out of scope v1** | neither free path wants this |

### 1.2 1070 training envelope (targets)

```text
GPU:              GTX 1070, 8 GB GDDR5, CC 6.1
Trainable/job:    ≤ 40–60M params (AdamW fp32 states ≈ 12× weights → keep small)
Encoder:          FROZEN; prefer offline latent cache → encoder not in training graph
Audio chunk:      2.0–5.0 s @ 24 kHz
Batch size:       1–2 (latent-cached: 4–8 possible)
Seq tokens:       cap decoder teacher-force length (e.g. ≤ 512–1024)
Precision:        fp32 first (stable on 1070); try fp16 if backend supports without NaNs
Checkpoint:       every N steps + every epoch; atomic rename
Resume:           --resume path/to/ckpt
```

### 1.3 Latent cache (ключевой приём под 8 GB)

```text
Phase L0 (once per dataset split):
  wav_i → MusicFM_frozen → Z_i ∈ R^{T×D}  →  disk: data/cache/latents/{id}.safetensors
  notes_i → tokens → data/cache/tokens/{id}.bin

Phase train jobs:
  load Z_i, tokens_i only
  trainable: adapter/presence/decoder_*
  VRAM ≈ activations of small net + batch of latents  (MusicFM weights optional not loaded)
```

**Правило:** если job можно сделать на latent cache → **только 1070**.  
Kaggle для latent train — только если local disk/time bottleneck, не VRAM.

### 1.4 Kaggle overflow runbook (когда всё же Kaggle)

Использовать Kaggle **если**:

1. Phase 0: Burn CUDA на Windows 1070 не собирается / падает → Linux T4 sanity  
2. Нужен forward full MusicFM batch>1 без cache и 1070 OOM  
3. Нужен overnight > local practicality **и** есть 30h quota  

Kaggle job contract:

```text
- Upload: prebuilt data cache (latents+tokens) as Dataset, not raw wav if possible
- Binary: prefer precompiled musl/gnu linux release artifact OR cargo build in notebook once
- Always --resume from Kaggle output / Dataset version
- Mirror metrics JSONL back to local for TensorBoard
```

---

## 2. System architecture (technical)

### 2.0 Какое семейство: CNN / RNN / Transformer?

**Итог: гибрид, не «чистая CNN» и не «чистая RNN».**  
Ближайший класс в литературе: **Conformer-encoder + Transformer-decoder seq2seq (MT3 / YourMT3+ / MIROS lineage)**, плюс наши modular multi-decoder + presence.

| Блок | Тип слоёв | Роль | Аналог |
|------|-----------|------|--------|
| **Audio frontend** | **CNN-подобные** свёртки + mel/log-mel (как в MusicFM) | wav → time-frequency features | MusicFM frontend |
| **Encoder (MusicFM)** | **Conformer = CNN local + Transformer (self-attention) + часто conv modules** | SSL-латенты ~25 Hz, «понимание» звука | MusicFM / Conformer |
| **Adapter** | **лёгкий RNN (recurrent) и/или 1–2 Transformer/MLP блока** | привести Z к декодерам, conditioning | MIROS recurrent adapter |
| **Presence head** | **MLP / linear** (не deep CNN) | multi-label «какие инструменты» | classifier head |
| **Token decoders (×G)** | **Transformer decoder** (AR, cross-attn к encoder/adapter states) | MIDI-event tokens | MT3 / T5-style |
| **Aux onset/pitch (opt.)** | **linear / small CNN over time** на Z | multi-task стабилизация | OaF-style heads |
| **FSM / merge** | **не нейросеть** — правила на токенах | anti-leakage, budget | наш `denote-tokens` |

```text
                    ┌──────────────────────────────────────┐
  wav               │  ГИБРИД                              │
    │               │  CNN frontend                        │
    ▼               │       +                              │
  MusicFM encoder   │  Conformer (CNN + Transformer)       │  FROZEN pretrained
    │ Z             │       +                              │
    ▼               │  Adapter (RNN and/or small Transf.)  │  train Job A
  adapter+presence  │       +                              │
    │               │  Transformer decoders × G (AR)       │  train Job B_g
    ▼               │       +                              │
  tokens → FSM      │  rules (не NN)                       │
                    └──────────────────────────────────────┘
```

**Чем это НЕ является:**

| Архитектура | Почему не v1 |
|-------------|----------------|
| Чистая **CNN** (Basic Pitch / CRNN piano-roll) | Плохо масштабируется на multi-instrument seq; у нас legacy CRNN только для ablations |
| Чистая **RNN/LSTM** end-to-end | Слабее long-range; MT3/MIROS уже на Transformer decode |
| Чистый **decoder-only Transformer** как MuScriptor | Возможен later; v1 = encoder–decoder как MIROS, проще опереться на MusicFM |
| Классический **routed MoE LLM** | Не v1; optional Phase 5 inside decoder FFN |

**Одной фразой для диплома:**  
*Denote — гибридная seq2seq AMT-модель: Conformer/MusicFM encoder + (recurrent) adapter + набор Transformer-декодеров по группам инструментов, с presence-классификатором и rule-based FSM-декодированием.*

### 2.1 Modules

```text
DenoteModular {
  encoder: MusicFmEncoder,          // Conformer/CNN+Transformer, pretrained, freeze
  adapter: InstrumentAdapter,       // Linear/Conv + RNN and/or small Transformer
  presence: PresenceHead,           // MLP multi-label over K programs (K=8 challenge)
  decoders: [TokenDecoder; G],      // Transformer AR decoders, independent params
  // optional later: lora_encoder; MoE-FFN inside decoder (Phase 5)
}
```

**Not v1:** classic routed MoE (Switch-style).  
**Yes v1:** multi-decoder modular Transformer experts, **independent optimization**, merge at infer.

### 2.2 Shapes (MusicFM-compat)

```text
x_wav:     [B, N]                 N = chunk_sec * 24000
mel/feat:  frontend inside encoder
Z:         [B, T, D]              T ≈ 25 * chunk_sec, D = model_dim (e.g. 512/768 — match MusicFM)
presence:  [B, K]                 K = 8 (or #groups)
tokens_g:  [B, L_g]               teacher-forced per group g
```

### 2.3 Losses

| Head | Loss | Weight (start) |
|------|------|----------------|
| Token CE (active groups only) | cross-entropy w/ ignore pad | 1.0 |
| Presence | BCE with logits (multi-label) | 0.2–0.5 |
| Aux onset/pitch (optional) | BCE / CE on pooled Z | 0.1–0.3 |

### 2.4 Infer merge

```text
Z = encoder(wav) | load cache
p = σ(presence(adapter(Z)))
for g in groups:
  if p[programs_in_g] > τ OR force_all:
    toks_g = AR_decode(decoder_g, adapter(Z), fsm_g)
notes = detokenize_merge(toks_*) + global FSM (program budget, no double notes)
```

### 2.5 Parameter budgets (v1)

| Module | Trainable | Job |
|--------|-----------|-----|
| MusicFM | 0 (frozen) | L0 cache only |
| adapter + presence + aux | ~8–25M | **Job A** @ 1070 |
| decoder_g | ~15–40M each | **Job B_g** @ 1070 sequential |
| System total | ~150–350M with frozen enc | infer |

### 2.6 Instrument groups (challenge)

```text
G0 keys:      Piano
G1 strings:   Violin, Viola, Cello
G2 wood:      Flute, Oboe, Bassoon
G3 brass:     Trombone
```

Config-driven; can split to G=8 later without redesign.

---

## 3. Repo layout (delta)

```text
configs/
  smoke.toml
  free_1070_cache.toml       # L0 latent export
  free_1070_jobA.toml        # adapter+presence
  free_1070_jobB_{g}.toml    # per decoder
  kaggle_overflow_*.toml     # larger batch / full forward only if needed

crates/denote-train/
  src/
    metrics_log.rs           # JSONL + pluggable TB
    jobs/
      cache_latents.rs
      train_shared.rs
      train_decoder.rs

crates/denote-model/
  modular.rs                 # DenoteModular, freeze flags
  presence.rs
  multi_decoder.rs

runs/
  {exp_id}/
    config.toml.copy
    metrics.jsonl
    tb/                      # TensorBoard event files
    checkpoints/
    hparams.json

scripts/
  train_1070.md
  kaggle_overflow.md
  tensorboard_local.md
```

---

## 4. Observability & training control (TensorBoard и лучше)

### 4.1 Evaluation of `tensorboard-rs`

| | |
|--|--|
| Crate | [`tensorboard-rs`](https://crates.io/crates/tensorboard-rs) |
| Last meaningful publish | **~2022** (v0.5.x) — stale |
| Features | scalar, image, histogram event write |
| Risk | unmaintained protobuf/API drift; bitrot; not Burn-integrated |

**Вердикт:** можно как optional backend, **не** как единственная зависимость. Не «best stable default» в 2026.

### 4.2 Recommended stack (3 layers)

```text
┌─────────────────────────────────────────────────────────┐
│ L1  Burn-native (primary, most stable for Burn 0.21)   │
│     burn::train metrics + CLI dashboard / Learner logs │
│     LossMetric, AccuracyMetric (where applicable),     │
│     custom Metric for presence F1 / token ppl          │
├─────────────────────────────────────────────────────────┤
│ L2  JSONL metrics (always on, source of truth on disk) │
│     runs/{exp}/metrics.jsonl                           │
│     {step, split, loss_ce, loss_pres, lr, gnorm, ...}  │
│     portable: plot in TB/Python/Excel; sync Kaggle→PC  │
├─────────────────────────────────────────────────────────┤
│ L3  TensorBoard UI (local visualization)               │
│     Prefer: thin EventFileWriter (scalars) OR          │
│     generate TB events from JSONL via small tool       │
│     Viewer: `tensorboard --logdir runs` (Python pip)   │
│     Optional crate: evaluate `tboard` / hand-rolled    │
│     scalar Summary protocol; fallback tensorboard-rs   │
│     behind feature flag `tb-rs` only if tests pass     │
└─────────────────────────────────────────────────────────┘
```

### 4.3 Why not W&B / Neptune as primary

| Tool | Free? | Offline 1070? | Rust-native? |
|------|-------|---------------|--------------|
| **Burn metrics + JSONL** | yes | yes | yes |
| **TensorBoard local** | yes | yes (logdir local) | via event writer |
| W&B free | yes (cloud) | needs net; account | no first-class Burn |
| MLflow | yes self-host | heavy | overkill v1 |

**Primary product decision:**  
`JSONL + Burn CLI + local TensorBoard` — offline, free, diploma-reproducible.

### 4.4 Metrics to log (every N steps + epoch)

**Scalars**

- `train/loss_ce`, `train/loss_presence`, `train/loss_total`
- `train/lr`, `train/grad_norm`, `train/tokens_per_sec`, `train/step_time_ms`
- `train/vram_mb` (if NVML available; else skip)
- `val/loss_ce`, `val/presence_f1` (micro/macro)
- `val/multi_onset_f1` (epoch-level, expensive — rarer)
- `data/n_instruments_mean`, `data/chunk_sec`

**Text/hparams (once)**

- full resolved config TOML hash, git commit, GPU name, backend

**Optional later**

- histogram of token logits / presence probs (TB hist) — only if event writer stable  
- audio/MIDI preview — export files to `runs/.../samples/`, not necessarily TB

### 4.5 Implementation sketch (`denote-train`)

```rust
// metrics_log.rs — conceptual
pub struct RunLogger {
    jsonl: std::fs::File,
    tb: Option<Box<dyn TbWriter>>, // trait: add_scalar(tag, value, step)
}

pub trait TbWriter: Send {
    fn add_scalar(&mut self, tag: &str, value: f32, step: i64);
    fn flush(&mut self);
}

// feature = "tensorboard"
// try tboard or minimal event writer; feature = "tensorboard-rs" legacy

impl RunLogger {
    pub fn log(&mut self, step: u64, kv: impl IntoIterator<Item = (&str, f32)>) { ... }
}
```

CLI:

```powershell
# train
cargo run -p denote-train --release --features cuda -- `
  --config configs/free_1070_jobA.toml --run-dir runs/jobA_001

# visualize (Python env once)
pip install tensorboard
tensorboard --logdir runs --port 6006
```

If TB writer crate fails CI: **JSONL still works**; script `tools/jsonl_to_tb.py` (optional, ~30 lines) converts scalars → events using TensorFlow/tensorboard Python API — **stable**, offline after train.

**Рекомендация плана:**

| Priority | Mechanism |
|----------|-----------|
| P0 | JSONL + console/Burn metrics |
| P1 | `tools/jsonl_to_tb.py` **or** native scalar event writer |
| P2 | feature-gated `tensorboard-rs` only if still builds; not required |
| P3 | Burn Learner dashboard when loop migrates fully to `burn::train::Learner` |

---

## 5. Training jobs (technical schedule)

### Job L0 — Latent + token cache (1070 preferred)

```text
Input:  wav list + MIDI/notes
Output: latents/*.safetensors, tokens/*.bin, manifest.jsonl
Trainable: none
VRAM: encoder forward only, B=1, chunked
Fail → Kaggle overflow for L0 only
```

### Job A — Shared adapter + presence (1070)

```text
Trainable: adapter, presence, aux
Frozen: encoder (not loaded if cache)
Data: latent cache, multi-label presence from note programs
Steps: until val presence F1 plateaus
Ckpt: runs/jobA/checkpoints/last.mpk (or safetensors per module)
```

### Job B_g — Decoder group g (1070, sequential)

```text
Load: Job A weights (freeze adapter optional: train lightly with small LR or freeze)
Trainable: decoder_g only
Loss: CE on tokens of programs in group g (mask others)
Repeat g = 0..G-1
```

### Job C — Optional joint polish (1070 if VRAM ok)

```text
Unfreeze adapter (small LR) + all decoders with low LR, short run
If OOM → skip; use infer-only merge of Job A+B
If still want → Kaggle overflow short joint
```

### Job E — Eval

```text
Load modular ckpts → multi_onset_f1 on challenge-like val
Log to metrics + JSON summary for thesis tables
```

---

## 6. Data pipeline (technical)

```text
MIDI (Lakh/classical) 
  → tools/render_fluidr3.py (CPU, SoundFont FluidR3_GM)
  → wav 24 kHz mono
  → Job L0 MusicFM latents + denote-tokens sequences
  → optional Slakh stem mix: sum random stems → re-encode notes multi-program

Augment (on-the-fly in train, cheap):
  - time crop already chunked
  - gain / mild noise on wav only if not using latent cache
  - for latent cache: mix latents only if same-time aligned stems precomputed
```

Manifest schema:

```json
{"id":"...", "latent":"...", "tokens":"...", "programs":[0,1], "n_instruments":2, "split":"train"}
```

---

## 7. Phased delivery

### Phase 0 — Platform proof (1070)

- [ ] Burn CUDA (or working GPU backend) on GTX 1070  
- [ ] Smoke train existing miros path + VRAM log  
- [ ] `RunLogger` JSONL + console  
- [ ] TB path: jsonl_to_tb **or** scalar writer spike  
- [ ] Document OOM knobs in `scripts/train_1070.md`

**Exit:** 100 steps Job-like loop on 1070 without OOM; TensorBoard shows `loss` curve from a run.

### Phase 1 — Cache + Job A

- [ ] MusicFM weight load / convert  
- [ ] L0 latent cache on 1070  
- [ ] Job A config + train + TB metrics  
- [ ] presence val metrics  

### Phase 2 — Modular decoders B_g

- [ ] multi_decoder API freeze flags  
- [ ] sequential Job B on 1070  
- [ ] infer merge + FSM  
- [ ] ablation single vs multi decoder  

### Phase 3 — Data scale + F1 vs MIROS

- [ ] more FluidR3 + stem  
- [ ] curriculum  
- [ ] multi_onset_f1  
- [ ] Kaggle **only** if L0/joint overflow  

### Phase 4 — Product

- [ ] `denote-infer` loads module dir  
- [ ] GuitarSet light Job B-style finetune on 1070  

### Phase 5 — Scale system → ~1B (endgame, after modular works)

**Не ломает C1:** 1B ≠ один AdamW job на 1070.

| Step | What | Params effect | Compute |
|------|------|---------------|---------|
| 5.1 | Widen/deepen each `decoder_g` (still one job at a time) | +per module | 1070 if ≤60M trainable; else Kaggle |
| 5.2 | G=8 per-instrument decoders (more modules) | +system capacity | sequential 1070 |
| 5.3 | Optional MoE-FFN **inside** a decoder (train one decoder job) | +capacity, sparse | 1070/Kaggle by size |
| 5.4 | LoRA / last-N encoder blocks FT (small trainable) | +quality | 1070 latent or Kaggle |
| 5.5 | Count **system params** (frozen MusicFM + all modules) → target **~1B** | reporting | infer-only assemble |
| 5.6 | If ever 24GB+ GPU: optional short joint polish | quality | overflow / future |

**Honest reporting for thesis:**

- `N_total` = sum of all loaded modules + encoder  
- `N_trainable_per_job` = what 1070 actually trained  
- Ablation: modular-arch @ ~MIROS budget vs scaled ~1B system  

---

## 8. Config sketch (`free_1070_jobA.toml`)

```toml
[run]
name = "jobA_presence"
run_dir = "runs/jobA_presence"
device = "cuda:0"          # 1070
backend = "cuda"           # fallback wgpu documented

[data]
latent_manifest = "data/cache/latents/manifest.jsonl"
batch_size = 2
num_workers = 2
chunk_sec = 4.0

[model]
freeze_encoder = true
use_latent_cache = true
d_model = 1024
presence_k = 8
adapter_layers = 2

[optim]
lr = 3e-4
weight_decay = 0.01
grad_clip = 1.0
max_steps = 50000
ckpt_every = 500

[log]
jsonl = true
tensorboard = true          # writes tb/ or defers to jsonl_to_tb
log_every = 10
val_every = 500
```

---

## 9. Risks (technical)

| Risk | Mitigation |
|------|------------|
| Burn CUDA + Win + 1070 (CC 6.1) | Phase 0 spike; wgpu/libtorch fallback; Kaggle Linux for proof |
| tensorboard-rs bitrot | JSONL P0; Python jsonl→tb; feature-gate rust TB |
| 8 GB OOM | latent cache; B=1; short chunk; freeze encoder unloaded |
| Kaggle 30h waste on compile | cache compile or upload binary; prefer data-heavy local |
| Modular underperforms single decoder | Job C polish; ablation; keep single-decoder baseline config |

---

## 10. Definition of done

### v1 (must ship on free/1070)

- [ ] All mandatory train jobs runnable on **GTX 1070** with resume  
- [ ] Kaggle used **only** for documented overflow cases  
- [ ] Modular A + B_g checkpoints → single infer  
- [ ] Arch edges vs MIROS implemented + ablations  
- [ ] **JSONL + TensorBoard** curves for loss/presence/F1  
- [ ] Multi-onset F1 reported; target > 0.5998 best-effort  
- [ ] **Practically full Rust/Burn** (Python = tools only)  

### endgame (Phase 5)

- [ ] Documented path + config to **system ~1B** via modular scale (not dense 1B on 8GB)  
- [ ] Thesis-ready param accounting: total vs trainable-per-job  

---

## 11. Immediate next steps (after approve)

1. Phase 0: 1070 backend + VRAM smoke  
2. `RunLogger` (JSONL) + TB bridge (jsonl_to_tb vs feature-gated crate)  
3. Latent cache Job L0  
4. Job A / Job B modular (arch > MIROS) — scale-to-1B only after this works  

---

*North Star: arch > MIROS · practically full Rust · endgame ~1B modular scale · Compute: 1070 primary, Kaggle overflow · Monitor: Burn + JSONL + local TensorBoard*

