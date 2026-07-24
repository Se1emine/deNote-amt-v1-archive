# Phase 1 — Cache + Job A

**Master-план:** [`PLAN_Denote_Free_Modular.md`](PLAN_Denote_Free_Modular.md)  
(ссылки: §1.3 Latent cache · §2 Architecture · §5 Job L0 / Job A · §6 Data · §7 Phase 1 · §8 `free_1070_jobA.toml`)

**Зависимость:** Phase 0 закрыта по exit criteria ([`PLAN_Phase0_Platform_Proof.md`](PLAN_Phase0_Platform_Proof.md)): working backend, `RunLogger`/JSONL, mini-train, resume, TB path.

**Цель фазы:**  
1) **Job L0** — построить **token cache + latent cache** (MusicFM encoder frozen forward → диск);  
2) **Job A** — обучить **shared adapter + presence (+ aux)** на latents **без** encoder в training graph;  
3) залогировать presence metrics + CE (если decoder в loop) в `runs/` + TensorBoard.

**Не цель фазы:** multi-decoder Job B_g, full multi-onset F1 vs MIROS leaderboard, scale 1B, FluidR3 full corpus (можно mini-subset), joint polish Job C.

---

## 1. Контекст: что требует master-план

### 1.1 Job L0 (master §5)

```text
Input:  wav list + MIDI/notes
Output: latents/*.safetensors (or .bin), tokens/*.bin, manifest.jsonl
Trainable: none
VRAM: encoder forward only, B=1, chunked
Fail → Kaggle overflow for L0 only
```

### 1.2 Job A (master §5)

```text
Trainable: adapter, presence, aux
Frozen: encoder (not loaded if cache)
Data: latent cache, multi-label presence from note programs
Steps: until val presence F1 plateaus
Ckpt: modular adapter/presence weights
```

### 1.3 Latent cache (master §1.3) — ключ для GTX 1070 8 GB

```text
L0 once:  wav → MusicFM_frozen → Z[T,D] → disk
Job A:    load Z + labels only  (encoder weights optional not in VRAM)
```

### 1.4 Manifest (master §6)

```json
{"id":"...","latent":"...","tokens":"...","programs":[0,1],"n_instruments":2,"split":"train"}
```

### 1.5 Compute policy (master §1)

| Workload | Где |
|----------|-----|
| FluidR3 / preprocess | **CPU local** |
| L0 latent export | **1070** preferred; Kaggle если OOM/too slow |
| Job A train on cache | **1070 only** (must fit) |

---

## 2. Baseline в репозитории (на старте Phase 1)

| Компонент | Статус | Где |
|-----------|--------|-----|
| `MusicFmEncoder` | scaffold Burn, smoke sizes | `denote-encoder` |
| `convert_musicfm.py` | PT → numpy dumps | `tools/convert_musicfm.py` |
| Bit-exact MusicFM load | **нет / incomplete** | нужен WP1.2 |
| `InstrumentAdapter` + aux pitch/onset | есть (LSTM + heads) | `denote-model/adapter.rs` |
| `PresenceHead` multi-label | **нет** | нужен WP1.4 |
| `DenoteMiros` | encoder+adapter+single decoder | `miros.rs` |
| Forward from **latents** (skip encoder) | **нет** | нужен `forward_from_latents` |
| `TokenFeatureCache` | есть | `denote-data/token_cache.rs` |
| Latent cache type | **нет** | нужен `latent_cache.rs` |
| Phase 0 RunLogger / resume | **ожидается done** | `denote-train` |
| FluidR3 render tool | **нет** | mini: synth/sine + optional tiny MIDI set |

---

## 3. Exit criteria (Definition of Done Phase 1)

| ID | Критерий | Проверка |
|----|----------|----------|
| E1.0 | Phase 0 E0.* green (или explicit waiver + CPU path only) | checklist Phase 0 |
| E1.1 | MusicFM weights: convert script + Burn load **shape-compatible** (golden optional soft) | load without panic; log param count |
| E1.2 | Job L0: ≥ **N_min** samples cached (tokens + latents + manifest) | `data/cache/.../manifest.jsonl` |
| E1.3 | Train Job A **without loading encoder** (latent-only path) | code path + VRAM note in docs |
| E1.4 | Presence head: multi-label BCE; val **presence F1** (macro or micro) logged | `metrics.jsonl` keys |
| E1.5 | Adapter (+ aux optional) trained; finite losses; resume works | `runs/jobA_*/` |
| E1.6 | Configs: `free_1070_cache.toml`, `free_1070_jobA.toml` | files exist |
| E1.7 | Docs: `scripts/job_L0_cache.md`, section Job A in `scripts/train_1070.md` or `scripts/job_A.md` | commands reproducible |
| E1.8 | Checkpoint layout modular: `adapter.*`, `presence.*` (decoder optional/frozen) | loadable independently |

**N_min (hard):** ≥ **32** train + ≥ **8** val synthetic/real clips (smoke scale OK).  
**N_min (soft / better):** ≥ 500–2000 clips if disk/time allow (still free/1070).

**Soft success:** presence val F1 ≫ random (e.g. > 0.3 micro on multi-label with K=8) on held-out cache.  
**Hard success:** E1.1–E1.8 without requiring F1 threshold (data may be tiny).

**Не блокирует Phase 1:** multi-onset note F1, Job B, F1 > 0.5998, full MusicFM bit-exact MSE=0 (shape+stable train enough; golden latent MSE — stretch).

---

## 4. Scope / Non-scope

### In scope

1. Offline MusicFM convert + Burn weight load (best-effort mapping)  
2. Encoder freeze + **latent export** CLI/job  
3. Token cache build (extend existing) aligned with same `id`s  
4. `manifest.jsonl` reader/writer  
5. `PresenceHead` + loss BCE  
6. Job A train loop on latent cache (1070)  
7. Optional aux pitch/onset loss if frame labels cheap from notes  
8. Modular checkpoint save for adapter/presence  
9. Metrics: `loss_presence`, `loss_aux_*`, `presence_f1`, `lr`, `step_time_ms`  
10. Mini real-or-synth dataset path documented  

### Out of scope (Phase 2+)

- Multiple `TokenDecoder` groups / Job B_g  
- Infer merge + FSM multi-program product path (beyond smoke)  
- Cross-stem Slakh mix at scale  
- Challenge multi-onset F1 leaderboard  
- Dense encoder FT / LoRA (optional stretch only if time)  
- Classic MoE  

---

## 5. Architecture delta (Phase 1 only)

### 5.1 Modules after Phase 1

```text
DenoteJobA {
  // encoder: used ONLY in L0 export (or not loaded in Job A)
  adapter: InstrumentAdapter,     // existing LSTM adapter
  presence: PresenceHead,         // NEW: Linear/MLP → [B, K] logits
  // decoder: optional in Job A
  //   - Mode A1 (recommended): train presence+adapter (+aux); decoder frozen or absent
  //   - Mode A2 (optional): also TF token CE on single decoder (pretrain for Phase 2)
}
```

Master architecture keeps Transformer multi-decoders for Phase 2; Phase 1 **introduces presence** and **proves latent-cache training**.

### 5.2 Presence definition (challenge-aligned)

**K = 8** programs (master §2.6 / challenge GM set):

| idx | Program |
|-----|---------|
| 0 | Piano |
| 1 | Violin |
| 2 | Viola |
| 3 | Cello |
| 4 | Flute |
| 5 | Bassoon |
| 6 | Trombone |
| 7 | Oboe |

Label: multi-hot `y[k]=1` if any note with that program in the **chunk**.  
Head: `presence_logits = MLP(pool(adapter.memory))` or `MLP(mean(Z))` then after adapter — **prefer after adapter** so Job A trains adapter for presence.

Pooling: mean over time (start simple).

### 5.3 Forward paths

```text
L0:
  mel = audio_to_mel(wav)
  Z = encoder.forward_mel(mel)   // no grad
  save Z

Job A:
  Z = load_latent(id)            // [T,D] or [1,T,D]
  out = adapter.forward(Z, program_ids_dummy_or_zero)
  // For presence multi-label, program conditioning during train:
  // use program_ids = 0 or learned "null", OR sum of active program embs (better)
  presence_logits = presence.forward(out.memory)
  loss_p = BCE(presence_logits, y_multi)
  loss_aux = ... optional from note frames
  // Mode A2: token CE with single decoder
```

**Program emb for multi-instrument chunks:**  
sum/mean embeddings of active programs broadcast on T (document choice; default **sum of active program embeddings**).

### 5.4 Losses (master §2.3, Job A)

| Loss | Weight start | Required? |
|------|--------------|-----------|
| `loss_presence` BCE | 1.0 | **yes** |
| `loss_aux_onset` / `loss_aux_pitch` | 0.1–0.3 | optional if labels ready |
| `loss_ce` token | 0.5–1.0 | Mode A2 only |

```text
loss_total = w_p * loss_presence + w_aux * loss_aux [+ w_ce * loss_ce]
```

---

## 6. Data: minimal vs better

### 6.1 Hard path (always — closes E1.2)

**Synthetic mini-set** (no external download):

1. Generate N short wavs (sine / noise / multi-tone) **or** reuse render from MIDI with simple synth  
2. Notes JSON with `program` fields for K labels  
3. Encode tokens via `denote-tokens`  
4. L0: mel → encoder (random **or** loaded MusicFM) → latents  

Even with **random** encoder weights, Job A pipeline is valid engineering; presence learns on those latents.  
**Quality** needs pretrained MusicFM (WP1.2).

### 6.2 Soft path (preferred quality)

1. Small public MIDI classical subset  
2. `tools/render_fluidr3.py` (CPU) + FluidR3_GM.sf2  
3. Same L0/Job A  

If FluidR3 not ready: defer full render to Phase 3; Phase 1 soft path can be **subset of any wav+midi you already have** (GuitarSet single-instrument still trains presence with sparse labels).

### 6.3 Cache directory layout

```text
data/cache/
  phase1_mini/   # or free_1070/
    manifest.jsonl
    tokens/{id}.tok.bin          # existing TokenFeatureCache format OK
    latents/{id}.lat.safetensors # or .lat.bin + header
    meta/{id}.json               # optional per-sample programs, duration
```

**Latent file schema (proposed):**

```text
header: magic, version, T, D, dtype=f32
body: T*D floats row-major
# or safetensors: {"z": [T,D]}
```

**Manifest line:**

```json
{
  "id": "clip_0001",
  "latent": "latents/clip_0001.lat.safetensors",
  "tokens": "tokens/clip_0001.tok.bin",
  "programs": [0, 2],
  "n_instruments": 2,
  "split": "train",
  "duration_sec": 4.0,
  "sample_rate": 24000,
  "d_model": 512,
  "n_frames": 100
}
```

---

## 7. Work packages

### WP1.0 — Gate: Phase 0 ready (0.25 day)

- [ ] Confirm Phase 0 checklist  
- [ ] Reuse `RunLogger`, `--run-dir`, `--resume`, device resolve  
- [ ] If Phase 0 incomplete: finish E0 hard path before Job A long runs  

**Acceptance:** mini-train still runs.

---

### WP1.1 — Latent cache crate API (1 day)

**Files:**

- `crates/denote-data/src/latent_cache.rs`  
- export from `lib.rs`  
- tests: write/read roundtrip  

```rust
pub struct LatentMeta { pub id: String, pub n_frames: usize, pub d_model: usize, ... }
pub struct LatentSample { pub meta: LatentMeta, pub z: Vec<f32> } // [T*D]
pub struct LatentFeatureCache { pub root: PathBuf }
impl LatentFeatureCache {
    pub fn save(&self, sample: &LatentSample) -> Result<()>;
    pub fn load(&self, id: &str) -> Result<LatentSample>;
}
pub struct CacheManifest { /* read/write jsonl */ }
```

**Acceptance:** unit test roundtrip; manifest append/list by split.

---

### WP1.2 — MusicFM weights: convert + Burn load (1–3 days)

**Steps:**

1. Document download of official MusicFM `pretrained_msd.pt` + stats (link in script README)  
2. Run `tools/convert_musicfm.py` → `assets/musicfm_msd/`  
3. Implement `denote-encoder` loader: numpy/safetensors → module params  
4. Mapping table `name_pt → burn param path`; log unmatched keys  
5. **Soft golden:** optional compare one mel clip latent L2 vs Python (stretch)  
6. **Hard:** load + `forward_mel` runs; shapes `[B,T',D]` consistent  

**Config dims:** align `MusicFmEncoderConfig` with real MusicFM (not only smoke).  
If full Conformer parity incomplete:  

- **Fallback A:** load what matches; random rest; still freeze  
- **Fallback B:** use smoke encoder for pipeline, mark E1.1 partial  

Document honestly in `scripts/job_L0_cache.md`.

**Acceptance:** E1.1 hard or documented partial with pipeline unblocked.

---

### WP1.3 — Job L0 CLI: export latents + tokens (1–2 days)

**Binary surface (extend `denote-train` or `denote-data` bin):**

```powershell
cargo run -p denote-train --release --features <gpu> -- `
  --config configs/free_1070_cache.toml `
  --job cache-l0 `
  --manifest-out data/cache/phase1_mini/manifest.jsonl
```

**Algorithm:**

```text
for each item in input list:
  load wav → resample 24 kHz → mel (MusicFM frontend path)
  Z = encoder.forward_mel(mel)  // no_grad
  save latent
  notes → token_ids → TokenFeatureCache.save
  programs = unique programs in notes
  append manifest (split train/val by hash or list)
```

**OOM knobs:** chunk_sec 2–5, B=1, process long files in sliding windows (each window = sample id).

**Acceptance:** E1.2; encoder not trained; disk files loadable.

---

### WP1.4 — PresenceHead + losses (0.5–1 day)

**Files:**

- `crates/denote-model/src/presence.rs`  
- wire in `lib.rs`  
- `DenoteJobA` or extend `DenoteMiros` with optional presence  

```rust
pub struct PresenceHeadConfig { d_model, n_programs: K, hidden }
pub struct PresenceHead<B> { pool → linear → logits [B,K] }
```

**Metrics:**

- presence F1 micro/macro (threshold 0.5 on sigmoid)  
- per-program AP optional later  

**Acceptance:** unit test forward shapes; BCE decreases on overfit 4-batch.

---

### WP1.5 — `forward_from_latents` + freeze API (0.5 day)

```rust
impl DenoteMiros/JobA {
  pub fn forward_from_latents(&self, z, tokens_in?, program_ids) -> ...
}
```

- Job A optimizer **only** on adapter + presence (+ aux + optional decoder)  
- Encoder module **not constructed** or `require_grad(false)` and not in optim  

**Acceptance:** E1.3; code review / log “trainable params = …”.

---

### WP1.6 — Job A training loop (1–2 days)

**Config:** `configs/free_1070_jobA.toml` (master §8 as base)

```toml
[run]
name = "jobA_presence"
run_dir = "runs/jobA_presence"

[data]
latent_manifest = "data/cache/phase1_mini/manifest.jsonl"
batch_size = 2
chunk_sec = 4.0

[model]
use_latent_cache = true
freeze_encoder = true
d_model = 512          # match encoder
presence_k = 8
# mode = "presence_adapter" | "presence_adapter_decoder"

[optim]
lr = 3e-4
weight_decay = 0.01
max_steps = 20000
ckpt_every = 500

[log]
jsonl = true
log_every = 10
val_every = 200
```

**CLI:**

```powershell
cargo run -p denote-train --release --features <gpu> -- `
  --config configs/free_1070_jobA.toml `
  --job job-a `
  --run-dir runs/jobA_001
```

**Loop:** Phase 0 mini-train pattern + DataLoader over manifest.

**Val:** presence F1 each `val_every`; early stop optional.

**Acceptance:** E1.4, E1.5; TB curves for `train/loss_presence`, `val/presence_f1`.

---

### WP1.7 — Modular checkpoints (0.5 day)

```text
runs/jobA_001/checkpoints/
  last/
    adapter.mpk | adapter.safetensors
    presence.mpk
    state.json          # step, config hash, d_model, K
  step_000500/
```

Phase 2 will load these and train decoders with frozen adapter (or low LR).

**Acceptance:** E1.8; reload adapter+presence → same presence logits on fixed Z (atol loose).

---

### WP1.8 — Dataset builders (1 day)

1. `tools/build_phase1_mini.py` or Rust bin: synth clips + notes with multi-hot programs  
2. Optional: `tools/render_fluidr3.py` stub + docs (full data Phase 3)  
3. Split train/val 80/20  

**Acceptance:** one command produces input list for L0.

---

### WP1.9 — Documentation (0.5 day)

- `scripts/job_L0_cache.md` — L0 commands, VRAM, layout, Kaggle overflow  
- `scripts/job_A.md` — Job A train, metrics, resume  
- Cross-link from Phase 0 `train_1070.md`  

**Do not modify** `PLAN_Denote_Free_Modular.md` body (master immutable per project rule).

---

## 8. File change map

| Path | Action |
|------|--------|
| `PLAN_Phase1_Cache_JobA.md` | this plan (root) |
| `configs/free_1070_cache.toml` | create |
| `configs/free_1070_jobA.toml` | create |
| `crates/denote-data/src/latent_cache.rs` | create |
| `crates/denote-data/src/manifest.rs` | create (or inside latent_cache) |
| `crates/denote-model/src/presence.rs` | create |
| `crates/denote-model/src/job_a.rs` or extend `miros.rs` | create/extend |
| `crates/denote-encoder/src/load_weights.rs` | create |
| `crates/denote-train/src/jobs/cache_l0.rs` | create |
| `crates/denote-train/src/jobs/job_a.rs` | create |
| `crates/denote-train/src/main.rs` | `--job` dispatch |
| `tools/convert_musicfm.py` | extend if needed |
| `tools/build_phase1_mini.py` | create |
| `tools/render_fluidr3.py` | optional stub |
| `scripts/job_L0_cache.md` | create |
| `scripts/job_A.md` | create |
| `assets/musicfm_msd/` | generated (gitignore weights if large) |
| `data/cache/phase1_mini/` | generated |

---

## 9. Implementation order

```text
1. WP1.0 gate Phase 0
2. WP1.1 latent cache + manifest API + tests
3. WP1.4 presence head + unit tests
4. WP1.5 forward_from_latents + trainable filter
5. WP1.8 mini dataset builder (synth)
6. WP1.2 MusicFM load (parallel with 5 if possible)
7. WP1.3 Job L0 export (random enc → then pretrained)
8. WP1.6 Job A train loop + metrics
9. WP1.7 modular ckpt
10. WP1.9 docs + E1.* checklist
```

**Critical path:** WP1.1 → WP1.5 → WP1.3 → WP1.6.  
MusicFM golden (WP1.2 soft) can lag if shape load works.

---

## 10. 1070 envelope (Job A)

| Knob | Start value |
|------|-------------|
| batch_size | 2 (try 4–8 on cache) |
| d_model | match encoder (e.g. 512) |
| adapter | existing LSTM size |
| presence | small MLP |
| max T frames | from chunk_sec≈4 → T≈100 @ 25 Hz |
| precision | fp32 first |
| trainable params | target ≪ 60M (adapter+presence typically mid/low M) |

If OOM: reduce batch, reduce T, ensure encoder **not loaded**.

---

## 11. Metrics & TensorBoard (Job A)

Log every `log_every` / val:

- `train/loss_presence`, `train/loss_aux`, `train/loss_total`  
- `train/loss_ce` (if Mode A2)  
- `val/presence_f1_micro`, `val/presence_f1_macro`  
- `val/loss_presence`  
- `train/lr`, `train/step_time_ms`, `train/samples_per_sec`  

Reuse Phase 0 JSONL + `tools/jsonl_to_tb.py`.

---

## 12. Test plan

| Test | How |
|------|-----|
| Latent roundtrip | `cargo test -p denote-data latent` |
| Presence shapes | `cargo test -p denote-model presence` |
| L0 export smoke | 4 clips → 4 latents + manifest |
| Job A overfit | 4 train samples, 200 steps, loss↓, F1↑ |
| Resume Job A | 100 steps + resume |
| No-encoder VRAM | log trainable modules list |
| Weight load | load MusicFM → one forward |

---

## 13. Risks

| Risk | Mitigation |
|------|------------|
| MusicFM Burn mapping incomplete | Fallback partial load / smoke encoder; don't block cache API |
| 1070 OOM on L0 full encoder | B=1, short chunks; Kaggle overflow L0 only |
| Presence labels wrong program map | single `programs.toml` shared tokens↔ids |
| Adapter program emb multi-hot unclear | document sum-emb; ablation mean vs sum |
| Phase 0 not done | hard gate WP1.0 |
| Bit-exact latent mismatch | train still OK; quality later; golden stretch |

---

## 14. Handoff → Phase 2

Phase 2 expects:

1. Frozen **adapter + presence** checkpoints from Job A  
2. Latent (+ token) **manifest** convention  
3. Presence gate API for infer (`p_k > τ`)  
4. Single-decoder optional warmup (Mode A2) as baseline before multi-decoder  

Phase 2 work: multi-decoder groups G0–G3, Job B_g sequential, FSM merge (see master §7 Phase 2).

---

## 15. Checklist

- [ ] WP1.0 Phase 0 gate  
- [ ] WP1.1 latent cache + manifest  
- [ ] WP1.2 MusicFM load (hard or documented partial)  
- [ ] WP1.3 Job L0 export  
- [ ] WP1.4 PresenceHead  
- [ ] WP1.5 forward_from_latents / freeze  
- [ ] WP1.6 Job A train + metrics  
- [ ] WP1.7 modular checkpoints  
- [ ] WP1.8 mini dataset  
- [ ] WP1.9 docs  
- [ ] E1.1–E1.8  

---

## 16. Definition of done (кратко)

> На GTX 1070 (или documented backend): из wav+notes построен cache (tokens+latents+manifest); Job A учит adapter+presence на latents без encoder в optim; presence F1 логируется; чекпоинты модульные; команды в `scripts/job_L0_cache.md` и `scripts/job_A.md`.

---

*Phase 1 = Cache (L0) + Job A only. Multi-decoder / MIROS F1 = Phase 2–3. Master plan file not modified.*