# Phase 3 — Data scale + F1 vs MIROS

**Master-план:** [`PLAN_Denote_Free_Modular.md`](PLAN_Denote_Free_Modular.md)  
(ссылки: North Star C6–C7 · §1 compute 1070/Kaggle · §2.3–2.4 losses/infer · §5 Job C/E · §6 Data pipeline · §7 Phase 3 · metric Multi Onset F1 **0.5998**)

**Зависимости:**

| Фаза | Документ | Что должно быть готово |
|------|----------|------------------------|
| Phase 0 | [`PLAN_Phase0_Platform_Proof.md`](PLAN_Phase0_Platform_Proof.md) | backend, RunLogger, resume, TB |
| Phase 1 | [`PLAN_Phase1_Cache_JobA.md`](PLAN_Phase1_Cache_JobA.md) | L0 cache pipeline, Job A, presence |
| Phase 2 | [`PLAN_Phase2_Modular_Decoders.md`](PLAN_Phase2_Modular_Decoders.md) | Job B_g, modular infer merge, FSM, groups |

**Цель фазы:**

1. **Масштаб данных** под challenge-domain: FluidR3 render, multi-instrument mixes, optional Slakh **cross-stem** (master §6 — edge vs MIROS, который stem aug почти не использовал).  
2. **Curriculum** 1 → 2 → 3 instruments (master Phase 3 checklist).  
3. Дообучить / продолжить Job A / Job B_g (и опционально **Job C** joint polish) на enlarged cache **на 1070**; Kaggle **только overflow**.  
4. Реализовать и прогнать **Job E**: multi-instrument **Note Onset F1** (program + pitch + onset ±50 ms), macro по инструментам — protocol 2025 AMT Challenge.  
5. **Сравнить с MIROS 0.5998** (best-effort on free wall-clock); ablations architecture (presence / modular / stem / FSM).  
6. Thesis-ready tables: metrics JSON + markdown report.

**Не цель фазы:** product GuitarSet packaging (Phase 4), system ~1B scale (Phase 5), paid GPU requirement, bit-exact reimplementation of MIROS training stack in PyTorch.

---

## 1. Контекст master-плана

### 1.1 North Star metrics (C6, C7)

| ID | Content |
|----|---------|
| C6 | Multi-instrument Note Onset F1 (program + pitch + onset ±50 ms) **> MIROS 0.5998** — **best-effort** on free compute |
| C7 | Architecture > scale: ablations at matched budget before any “we beat them because bigger” claim |

**Published MIROS challenge ref (master):** Multi Onset F1 **0.5998** (paper *Advancing Multi-Instrument Music Transcription… 2025 AMT Challenge*).

**Honest framing:** free/1070 wall-clock may not reach 0.5998; Phase 3 **must** still ship: eval protocol, best score, ablations, gap analysis. Soft success = beat internal baselines; hard success = protocol + report; stretch = F1 > 0.5998.

### 1.2 Phase 3 bullets (master §7)

```text
- more FluidR3 + stem
- curriculum
- multi_onset_f1
- Kaggle only if L0/joint overflow
```

### 1.3 Data pipeline (master §6)

```text
MIDI (Lakh/classical)
  → tools/render_fluidr3.py (CPU, FluidR3_GM)
  → wav 24 kHz mono
  → Job L0 latents + tokens (+ tokens_by_group)
  → optional Slakh stem mix → re-encode multi-program notes
```

### 1.4 Job E (master §5)

```text
Load modular ckpts → multi_onset_f1 on challenge-like val
Log metrics + JSON summary for thesis tables
```

### 1.5 Job C optional (master §5)

```text
Unfreeze adapter (small LR) + all decoders low LR, short run
If OOM → skip; infer-only merge of Job A+B
If need → Kaggle overflow short joint
```

### 1.6 Compute (master §1)

| Workload | Where |
|----------|-------|
| FluidR3 render, MIDI prep | **CPU local** (0 GPU quota) |
| L0 re-cache large set | **1070**; Kaggle if OOM/time |
| Continue Job A/B | **1070** latent cache |
| Job C joint | 1070 if fits; else skip or Kaggle |
| Final eval sweep | 1070 or Kaggle free |

---

## 2. Baseline после Phase 0–2

| Компонент | Ожидание | Пробел Phase 3 |
|-----------|----------|----------------|
| Modular train/infer | Job A + B_g + merge | scale data only |
| Mini cache | dozens of clips | **orders of magnitude** more |
| FluidR3 tool | stub/optional Phase 1 | **production render** |
| Cross-stem mix | not required | **implement** (Slakh or synth stems) |
| Curriculum sampler | none | 1→2→3 instrument sampling |
| `denote-eval` | onset F1 pitch+time, **no program** in match | **multi-instrument onset F1 + program** |
| Challenge-like val | none | held-out classical multi, FluidR3 domain |
| Report vs MIROS | none | tables + JSON |

**Eval gap (code):** current `evaluate_notes` matches pitch + onset (±tol), optional offset — **no program dimension**. Challenge multi-onset needs **program+pitch+onset**. Extend in WP3.4.

---

## 3. Exit criteria (Definition of Done Phase 3)

| ID | Критерий | Проверка |
|----|----------|----------|
| E3.0 | Phase 2 hard success (modular infer + ≥2 group decoders) | Phase 2 checklist |
| E3.1 | FluidR3 render path documented + runs on CPU for ≥ **N_render** MIDI files | `tools/render_fluidr3.py` + `scripts/data_scale.md` |
| E3.2 | Scaled cache: train manifest ≥ **N_train** chunks; val ≥ **N_val** | manifest counts |
| E3.3 | Cross-stem **or** multi-source mix pipeline exists (even if subset of data) | code + example |
| E3.4 | Curriculum sampler: stages by `n_instruments` ∈ {1,2,3} | config + logs `data/n_instruments_mean` |
| E3.5 | Continued training Job A and/or Job B_g on scaled data with resume | `runs/phase3_*` |
| E3.6 | **Multi-onset F1** metric: program + pitch + onset ±50 ms; macro over instruments | unit tests + Job E |
| E3.7 | Job E eval on challenge-**like** val (not train leakage); JSON report | `runs/eval_*/report.json` |
| E3.8 | Ablations logged: ≥3 of {no presence gate, single decoder, no stem, no FSM} | `runs/ablation_phase3.md` |
| E3.9 | Comparison table vs MIROS 0.5998 + internal baselines; gap analysis if below | thesis-ready md/json |
| E3.10 | Kaggle used only if documented overflow; primary path 1070+CPU | `scripts/data_scale.md` §compute |

**Quantity targets (tunable; disk/time limited):**

| Symbol | Hard min (close phase) | Soft target (quality) |
|--------|------------------------|------------------------|
| N_render | 50 MIDI renders | 500–5000+ |
| N_train chunks | 500 | 5k–50k |
| N_val chunks | 50 | 200–500 |
| Max instruments in mix | 3 | 3 (challenge-like) |

**Stretch (C6):** Multi Onset F1 **> 0.5998** on defined val protocol.  
**Hard success without stretch:** E3.1–E3.10 with best F1 reported honestly.

**Leakage rules:**

- Do **not** train on official challenge test pieces if labels/rules forbid.  
- Challenge samples only style-check / optional separate probe (master data table).  
- Val must be disjoint MIDI ids from train.

---

## 4. Scope / Non-scope

### In scope

1. `tools/render_fluidr3.py` + SoundFont path docs  
2. MIDI corpus ingest (Lakh subset / classical MIDI / public domain) filtered to challenge-like programs when possible  
3. Chunking long pieces → many cache items  
4. Rebuild/extend L0 latent+token(+group) cache for scaled set  
5. Cross-stem mixing (Slakh stems preferred; fallback: mix independent mono renders)  
6. Curriculum scheduling in train dataloaders  
7. Retrain/finetune Job A + Job B_g (and optional Job C)  
8. Multi-instrument onset F1 with **program**  
9. Job E CLI + reports  
10. Ablations + MIROS comparison writeup  
11. Optional: offset F1 / overlap metrics as secondary (paper also reports overlap)  

### Out of scope

- Shipping deNote product UX (Phase 4)  
- Widening to ~1B params (Phase 5)  
- Guaranteed SOTA without enough data/hours  
- Full official challenge private leaderboard submit (unless public protocol allows)  
- Training encoder from scratch  

---

## 5. Metric specification (Job E) — critical

### 5.1 Multi-instrument Note Onset F1 (challenge-aligned)

**Match est note ↔ ref note if all hold:**

1. **Program** equal (GM program id or mapped challenge instrument id)  
2. **Pitch** equal (MIDI number)  
3. **Onset** \|t_est − t_ref\| ≤ **50 ms** (0.05 s)

Then greedy bipartite matching (as current `evaluate_notes`) per piece; aggregate:

- **Per-instrument** P/R/F1  
- **Macro F1** = unweighted mean over instruments present in ref (or over fixed K=8; **document choice** — prefer macro over instruments that appear in val)  
- Optional micro F1 for analysis  

**Secondary (optional, paper-like):**

- Note with offset F1 (existing offset rule)  
- Overlap / IoU style if easy  
- Presence-only F1 (from Phase 1 head)  

### 5.2 API sketch

```rust
// denote-eval
pub struct RefNoteMulti {
  pub start_sec: f64,
  pub end_sec: f64,
  pub pitch_midi: u8,
  pub program: u8,  // NEW
}

pub fn evaluate_notes_multi_onset(
  est: &[NoteEvent],  // NoteEvent must carry program
  reference: &[RefNoteMulti],
  onset_tolerance_sec: f64, // 0.05
) -> MultiInstrumentReport;
```

Ensure `NoteEvent` / detokenize emits **program**.

### 5.3 Job E outputs

```text
runs/eval_<timestamp>/
  report.json       # overall + per-instrument F1
  per_file.jsonl    # one line per val clip
  config_snapshot.toml
  ckpt_paths.json
```

CLI:

```powershell
cargo run -p denote-eval --release -- `
  --manifest data/cache/phase3/manifest_val.jsonl `
  --job-a runs/jobA_phase3/last `
  --decoders runs/jobB_g0/... `
  --onset-tol 0.05 `
  --out runs/eval_phase3
```

(or `denote-infer --eval` dispatch — choose one entrypoint, document it)

---

## 6. Data scale design

### 6.1 Sources (priority order)

| Priority | Source | Role |
|----------|--------|------|
| 1 | Classical / multi-track MIDI → **FluidR3_GM** | domain match challenge synth |
| 2 | **Slakh2100** stems | cross-stem mix, multi-instrument |
| 3 | MusicNet / URMP | real multi (finetune / secondary val) |
| 4 | MAESTRO | piano-heavy (careful not to dominate) |
| 5 | Phase 1 mini synth | smoke only |

### 6.2 FluidR3 render tool

**File:** `tools/render_fluidr3.py`

```text
Input:  MIDI path, sf2 path (FluidR3_GM.sf2), out wav, sr=24000, mono
Deps:   pretty_midi / mido + fluidsynth bindings or CLI fluidsynth
Output: wav + sidecars notes.json (program, pitch, onset, offset, vel)
```

**CPU-only**; batch script over MIDI list.  
Document SF2 download license path in `scripts/data_scale.md` (user supplies SF2).

### 6.3 Chunking

```text
long wav/notes → sliding or non-overlap windows chunk_sec=4–5
each chunk = cache id, programs present in window, n_instruments
```

Prefer non-overlap for train volume; optional 50% overlap for more samples.

### 6.4 Cross-stem augmentation (MIROS gap)

**Mode Stem-A (Slakh):**

```text
sample 1..3 stems from different instruments
mix audio sum (normalize peak)
union note lists with program ids
→ new synthetic multi clip → L0
```

**Mode Stem-B (no Slakh):**

```text
render monophonic MIDI per program separately
mix N renders + merge note lists
```

Rate: e.g. 30–50% of train batches from mixes (config).

### 6.5 Curriculum (1→2→3)

```text
Stage C1: only n_instruments == 1   (steps 0..S1)
Stage C2: n_instruments <= 2        (S1..S2)
Stage C3: n_instruments <= 3        (S2..end)
```

Sampling: within stage, uniform over eligible cache ids.  
Log `data/n_instruments_mean` each epoch.

### 6.6 Cache layout (scale)

```text
data/cache/phase3/
  manifest_train.jsonl
  manifest_val.jsonl
  latents/
  tokens/
  tokens_g/
  notes/           # optional raw note json per id
  render_index.jsonl  # midi → wav provenance
```

Rebuild L0 using Phase 1 tools; **incremental** cache append if supported (skip existing ids).

---

## 7. Training plan on scaled data

### 7.1 Order

```text
1) Render + chunk + L0 cache (CPU + 1070 L0)
2) Finetune Job A (presence+adapter) on new manifest — resume Job A ckpt
3) Finetune each Job B_g sequential — resume Job B ckpts
4) Optional Job C short joint polish
5) Job E eval + ablations
6) Iterate data hard cases (false programs, 3-inst mixes) if F1 low
```

### 7.2 Hyperparams (starting points)

| Job | LR | Notes |
|-----|-----|-------|
| A finetune | 1e-4 | lower than Phase 1 scratch |
| B_g finetune | 1e-4 | per group; more steps on underperforming groups |
| C joint | 1e-5 adapter, 3e-5 decoders | few k steps; skip if OOM |

### 7.3 Hard negatives

- Co-occurring similar instruments (violin/viola)  
- Dense polyphony piano + strings  
- Curriculum final stage emphasize n_instruments=3  

### 7.4 Optional Job C (WP3.6)

Only if VRAM allows loading adapter + all G decoders with small batch.  
Else document skip; Phase 3 success does not require Job C.

---

## 8. Ablations (C7)

Minimum set for `runs/ablation_phase3.md`:

| ID | Condition | What we learn |
|----|-----------|---------------|
| A0 | Full system (presence + modular + FSM + stem data) | main result |
| A1 | No presence gate (force_all decode) | leakage / precision |
| A2 | Single decoder (Phase 2 baseline ckpt finetuned same data) | modular value |
| A3 | No stem-mix data (FluidR3 only plain) | stem aug value |
| A4 | Greedy AR without FSM | constraint value |

Same val set; report Multi Onset F1 + precision/recall.

---

## 9. Work packages

### WP3.0 — Gate Phase 2 (0.25 day)

- [ ] Modular infer produces notes with program  
- [ ] Job A/B ckpts load  
- [ ] Mini eval smoke on old cache  

---

### WP3.1 — FluidR3 render pipeline (1–2 days)

- [ ] `tools/render_fluidr3.py`  
- [ ] Batch driver + notes JSON export  
- [ ] `scripts/data_scale.md` §render  
- [ ] Smoke: 5 MIDI → 5 wav+notes  

**Acceptance:** E3.1 hard min path.

---

### WP3.2 — Corpus + chunk + manifests (1–2 days)

- [ ] MIDI list curation (challenge-like programs preferred)  
- [ ] Chunker → per-chunk notes + wav segments  
- [ ] train/val split by piece id  
- [ ] Provenance index  

**Acceptance:** N_train/N_val hard mins.

---

### WP3.3 — Cross-stem mix (1–2 days)

- [ ] Stem mix implementation (Slakh or Stem-B)  
- [ ] Config fraction  
- [ ] Mixed items enter L0 queue  

**Acceptance:** E3.3.

---

### WP3.4 — Multi-onset F1 with program (1 day)

- [ ] Extend `NoteEvent` / eval types with program  
- [ ] `evaluate_notes_multi_onset`  
- [ ] Unit tests: TP/FP/FN with program mismatch cases  
- [ ] Macro aggregation  

**Acceptance:** E3.6 unit tests green.

---

### WP3.5 — L0 re-cache scaled set (1–3 days wall)

- [ ] Batch L0 on 1070; resume/skip existing  
- [ ] tokens_g for all groups  
- [ ] Disk budget estimate in docs  

**Acceptance:** E3.2; Kaggle only if written overflow log.

---

### WP3.6 — Curriculum + finetune Job A/B (2–5 days wall)

- [ ] Curriculum sampler in train data  
- [ ] Resume Job A finetune  
- [ ] Resume Job B_g × G  
- [ ] Optional Job C  
- [ ] TB/JSONL metrics continuous  

**Acceptance:** E3.4, E3.5.

---

### WP3.7 — Job E eval harness (1 day)

- [ ] CLI end-to-end val  
- [ ] `report.json` schema  
- [ ] Threshold τ sweep optional (presence)  

**Acceptance:** E3.7.

---

### WP3.8 — Ablations + MIROS comparison report (1 day)

- [ ] Run A0–A4 (subset if time: A0,A1,A2 minimum)  
- [ ] Table vs 0.5998  
- [ ] Gap analysis (data hours, domain, model size)  
- [ ] `runs/ablation_phase3.md` + thesis numbers  

**Acceptance:** E3.8, E3.9.

---

### WP3.9 — Documentation (0.5 day)

- [ ] `scripts/data_scale.md` — render, cache, curriculum, Kaggle overflow  
- [ ] `scripts/eval_challenge_f1.md` — metric definition, Job E commands  
- [ ] **Do not modify** `PLAN_Denote_Free_Modular.md` body  

---

## 10. File change map

| Path | Action |
|------|--------|
| `PLAN_Phase3_Data_F1_vs_MIROS.md` | this file (root) |
| `tools/render_fluidr3.py` | create/complete |
| `tools/chunk_and_manifest.py` or Rust bin | create |
| `tools/stem_mix.py` or Rust | create |
| `crates/denote-eval` | multi-onset + program |
| `crates/denote-decode` / notes | ensure program on NoteEvent |
| `crates/denote-train` | curriculum sampler; phase3 configs |
| `crates/denote-infer` or eval bin | Job E |
| `configs/phase3_jobA_ft.toml` | create |
| `configs/phase3_jobB_g*.toml` | create |
| `configs/phase3_curriculum.toml` | create |
| `scripts/data_scale.md` | create |
| `scripts/eval_challenge_f1.md` | create |
| `data/cache/phase3/` | generated |
| `runs/eval_*`, `runs/ablation_phase3.md` | generated |

---

## 11. Implementation order

```text
1. WP3.0 gate
2. WP3.4 metric (can parallel early — unblocks eval mindset)
3. WP3.1 FluidR3 render
4. WP3.2 corpus chunk manifests
5. WP3.3 stem mix
6. WP3.5 L0 scaled cache
7. WP3.6 curriculum + finetune A/B
8. WP3.7 Job E
9. WP3.8 ablations + MIROS table
10. WP3.9 docs
```

**Critical path:** render → cache → finetune → Job E.  
**Metric WP3.4** early reduces last-minute eval bugs.

---

## 12. 1070 / free compute envelope

| Activity | GPU? | Notes |
|----------|------|-------|
| Render FluidR3 | no | multi-core CPU |
| Stem mix | no | CPU |
| L0 encode | yes | B=1; nights/weekends; resume |
| Job A/B ft | yes | latent cache; batch 2–8 |
| Job C | yes/risky | optional |
| Job E | yes/light | AR decode bound |

**Kaggle overflow triggers:** L0 backlog too large; Job C OOM; need longer continuous session than local practicality. Always pull metrics JSONL back local for TB.

---

## 13. Test plan

| Test | How |
|------|-----|
| Program mismatch → no TP | unit eval |
| Onset 49 ms match / 51 ms miss | unit |
| Macro F1 hand-computed fixture | unit |
| Render 1 MIDI | manual |
| Curriculum only n=1 at stage C1 | log assert |
| Job E dry-run 5 val clips | manual |
| Ablation scripts runnable | manual |

---

## 14. Risks

| Risk | Mitigation |
|------|------------|
| Disk full (latents) | float16 latents optional; chunk limit; external drive |
| FluidSynth install pain on Windows | document CLI fluidsynth; WSL fallback |
| F1 ≪ 0.5998 | ablations + more data iterate; honest report; architecture still thesis value |
| Val contamination | piece-level split; id lists committed |
| Eval without program on notes | fix detokenize before Job E |
| Slakh download huge | Stem-B mixes from FluidR3 monos first |
| 1070 weeks of train | curriculum focus; train worst groups more; free Kaggle overflow |

---

## 15. Handoff → Phase 4

Phase 4 (Product) expects:

1. Best Phase 3 modular ckpt bundle  
2. Job E script for regression  
3. Documented F1 number + ablations  
4. Stable infer merge for wav→notes  

Phase 4: `denote-infer` product UX, GuitarSet light finetune — **not** redefining challenge metric.

---

## 16. Checklist

- [ ] WP3.0 Phase 2 gate  
- [ ] WP3.1 FluidR3 render  
- [ ] WP3.2 corpus + chunks  
- [ ] WP3.3 cross-stem  
- [ ] WP3.4 multi-onset F1 + program  
- [ ] WP3.5 L0 scaled cache  
- [ ] WP3.6 curriculum + finetune A/B/(C)  
- [ ] WP3.7 Job E reports  
- [ ] WP3.8 ablations + vs MIROS 0.5998  
- [ ] WP3.9 docs  
- [ ] E3.1–E3.10  

---

## 17. Definition of done (кратко)

> На free/1070 path: challenge-like данные (FluidR3 ± stem) закэшированы; curriculum 1→3; модели A/B дообучены; Job E считает **multi-instrument onset F1 (program+pitch+onset ±50 ms)**; есть ablations и таблица vs **MIROS 0.5998** (stretch: превзойти). Master-план не изменялся.

---

*Phase 3 = data + metric + fair comparison. Architecture already from Phase 1–2; here we prove it on real volume and challenge protocol.*