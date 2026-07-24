# Phase 2 — Modular decoders B_g

**Master-план:** [`PLAN_Denote_Free_Modular.md`](PLAN_Denote_Free_Modular.md)  
(ссылки: §2.1 Modules · §2.4 Infer merge · §2.5–2.6 groups/budget · §3 configs `free_1070_jobB_{g}` · §5 Job B_g · §7 Phase 2)

**Зависимости:**

| Фаза | Документ | Что должно быть готово |
|------|----------|------------------------|
| Phase 0 | [`PLAN_Phase0_Platform_Proof.md`](PLAN_Phase0_Platform_Proof.md) | backend, RunLogger/JSONL, resume, TB |
| Phase 1 | [`PLAN_Phase1_Cache_JobA.md`](PLAN_Phase1_Cache_JobA.md) | latent+token cache, manifest, Job A adapter+presence ckpt, `forward_from_latents` |

**Цель фазы:**

1. Ввести **модульные Transformer-декодеры** `decoder_g` по **группам инструментов** G0…G−1 (master §2.6).  
2. **Job B_g** — обучать **по одному** декодеру на 1070 (encoder + adapter/presence frozen), sequential jobs.  
3. **Infer merge:** presence gate → AR decode активных групп → **FSM** → merge notes.  
4. **Ablation:** single shared decoder vs modular multi-decoder (хотя бы на mini val).

**Не цель фазы:** full FluidR3 corpus scale (Phase 3), multi-onset F1 > MIROS 0.5998 как hard gate (Phase 3 target), Job C joint polish (optional soft), scale ~1B (Phase 5), classic routed MoE.

---

## 1. Контекст master-плана

### 1.1 Job B_g (master §5)

```text
Load: Job A weights (freeze adapter; optional tiny LR later in Job C)
Trainable: decoder_g only
Loss: CE on tokens of programs in group g (mask others)
Repeat g = 0..G-1
Compute: GTX 1070 sequential; latent cache (no encoder in VRAM)
```

### 1.2 Infer merge (master §2.4)

```text
Z = encoder(wav) | load cache
p = σ(presence(adapter(Z)))
for g in groups:
  if max/any p[programs_in_g] > τ OR force_all:
    toks_g = AR_decode(decoder_g, adapter(Z), fsm_g)
notes = detokenize_merge(toks_*) + global FSM (program budget, no double notes)
```

### 1.3 Instrument groups (master §2.6) — default G=4

| Group id | Name | Programs (K=8 challenge) |
|----------|------|---------------------------|
| G0 | keys | Piano (0) |
| G1 | strings | Violin (1), Viola (2), Cello (3) |
| G2 | wood | Flute (4), Oboe (7), Bassoon (5) |
| G3 | brass | Trombone (6) |

Config-driven (`groups.toml` / section in jobB config). Optional later: G=8 (one decoder per program) without redesign.

### 1.4 Parameter budget (master §2.5)

| Module | Trainable in Job B_g |
|--------|----------------------|
| decoder_g | ~15–40M target (tiny configs OK for 1070 smoke) |
| adapter, presence, encoder | **frozen** |

### 1.5 Architecture family

Hybrid already fixed: Conformer encoder + RNN adapter + **Transformer decoders × G** + presence MLP + FSM rules.  
Phase 2 implements the **×G decoders + train/infer modular path**, not a new backbone family.

---

## 2. Baseline в репозитории (ожидаемое к старту Phase 2)

| Компонент | Ожидание после Phase 1 | Пробел Phase 2 |
|-----------|------------------------|----------------|
| `TokenDecoder` single | есть T5-style TF forward | нет multi-instance registry |
| `DenoteMiros` | 1 decoder | `DenoteModular` / multi-decoder |
| Job A ckpt | adapter + presence | load API for freeze |
| Latent+token cache | manifest | **per-group token sequences** filter |
| FSM `valid_next` | per-stream state | multi-stream merge + program budget |
| AR greedy decode | partial / missing in infer | constrained AR per group |
| Job B train loop | нет | WP2.5 |
| Ablation harness | нет | WP2.8 |

---

## 3. Exit criteria (Definition of Done Phase 2)

| ID | Критерий | Проверка |
|----|----------|----------|
| E2.0 | Phase 1 E1.* hard success (or waiver + documented stubs) | checklist Phase 1 |
| E2.1 | Config groups G0–G3 (+ mapping program→group) | `configs/groups_challenge8.toml` |
| E2.2 | `DenoteModular` (or equiv): N decoders, freeze flags, load Job A | unit + load test |
| E2.3 | Per-group token targets: filter notes/tokens by programs in g | cache or on-the-fly builder |
| E2.4 | **Job B_g** runs for **each** g (or ≥2 groups on mini data if time) with only `decoder_g` trainable | log trainable param names |
| E2.5 | Sequential train on 1070 / documented backend; resume per group | `runs/jobB_g{0..}/` |
| E2.6 | Infer path: presence gate + AR decode active groups + detokenize merge | CLI or lib test on 1 clip |
| E2.7 | FSM: illegal tokens blocked in AR; optional program budget | unit tests + decode smoke |
| E2.8 | Ablation table (mini val): modular vs single-decoder baseline | markdown/json metrics |
| E2.9 | Docs: `scripts/job_B.md`, configs `free_1070_jobB_{g}.toml` | reproducible commands |

**Hard success:** E2.1–E2.7 + E2.9 on mini cache (G≥2 trained).  
**Soft success:** all G=4 trained + E2.8 ablation shows modular ≥ single on multi-instrument clips (leakage/precision proxy).

**Не блокирует Phase 2:** multi-onset F1 > 0.5998, full FluidR3, Job C, bit-exact MusicFM.

---

## 4. Scope / Non-scope

### In scope

1. Group config + program→group map  
2. Multi-decoder module container + independent ckpt I/O  
3. Token sequence **filtering / re-encode** per group from notes  
4. Job B train loop (latent cache, freeze adapter/presence)  
5. Constrained AR decode (greedy) with FSM per group stream  
6. Presence-gated skip of inactive groups  
7. Merge notes from multiple streams (time-order, program tags)  
8. Single-decoder baseline config for ablation  
9. Metrics: `loss_ce` per group, optional token accuracy, simple note F1 if cheap  
10. Docs + 1070 sequential schedule  

### Out of scope

- Beam search / length penalty (optional stretch; master lists beam as later edge)  
- Cross-stem Slakh curriculum at scale (Phase 3)  
- Job C joint unfreeze  
- Presence re-train (freeze Job A; optional light LR stretch only)  
- MoE-FFN inside decoder (Phase 5)  
- Product GuitarSet finetune (Phase 4)  

---

## 5. Design details

### 5.1 Module layout

```text
DenoteModular {
  adapter: InstrumentAdapter,          // load Job A, freeze
  presence: PresenceHead,              // load Job A, freeze
  decoders: Vec<TokenDecoder> | struct { g0, g1, g2, g3 },
  // encoder: optional at infer from wav; Job B uses Z from cache
}

// Train Job B_g:
//   optim = params(decoders[g]) only
//   memory = adapter.forward_from_latents(Z, program_cond_g)
//   logits = decoders[g].forward(tokens_in_g, memory)
//   loss = CE(logits, tokens_tgt_g, ignore_pad)
```

**Program conditioning for group g:**  
sum/mean embeddings of programs **in group g that are active in the clip** (from multi-hot labels); if none active, sample may be skipped for that Job B_g (or train empty→EOS only — prefer **skip** samples with no notes in g).

### 5.2 Per-group token targets

Full multi-program token stream (Phase 1) is **not** ideal as-is for one group decoder.

**Required pipeline:**

```text
notes_clip
  → filter notes where program ∈ group_g
  → encode_notes → token_ids_g  (BOS … EOS)
  → store tokens_g/{id}.g{g}.tok.bin
     OR compute on the fly from notes JSON in manifest
```

Manifest extension (optional fields):

```json
{
  "id": "clip_0001",
  "latent": "latents/clip_0001.lat.safetensors",
  "tokens": "tokens/clip_0001.tok.bin",
  "tokens_by_group": {
    "0": "tokens_g/clip_0001.g0.tok.bin",
    "1": "tokens_g/clip_0001.g1.tok.bin"
  },
  "programs": [0, 1],
  "groups_present": [0, 1],
  "split": "train"
}
```

**Empty group:** if no notes in g → either omit from Job B_g epoch or use sequence `[BOS, EOS]` with low weight — **default: skip sample for that g**.

### 5.3 Presence gate at infer

```text
p[k] = σ(presence_logits[k])
group_active[g] = max_{k in programs(g)} p[k] > τ
// τ default 0.5; config: presence_threshold
// force_all=true for ablation / debug
```

If group inactive → do not run decoder_g (saves time, reduces leakage).

### 5.4 AR decode + FSM

Per group stream:

```text
state = DecodeState::new()
state.program = primary_program_of_group or first active
tokens = [BOS]
loop until EOS or max_len:
  logits = decoder_g.forward_step / full TF-style prefix
  mask illegal via fsm::valid_next
  pick argmax among legal (greedy)
  update state
```

**Global merge:**

1. Detokenize each stream → `NoteEvent{start,end,pitch,velocity,program}`  
2. Concatenate all groups  
3. Sort by onset  
4. Global rules: drop duplicates (same program+pitch+onset±ε); optional max programs budget  

Reuse/extend `denote-tokens` + `denote-decode`.

### 5.5 Single-decoder baseline (ablation)

Config `free_1070_jobB_single.toml`:

- One `TokenDecoder` trained on **full** multi-program token sequences (Phase 1 tokens)  
- Same adapter freeze  
- Compare vs modular on: presence-gated note precision proxy / token CE on multi clips / simple note F1 if eval ready  

Master C7: architecture claim needs matched comparison.

### 5.6 Loss

| Loss | Weight | Notes |
|------|--------|-------|
| Token CE group g | 1.0 | ignore PAD |
| Presence | 0 | frozen Job A (no train) |
| Label smoothing | 0–0.1 | optional |

---

## 6. Work packages

### WP2.0 — Gate Phase 1 (0.25 day)

- [ ] Job A ckpt loadable  
- [ ] Manifest + latents exist for mini set  
- [ ] `forward_from_latents` works  

**Blocker:** if Phase 1 incomplete, only design stubs + unit tests for multi-decoder without full train.

---

### WP2.1 — Groups config (0.25 day)

**Files:**

- `configs/groups_challenge8.toml`  
- Rust: `denote-model` or `denote-tokens` `GroupsConfig { groups: Vec<GroupDef> }`  

```toml
[[group]]
id = 0
name = "keys"
programs = [0]

[[group]]
id = 1
name = "strings"
programs = [1, 2, 3]
# ...
```

**Acceptance:** parse + `program_to_group(p) -> Option<g>`.

---

### WP2.2 — Per-group token builder (1 day)

- Filter notes by group programs  
- Encode with existing `encode_notes` / vocab  
- CLI or train-side: `build-group-tokens --manifest ... --groups ...`  
- Tests: piano-only clip → only g0 non-empty  

**Acceptance:** E2.3.

---

### WP2.3 — DenoteModular + freeze/load (1–1.5 days)

**Files:**

- `crates/denote-model/src/modular.rs`  
- `multi_decoder.rs` if split  

API sketch:

```rust
pub struct DenoteModularConfig { adapter, presence, decoder: TokenDecoderConfig, n_groups }
pub struct DenoteModular<B> {
  adapter, presence, decoders: Vec<TokenDecoder<B>>,
}
impl DenoteModular {
  pub fn load_job_a(...);
  pub fn load_decoder_g(g, path);
  pub fn save_decoder_g(g, path);
  pub fn forward_group_tf(g, z, tokens_in, program_ids) -> logits;
  pub fn trainable_decoder_params(g) -> ...
}
```

**Acceptance:** E2.2; only decoder_g in optim list for Job B.

---

### WP2.4 — Constrained AR + merge (1–2 days)

**Files:**

- `denote-decode` or `denote-infer`: `ar_decode_group`, `merge_note_events`  
- Wire FSM `valid_next` / `step`  

**Acceptance:** E2.6–E2.7 unit tests: illegal NoteOff rejected; two groups merge sorted.

---

### WP2.5 — Job B train loop (1–2 days)

**Configs:**

- `configs/free_1070_jobB_g0.toml` … `g3.toml`  
  or one config + `--group 0`

```toml
[run]
name = "jobB_g0"
job_a_ckpt = "runs/jobA_001/checkpoints/last"

[data]
latent_manifest = "data/cache/phase1_mini/manifest.jsonl"
group_id = 0
batch_size = 2

[model]
freeze_adapter = true
freeze_presence = true
decoder_layers = 2   # tiny on 1070; scale later
# d_model from Job A

[optim]
lr = 3e-4
max_steps = 20000
ckpt_every = 500
```

**CLI:**

```powershell
cargo run -p denote-train --release --features <gpu> -- `
  --config configs/free_1070_jobB_g0.toml `
  --job job-b `
  --run-dir runs/jobB_g0
```

**Data rule:** skip samples without group notes.

**Acceptance:** E2.4–E2.5; loss finite; resume works.

**Schedule on 1070 (example):**

```text
Week: g0 train → g1 train → g2 → g3
Each: own run_dir, own metrics.jsonl
```

---

### WP2.6 — Infer CLI smoke (0.5–1 day)

```powershell
cargo run -p denote-infer --release --features <gpu> -- `
  --job-a runs/jobA_.../last `
  --decoders runs/jobB_g0/last,runs/jobB_g1/last,... `
  --groups configs/groups_challenge8.toml `
  --latent data/cache/.../clip.lat   # or --wav
  --presence-threshold 0.5 `
  --out out/notes.json
```

**Acceptance:** JSON/MIDI-like notes from multi-group path.

---

### WP2.7 — Metrics for Job B (0.5 day)

JSONL/TB:

- `train/loss_ce`, `train/lr`, `train/step_time_ms`  
- `val/loss_ce`  
- `data/frac_samples_with_group`  
- optional `val/token_acc`  

Reuse Phase 0 logger.

---

### WP2.8 — Ablation modular vs single (0.5–1 day)

1. Train `jobB_single` on same steps budget **sum** comparable (fairness note in doc)  
2. Eval on multi-instrument mini val:  
   - presence-gated ghost program rate (proxy leakage)  
   - token CE  
   - note onset F1 if `denote-eval` ready (else defer metric quality to Phase 3)  
3. Write `runs/ablation_phase2.md` table  

**Acceptance:** E2.8 artifact exists (numbers can be weak on mini data).

---

### WP2.9 — Documentation (0.5 day)

- `scripts/job_B.md` — sequential train, configs, infer merge, ablation  
- Link from Phase 1 scripts  
- **Do not edit** body of `PLAN_Denote_Free_Modular.md`

---

## 7. File change map

| Path | Action |
|------|--------|
| `PLAN_Phase2_Modular_Decoders.md` | this file (root) |
| `configs/groups_challenge8.toml` | create |
| `configs/free_1070_jobB_g0.toml` … `g3.toml` | create |
| `configs/free_1070_jobB_single.toml` | create (ablation) |
| `crates/denote-model/src/modular.rs` | create |
| `crates/denote-model/src/groups.rs` | create |
| `crates/denote-data` group token filter/builder | extend |
| `crates/denote-decode` AR + merge | extend |
| `crates/denote-tokens` FSM helpers if needed | extend |
| `crates/denote-train/src/jobs/job_b.rs` | create |
| `crates/denote-infer/src/main.rs` | modular load path |
| `scripts/job_B.md` | create |
| `runs/jobB_g*/` | generated |

---

## 8. Implementation order

```text
1. WP2.0 gate
2. WP2.1 groups config
3. WP2.2 per-group tokens
4. WP2.3 DenoteModular + freeze/load Job A
5. WP2.5 Job B train (start g0 on mini)
6. WP2.4 AR+FSM+merge (can parallel with 5 after TF train works)
7. WP2.6 infer smoke
8. WP2.7 metrics polish
9. WP2.8 ablation single vs modular
10. WP2.9 docs + E2.* checklist
```

**Critical path:** WP2.2 → WP2.3 → WP2.5 → WP2.4 → WP2.6.

---

## 9. 1070 envelope (Job B)

| Knob | Start |
|------|-------|
| batch_size | 1–2 |
| decoder n_layers | 2 (smoke) → 4–6 if VRAM allows |
| n_heads / d_ff | match d_model from Job A |
| max token len | 256–512 |
| samples | only clips with group active |
| encoder | **not loaded** (latent cache) |
| adapter+presence | load, freeze, no optim states for them if possible |

If OOM: fewer layers, shorter max_len, batch=1, gradient checkpoint if Burn supports easily.

---

## 10. Test plan

| Test | How |
|------|-----|
| groups map | unit |
| filter notes → tokens_g | unit |
| modular load Job A + one decoder | integration |
| Job B 50 steps g0 | manual 1070/CPU |
| resume Job B | manual |
| FSM rejects bad token | unit |
| merge two groups | unit |
| infer one multi clip | manual |
| ablation artifact | file exists |

---

## 11. Risks

| Risk | Mitigation |
|------|------------|
| Group token encode differs from full stream conventions | shared vocab; document Program tokens in-group stream |
| Adapter was trained with multi-hot emb different from group-cond | use same sum-of-active-programs rule as Job A |
| G=4 × train time long on 1070 | mini data first; train g0+g1 hard, g2+g3 soft |
| Presence false negatives skip true groups | τ sweep; force_all debug; log skip rate |
| Single vs modular unfair compute | document step budget; equal total decoder steps ≈ G × steps_single/G |
| Phase 1 ckpt missing | gate WP2.0; synthetic adapter random only for compile tests |

---

## 12. Handoff → Phase 3

Phase 3 (Data scale + F1 vs MIROS) expects:

1. All (or core) `decoder_g` checkpoints + Job A  
2. Stable infer merge + FSM  
3. Ablation methodology  
4. Ability to train Job B on **larger** FluidR3/Slakh manifests without API change  

Phase 3 adds data volume, curriculum 1→3 instruments, multi-onset F1 protocol, Kaggle only if overflow.

---

## 13. Checklist

- [ ] WP2.0 Phase 1 gate  
- [ ] WP2.1 groups config  
- [ ] WP2.2 per-group tokens  
- [ ] WP2.3 DenoteModular  
- [ ] WP2.4 AR + FSM + merge  
- [ ] WP2.5 Job B_g train sequential  
- [ ] WP2.6 infer smoke  
- [ ] WP2.7 metrics  
- [ ] WP2.8 ablation  
- [ ] WP2.9 docs  
- [ ] E2.1–E2.9  

---

## 14. Definition of done (кратко)

> Модульные `decoder_g` обучаются **по одному** на latent cache с замороженными adapter/presence (Job A); infer включает presence-gate, FSM-AR и merge нот; есть ablation vs single decoder; команды в `scripts/job_B.md`. Master-план не изменялся.

---

*Phase 2 = modular multi-decoder train + infer merge. Data scale & MIROS F1 = Phase 3. Not classic MoE.*