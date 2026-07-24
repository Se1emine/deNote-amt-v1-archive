# Phase 4 — Product

**Master-план:** [`PLAN_Denote_Free_Modular.md`](PLAN_Denote_Free_Modular.md)  
(ссылки: North Star practically full Rust · §2.4 Infer merge · §3 repo `denote-infer` · §7 Phase 4 · handoff from Phase 3)

**Зависимости:**

| Фаза | Документ | Что должно быть готово |
|------|----------|------------------------|
| Phase 0–2 | PLAN_Phase0… / Phase1 / Phase2 | train loop, modular model, infer merge+FSM |
| Phase 3 | [`PLAN_Phase3_Data_F1_vs_MIROS.md`](PLAN_Phase3_Data_F1_vs_MIROS.md) | best modular ckpt bundle, Job E regression, documented F1, stable wav→notes |

**Цель фазы (master §7 Phase 4):**

1. **`denote-infer` product path:** загрузка **module dir** (Job A + decoders_g + groups + vocab + thresholds), не legacy single CRNN ckpt.  
2. **End-user CLI:** `wav → JSON notes + optional MIDI`, chunked long audio, presence gate, FSM merge.  
3. **GuitarSet light finetune** (Job B-style / domain adapter) на **GTX 1070** — product domain deNote (гитара), secondary metric.  
4. **Release bundle:** weights layout, configs, README how-to-run, regression smoke.  
5. Подготовка к интеграции с **deNote** (контракт `NoteEvent`: start/end, midi pitch, velocity, program).

**Не цель фазы:** scale system ~1B (Phase 5), re-run full challenge data pipeline, new backbone architecture, paid GPU, mobile/WASM packaging (optional stretch only).

---

## 1. Контекст master-плана

### 1.1 Phase 4 checklist (master §7)

```text
- denote-infer loads module dir
- GuitarSet light Job B-style finetune on 1070
```

### 1.2 Infer merge contract (master §2.4) — product must implement

```text
Z = encoder(wav) | (optional precomputed — not for end users)
p = σ(presence(adapter(Z)))
for g in groups:
  if p[programs_in_g] > τ:
    toks_g = AR_decode(decoder_g, adapter(Z), fsm_g)
notes = detokenize_merge(toks_*) + global FSM
```

End users always start from **wav** (encoder loaded for infer; freeze still).

### 1.3 Stack constraint (C4)

- Infer primary: **Rust/Burn** (`denote-infer`)  
- Python only offline tools (already), not runtime for product path  
- deNote integration: consume JSON/MIDI or future FFI — Phase 4 defines **stable on-disk + CLI contract**

### 1.4 Compute (C1)

| Workload | Where |
|----------|-------|
| Product infer (demo tracks) | 1070 or CPU |
| GuitarSet finetune | **1070**, small Job B-style / LoRA-ish small head |
| Regression Job E subset | 1070/CPU |

---

## 2. Baseline в репозитории

| Компонент | Сейчас | Phase 4 gap |
|-----------|--------|-------------|
| `denote-infer` | **legacy CRNN** path (`DenoteCrnn`, `v1_synth.toml`) | modular MIROS path missing |
| Modular load | designed Phase 2–3 | unify into product CLI |
| MIDI export | optional flag exists (legacy) | ensure multi-program MIDI |
| Module dir layout | ad hoc `runs/job*` | **versioned release layout** |
| GuitarSet | mentioned in README / old configs | light FT pipeline |
| User docs | developer-centric | **product quickstart** |
| deNote contract | `NoteEvent` in decode | JSON schema + examples |

---

## 3. Exit criteria (Definition of Done Phase 4)

| ID | Критерий | Проверка |
|----|----------|----------|
| E4.0 | Phase 3 hard success **or** waiver with best available modular ckpts | documented |
| E4.1 | `denote-infer` loads **module directory** (not only single .mpk CRNN) | CLI help + run |
| E4.2 | Command: wav → `notes.json` with fields start/end/pitch/velocity/program | schema test |
| E4.3 | Optional `--midi out.mid` multi-track or program-tagged | opens in DAW/midi player |
| E4.4 | Long audio: chunked infer + stitch notes (overlap policy documented) | 30–60 s wav smoke |
| E4.5 | Presence threshold + force_all flags exposed | CLI |
| E4.6 | Release layout `models/denote_modular_v1/` with README | copy-paste run |
| E4.7 | GuitarSet: data prep + light finetune job on 1070 completes | `runs/guitarset_ft_*` |
| E4.8 | Post-FT or base model: qualitative guitar demo + optional GuitarSet note F1 | report snippet |
| E4.9 | Regression: Job E (or subset) still runnable on challenge-like val after packaging | no metric collapse silent |
| E4.10 | Docs: `scripts/product_infer.md`, update root README product section | user can run without reading all plans |

**Hard success:** E4.1–E4.6 + E4.10 (modular product path).  
**Soft success:** E4.7–E4.8 GuitarSet FT.  
**Stretch:** faster infer (batch chunks), simple quantize, deNote crate example consumer.

---

## 4. Scope / Non-scope

### In scope

1. Refactor/extend `denote-infer` for **DenoteModular**  
2. Module dir format + loader  
3. JSON schema for notes (+ pretty print)  
4. MIDI writer multi-program  
5. Chunked long-form transcription  
6. Config TOML for product infer (`configs/infer_modular.toml`)  
7. GuitarSet download/prep notes, L0-lite or wav finetune path, small train job  
8. Packaging script or doc: copy ckpts → `models/`  
9. Smoke tests / `--dry-run`  
10. README quickstart for end users / diploma demo  

### Out of scope

- Phase 5 width/MoE/1B  
- Retraining full challenge corpus  
- GUI application  
- Real-time streaming low-latency (stretch)  
- App Store / installers  
- Changing challenge metric definition  

---

## 5. Product design

### 5.1 Module directory layout (canonical)

```text
models/denote_modular_v1/
  manifest.json              # version, d_model, K programs, group file ref, git hash
  groups.toml                # copy of groups_challenge8 or product groups
  vocab.json                 # or embed defaults + fingerprint
  encoder/                   # MusicFM weights (or pointer + download script)
    weights.mpk | *.safetensors
  job_a/
    adapter.*
    presence.*
    state.json
  decoders/
    g0.*
    g1.*
    g2.*
    g3.*
  infer.toml                 # defaults: τ, chunk_sec, hop, max_ar_len, sample_rate
  LICENSE_WEIGHTS.txt        # MusicFM / data notices
  README.md                  # how to run
```

`manifest.json` example fields:

```json
{
  "format": "denote_modular_v1",
  "d_model": 512,
  "sample_rate": 24000,
  "n_mels": 128,
  "presence_k": 8,
  "n_groups": 4,
  "onset_tol_eval": 0.05,
  "created_from_runs": ["runs/jobA_phase3/...", "runs/jobB_g0/..."]
}
```

### 5.2 CLI contract (`denote-infer`)

```powershell
cargo run -p denote-infer --release --features cuda -- `
  --model-dir models/denote_modular_v1 `
  --audio path/to/song.wav `
  --out out/notes.json `
  --midi out/notes.mid `
  --device auto `
  --presence-threshold 0.5 `
  --chunk-sec 4.0 `
  --chunk-hop-sec 2.0
```

**Legacy CRNN path:** keep behind `--legacy-crnn` or separate config detection so old diploma ablations still work.

### 5.3 JSON note schema (deNote-facing)

```json
{
  "schema": "denote.notes.v1",
  "audio": "song.wav",
  "sample_rate": 24000,
  "model": "denote_modular_v1",
  "notes": [
    {
      "start_sec": 1.02,
      "end_sec": 1.40,
      "pitch_midi": 64,
      "velocity": 80,
      "program": 0,
      "group": 0
    }
  ]
}
```

### 5.4 Chunking long audio

```text
for window in sliding(chunk_sec, hop_sec):
  notes_w = infer_window(wav[window])
  shift onsets by window_start
  collect
merge: NMS-like drop duplicates (same program+pitch, onset within ε)
```

Defaults: `chunk_sec=4`, `hop_sec=2` (50% overlap) or hop=chunk for speed.

### 5.5 Device policy

```text
auto → cuda if feature+device ok → wgpu → cpu
```

Document expected RTF (realtime factor) on 1070 vs CPU in product README (measure in WP4.5).

---

## 6. GuitarSet light finetune

### 6.1 Goal

Improve **guitar-domain** transcription for deNote without destroying multi-instrument modular system.

### 6.2 Strategy (Job B-style / small)

| Option | Description | Prefer |
|--------|-------------|--------|
| **G-guit** | Add or map guitar to a group; finetune **one** decoder (or keys-like group) on GuitarSet | if guitar in vocab/programs |
| **Adapter FT** | Low LR Job A-style on guitar latents only + existing decoders freeze | if program id shared |
| **Single small head** | Tiny extra decoder `g_guitar` trained Job B-style | cleanest modular |

**Default plan:** introduce product program id for guitar (e.g. GM acoustic guitar 24/25 or dedicated id in product vocab) + **Job B-style train one decoder** or finetune closest group; keep other decoders frozen.

### 6.3 Pipeline

```text
1) Download GuitarSet (user) — document URL/license
2) Prep: wav + note annotations → tokens (+ optional L0 latents)
3) configs/product_guitarset_ft.toml
4) Train on 1070: small steps, freeze encoder+most modules
5) Export updated module dir models/denote_modular_v1_guitar/
6) Qualitative listen + optional note F1 on GuitarSet split
```

### 6.4 Compute envelope

- batch 1–2, latent cache if possible  
- max few hours–days on 1070, not Phase 3 scale  

---

## 7. Work packages

### WP4.0 — Gate Phase 3 artifacts (0.25–0.5 day)

- [ ] Select best ckpt paths (A + B_g)  
- [ ] Confirm Job E command still works  
- [ ] List missing weights (encoder)  

**Acceptance:** written `models/SOURCES.md` draft.

---

### WP4.1 — Module dir format + pack script (0.5–1 day)

- [ ] Spec in this plan + `models/README.md`  
- [ ] `tools/pack_modular_model.py` or `cargo run -p denote-train -- --job pack`  
- [ ] Copy Job A/B + groups + infer.toml  

**Acceptance:** E4.6 layout exists from Phase 3 runs.

---

### WP4.2 — `denote-infer` modular backend (2–3 days)

- [ ] Load encoder + adapter + presence + decoders from `--model-dir`  
- [ ] Wav load 24 kHz mono (existing `denote-audio`)  
- [ ] Mel → encoder → adapter → presence → AR groups → merge  
- [ ] Flags: threshold, chunk, device  
- [ ] Keep legacy CRNN path  

**Files:** `crates/denote-infer/src/main.rs`, maybe `modular_infer.rs`, `chunk.rs`, `midi_out.rs`.

**Acceptance:** E4.1–E4.5 on sample wav.

---

### WP4.3 — JSON + MIDI export (0.5–1 day)

- [ ] Serde schema `denote.notes.v1`  
- [ ] MIDI: one track per program or channel=program map  
- [ ] Unit test roundtrip JSON  

**Acceptance:** E4.2, E4.3.

---

### WP4.4 — Long audio stitch (0.5–1 day)

- [ ] Sliding windows  
- [ ] Dedup merge  
- [ ] Smoke 30–60 s file  

**Acceptance:** E4.4.

---

### WP4.5 — Performance notes + smoke CI-ish (0.5 day)

- [ ] Time one 10 s clip on 1070/CPU → document RTF  
- [ ] `scripts/product_smoke.ps1`: pack (if needed) + infer fixture  

---

### WP4.6 — GuitarSet finetune (1–3 days)

- [ ] Data prep docs + script  
- [ ] Config + train job  
- [ ] Export guitar-aware module dir  
- [ ] Short eval / listening notes  

**Acceptance:** E4.7, E4.8 soft.

---

### WP4.7 — Regression + product docs (0.5–1 day)

- [ ] Re-run Job E subset after packaging (same metrics ±ε)  
- [ ] `scripts/product_infer.md`  
- [ ] Root `README.md` section: Product infer + GuitarSet (minimal; don't rewrite whole README architecture)  
- [ ] deNote integration notes: JSON fields  

**Acceptance:** E4.9, E4.10.

---

### WP4.8 — Optional stretch

- [ ] Half precision infer if Burn backend stable  
- [ ] `--programs` filter force subset  
- [ ] Example consumer Rust snippet reading JSON  

---

## 8. File change map

| Path | Action |
|------|--------|
| `PLAN_Phase4_Product.md` | this file (root) |
| `crates/denote-infer/src/modular_infer.rs` | create |
| `crates/denote-infer/src/chunk_stitch.rs` | create |
| `crates/denote-infer/src/midi_write.rs` | create |
| `crates/denote-infer/src/main.rs` | dual path modular/legacy |
| `configs/infer_modular.toml` | create |
| `configs/product_guitarset_ft.toml` | create |
| `models/README.md` | create |
| `tools/pack_modular_model.*` | create |
| `scripts/product_infer.md` | create |
| `scripts/product_smoke.ps1` | create |
| `scripts/guitarset_ft.md` | create |
| `README.md` | product quickstart blurb |
| `models/denote_modular_v1/` | generated/packed (git-lfs or not-in-git) |

**Do not modify** body of `PLAN_Denote_Free_Modular.md`.

---

## 9. Implementation order

```text
1. WP4.0 gate — freeze which ckpts are "product v1"
2. WP4.1 pack layout
3. WP4.2 modular infer (critical path)
4. WP4.3 JSON/MIDI
5. WP4.4 long audio
6. WP4.5 smoke + RTF
7. WP4.7 docs + regression (can start docs earlier)
8. WP4.6 GuitarSet FT (parallelizable after infer works)
9. WP4.8 stretch
```

---

## 10. Test plan

| Test | How |
|------|-----|
| Load module dir missing decoder | clear error |
| Short wav → non-empty or empty notes JSON valid schema | fixture |
| MIDI file header valid | midifile parse or external tool |
| Chunk stitch no crash on 60 s silence | manual |
| Legacy CRNN still runs | `--legacy` smoke |
| GuitarSet FT 100 steps | 1070 |
| Job E regression delta | compare report.json |

---

## 11. Risks

| Risk | Mitigation |
|------|------------|
| Phase 3 F1 weak → product sounds bad | ship path anyway; label “research demo”; GuitarSet FT helps niche |
| Encoder weights large / license | document; optional download script; don't commit GB to git |
| Windows CUDA infer flaky | CPU path + wgpu; document |
| Legacy vs modular CLI confusion | subcommands or explicit `--model-dir` vs `--checkpoint` |
| Guitar program not in challenge K=8 | product vocab extension; separate module dir |
| Overlap chunk double notes | dedup ε; tune hop |

---

## 12. Handoff → Phase 5

Phase 5 (scale ~1B) expects:

1. Stable **module dir** format (so larger decoders drop in as `decoders/g*`)  
2. Working product infer (regression while scaling)  
3. Train recipes Job B still valid for wider/deeper `decoder_g`  
4. No need to redesign CLI when params grow  

Phase 5 does **not** replace product path; it scales modules behind same interface.

---

## 13. Checklist

- [ ] WP4.0 gate Phase 3 artifacts  
- [ ] WP4.1 module dir + pack  
- [ ] WP4.2 modular denote-infer  
- [ ] WP4.3 JSON + MIDI  
- [ ] WP4.4 chunk stitch  
- [ ] WP4.5 smoke + RTF notes  
- [ ] WP4.6 GuitarSet light FT  
- [ ] WP4.7 regression + docs  
- [ ] E4.1–E4.10  

---

## 14. Definition of done (кратко)

> Пользователь (или deNote) указывает `--model-dir` и wav → получает JSON/MIDI ноты через modular presence+multi-decoder+FSM; есть упакованный bundle; опционально GuitarSet finetune на 1070; README/scripts для demo. Master-план не изменялся.

---

*Phase 4 = shippable infer + domain FT for product. Challenge SOTA already Phase 3; Phase 5 only scales capacity.*