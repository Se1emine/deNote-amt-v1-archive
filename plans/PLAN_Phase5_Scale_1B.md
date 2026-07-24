# Phase 5 — Scale system → ~1B (endgame)

**Master-план:** [`PLAN_Denote_Free_Modular.md`](PLAN_Denote_Free_Modular.md)  
(ссылки: North Star Q2 ~1B · C1/C5 modular scale · §2.5 budgets · §7 Phase 5 steps 5.1–5.6 · honest param accounting · C7 arch ablations)

**Зависимости:**

| Фаза | Документ | Что должно быть готово |
|------|----------|------------------------|
| Phase 0–2 | PLAN_Phase0…2 | platform, cache, modular Job A/B, infer merge |
| Phase 3 | [`PLAN_Phase3_Data_F1_vs_MIROS.md`](PLAN_Phase3_Data_F1_vs_MIROS.md) | data + Job E multi-onset F1 protocol |
| Phase 4 | [`PLAN_Phase4_Product.md`](PLAN_Phase4_Product.md) | **module dir** format, product `denote-infer`, regression path |

**Цель фазы:**

1. Довести **system size** (frozen MusicFM + adapter + presence + all decoders [+ optional MoE/LoRA]) до порядка **~0.9–1.1B параметров**.  
2. Обучать **по-прежнему кусками** (C1/C5): один trainable submodule / job ≤ ~40–60M на GTX 1070 когда возможно; иначе Kaggle overflow / skip dense joint.  
3. **Не** отождествлять 1B с одним AdamW job на 8 GB — master явно: *1B ≠ один train job*.  
4. Сохранить product interface Phase 4 (`models/denote_modular_v*`).  
5. Thesis accounting: `N_total` vs `N_trainable_per_job`; ablation modular-arch @ ~MIROS budget vs scaled system.  
6. Best-effort quality: Job E F1 не деградирует; ideally improves vs Phase 3 checkpoint.

**Не цель фазы:** обязательный paid multi-GPU cluster; classic Switch-MoE LLM router as core; rewrite stack in PyTorch; гарантированный F1 > 0.5998 solely from scale.

---

## 1. Контекст master-плана

### 1.1 North Star (Q2)

> ~1B как **endgame**, не первый train job.  
> v1 modular free ~150–350M system; Phase 5 progressive: wider/deeper decoders, more groups, optional MoE-FFN, LoRA on encoder → **system ~0.9–1.1B**.

### 1.2 Constraints

| ID | Implication for Phase 5 |
|----|-------------------------|
| C1 | Primary 1070 8 GB — each **job** must fit or move to Kaggle overflow |
| C5 | Scale = more/wider modules, **not** one 1B dense FT on 8 GB |
| C7 | Ablation at matched ~MIROS budget before “1B won because bigger” |
| C2 | Kaggle only overflow (large decoder job, joint polish) |

### 1.3 Master steps 5.1–5.6 (verbatim roadmap)

| Step | What | Params | Compute |
|------|------|--------|---------|
| **5.1** | Widen/deepen each `decoder_g` (one job at a time) | +per module | 1070 if ≤60M trainable; else Kaggle |
| **5.2** | G=8 per-instrument decoders | +system capacity | sequential 1070 |
| **5.3** | Optional MoE-FFN **inside** a decoder | +capacity, sparse | 1070/Kaggle by size |
| **5.4** | LoRA / last-N encoder blocks FT | +quality | 1070 latent or Kaggle |
| **5.5** | Count **system params** → target ~1B | reporting | infer assemble |
| **5.6** | Optional joint polish if 24GB+ ever | quality | overflow / future |

### 1.4 Honest reporting (master)

```text
N_total            = encoder + adapter + presence + Σ decoders [+ MoE experts] [+ LoRA]
N_trainable_per_job = params with grad in that job only
Ablation: modular @ ~370M-class vs scaled ~1B system on same Job E val
```

### 1.5 How to “get to ~1B” without dense 1B train

Illustrative budget (adjust to real MusicFM size when measured):

```text
MusicFM encoder (frozen)     ~100–300M   (measure in WP5.0)
Adapter + presence           ~10–50M
Decoders G=4 base            ~4 × 30–80M
Widen/deepen (5.1)           → 4 × 80–150M
G=8 (5.2)                    → 8 × 60–100M
MoE-FFN experts (5.3)        +100–300M capacity (active FLOPs << dense)
LoRA encoder (5.4)           +1–20M trainable
────────────────────────────────────────
N_total                      target 0.9–1.1B
```

If encoder alone is smaller, **compensate** with deeper decoders / more experts / G=8 — document actual numbers in `reports/param_count_phase5.json`.

---

## 2. Baseline after Phase 4

| Artifact | Role in Phase 5 |
|----------|-----------------|
| `models/denote_modular_v1/` | Template; spawn `v1_scale` / `v1_1b` |
| Job B recipe | Same CLI; larger `TokenDecoderConfig` |
| Job E | Regression gate after each scale step |
| Product infer | Must load larger modules without API break |
| Phase 3 data cache | Reuse; optional more data if F1 stalls |

---

## 3. Exit criteria (Definition of Done Phase 5)

| ID | Критерий | Проверка |
|----|----------|----------|
| E5.0 | Phase 4 modular product path works (or Phase 2–3 modular infer minimum) | smoke infer |
| E5.1 | Automated **param counter** for module dir / in-memory model | `reports/param_count_*.json` |
| E5.2 | Step 5.1: at least one group decoder **scaled** (depth/width ↑) and trained | ckpt + train log |
| E5.3 | Step 5.2 **or** equivalent capacity: G=8 **or** G=4 with width that reaches capacity plan | configs + ckpts |
| E5.4 | `N_total` in range **≥ 0.85B** (hard) or documented shortfall + path to 1B (soft waiver) | param report |
| E5.5 | Soft target `N_total ∈ [0.9B, 1.1B]` | param report |
| E5.6 | Each train job documents `N_trainable_per_job` and peak VRAM / OOM knobs | run README |
| E5.7 | Job E: scaled system F1 ≥ Phase 3 baseline − ε (no silent collapse); ideally > baseline | eval reports |
| E5.8 | Ablation table: **small/matched** modular vs **scaled** system (same val) | `runs/ablation_phase5.md` |
| E5.9 | Product module dir `models/denote_modular_v1_1b/` (or v2) packs and infers | CLI |
| E5.10 | Thesis section draft bullets: accounting + compute honesty | `docs` or `reports/thesis_scale_notes.md` |
| E5.11 | Optional 5.3 MoE and/or 5.4 LoRA: at least **one** attempted or explicitly deferred with reason | report |
| E5.12 | Step 5.6 joint polish: done **or** N/A (no 24GB) documented | report |

**Hard success:** E5.1, E5.2, E5.6, E5.7, E5.8, E5.9 + (E5.4 or honest waiver).  
**Soft success:** E5.5 (true ~1B band) + quality ↑ on Job E.  
**Stretch:** MoE+LoRA+ joint polish; F1 > MIROS 0.5998 if not already.

---

## 4. Scope / Non-scope

### In scope

1. Param counting utilities  
2. Larger `TokenDecoderConfig` grid (layers, d_model, d_ff, heads)  
3. Progressive init: grow from Phase 3/4 weights when shapes allow (or train from scratch per group)  
4. G=8 group configs + token filters  
5. Optional MoE-FFN inside decoder block (sparse top-2 or expert choice by program — **not** full LLM router required)  
6. Optional LoRA on last N encoder layers / adapter  
7. Sequential Job B (and light Job A) on 1070 / Kaggle  
8. Pack scaled module dir + product infer regression  
9. Ablations + thesis accounting  
10. VRAM profiling per job  

### Out of scope

- Single-job full-model AdamW 1B on 8 GB  
- Mandatory multi-node distributed Burn cluster  
- Replacing MusicFM with new SSL pretrain from scratch  
- Changing challenge metric  
- Product GUI  

---

## 5. Scaling design

### 5.1 Decoder width/depth ladder (step 5.1)

| Config tag | n_layers | d_model | n_heads | d_ff | ~params/decoder (order) |
|------------|----------|---------|---------|------|-------------------------|
| `dec_S` | 2–4 | 512 | 8 | 2048 | baseline Phase 2–3 |
| `dec_M` | 6–8 | 512–768 | 8–12 | 2048–3072 | mid |
| `dec_L` | 10–12 | 768–1024 | 12–16 | 3072–4096 | large; may exceed 1070 |

**Train rule:** one `dec_*` job per group; freeze adapter/presence/encoder.  
**Init:** if d_model matches Job A memory — reuse adapter; if d_model grows, need adapter projection resize + short Job A refresh (document).

**Prefer:** keep `d_model` fixed to Job A / encoder width; scale **layers and d_ff** first (fewer breaking changes).

### 5.2 G=8 per-instrument (step 5.2)

```text
G0 Piano, G1 Violin, G2 Viola, G3 Cello,
G4 Flute, G5 Oboe, G6 Bassoon, G7 Trombone
```

- Update `groups.toml`  
- Rebuild `tokens_g` for 8 groups  
- 8 sequential Job B runs (calendar time!)  
- Infer: 8 decoders, presence still K=8  

Capacity: more modules × similar size ≈ 2× decoder params vs G=4.

### 5.3 MoE-FFN inside decoder (step 5.3, optional)

```text
Replace dense FFN in some decoder layers with E experts, top-k=1 or 2
Train still one decoder_g job (all experts of that decoder)
N_total counts all experts; FLOPs ≈ top-k / E of dense
```

**1070 caution:** MoE can increase param memory even if FLOPs sparse — expert count limited (E=4–8).  
If unstable in Burn — **defer** with note (E5.11).

### 5.4 LoRA / last-N encoder (step 5.4, optional)

```text
Freeze MusicFM base; inject LoRA on last N Conformer blocks
OR train last N blocks full with tiny LR + latent not used (needs wav path)
Prefer: LoRA + occasional wav forward on Kaggle if 1070 OOM
N_trainable small; N_total almost unchanged unless counting full encoder (always count encoder in N_total)
```

### 5.5 Param accounting tool (step 5.5)

```powershell
cargo run -p denote-train -- --job count-params --model-dir models/denote_modular_v1_1b
# or rust unit that sums Module parameter dims
```

Output JSON:

```json
{
  "encoder": 0,
  "adapter": 0,
  "presence": 0,
  "decoders": {"g0": 0, "g1": 0},
  "moe_experts_extra": 0,
  "lora": 0,
  "N_total": 0,
  "notes": "encoder frozen; trained jobs listed separately"
}
```

Plus per-job files under `runs/jobB_g0_scale/trainable_params.json`.

### 5.6 Joint polish (step 5.6)

```text
If GPU ≥ 24 GB available someday:
  unfreeze adapter + all decoders, tiny LR, short steps
Else:
  mark N/A — product remains stitch of modular ckpts
```

---

## 6. Train / eval protocol while scaling

### 6.1 Regression gate

After **each** meaningful scale step (new decoder size or G=8 pack):

1. Pack module dir  
2. `denote-infer` smoke  
3. Job E on **fixed** Phase 3 val manifest  
4. Compare F1 to `baseline_phase3.json`  

**Fail gate:** F1 drop > absolute 0.05 without explanation → debug before more scale.

### 6.2 Data

- Default: **same** Phase 3 cache (fair ablation scale vs arch)  
- Optional: more FluidR3 if capacity underfits  
- Curriculum can stay Phase 3 settings  

### 6.3 Compute schedule (example calendar)

```text
Week 1:   param counter + dec_M for g0 only (prove 1070 fit)
Week 2–3: dec_M/L for all G=4 groups sequential
Week 4–5: G=8 migration OR deeper dec_L
Week 6:   MoE or LoRA optional; pack 1B; ablations; thesis notes
Overflow: Kaggle for any job with N_trainable > ~60M or OOM
```

---

## 7. Work packages

### WP5.0 — Gate + measure baseline params (0.5 day)

- [ ] Load Phase 4/3 module dir  
- [ ] Implement param counter  
- [ ] Record `N_total_baseline`, VRAM infer  

**Acceptance:** E5.0, start of E5.1.

---

### WP5.1 — Param counter hardened (0.5 day)

- [ ] CLI + JSON schema  
- [ ] Count from disk module dir without full train  
- [ ] Document formula in `scripts/scale_1b.md`  

**Acceptance:** E5.1.

---

### WP5.2 — Decoder ladder configs (0.5 day)

- [ ] `configs/scale/dec_S.toml`, `dec_M.toml`, `dec_L.toml`  
- [ ] `free_1070_jobB_g*_scale_M.toml`  
- [ ] Estimated params table filled with counter  

---

### WP5.3 — Step 5.1 train scaled decoders (days–weeks wall)

- [ ] For each g: train `dec_M` (then optional `dec_L`)  
- [ ] Resume, JSONL, 1070 OOM knobs  
- [ ] Job E after all groups updated  

**Acceptance:** E5.2, E5.6, partial E5.7.

---

### WP5.4 — Step 5.2 G=8 (optional path to capacity) (1–2 weeks wall)

- [ ] groups_challenge8_per_instrument.toml  
- [ ] rebuild tokens_g  
- [ ] 8× Job B (can reuse weights init from parent group decoder if mapping)  
- [ ] Infer + Job E  

**Acceptance:** E5.3 if this path chosen.

---

### WP5.5 — Step 5.3 MoE spike (1–3 days, optional)

- [ ] Design MoE FFN in Burn decoder layer  
- [ ] Single group prototype  
- [ ] Param count + train smoke  
- [ ] Go/no-go for all groups  

**Acceptance:** E5.11.

---

### WP5.6 — Step 5.4 LoRA encoder (1–3 days, optional)

- [ ] LoRA modules on last N blocks  
- [ ] Train job with wav or cached intermediate features  
- [ ] Merge/export for module dir  

**Acceptance:** E5.11.

---

### WP5.7 — Reach ~1B accounting + pack (1–2 days)

- [ ] Combine modules until E5.4/E5.5  
- [ ] If short: increase experts/layers with documented plan  
- [ ] Pack `models/denote_modular_v1_1b/`  
- [ ] Product infer smoke  

**Acceptance:** E5.4/5, E5.9.

---

### WP5.8 — Ablations + thesis notes (1 day)

- [ ] Matched-budget modular (Phase 3 size) vs scaled  
- [ ] Table F1, N_total, train GPU-hours  
- [ ] `runs/ablation_phase5.md` + `reports/thesis_scale_notes.md`  

**Acceptance:** E5.8, E5.10.

---

### WP5.9 — Step 5.6 joint polish decision (0.25 day)

- [ ] If no 24GB: write N/A  
- [ ] Else short joint run + Job E  

**Acceptance:** E5.12.

---

### WP5.10 — Docs (0.5 day)

- [ ] `scripts/scale_1b.md` — ladder, VRAM, Kaggle overflow, accounting  
- [ ] Link from product README “large model”  
- [ ] **Do not edit** `PLAN_Denote_Free_Modular.md` body  

---

## 8. File change map

| Path | Action |
|------|--------|
| `PLAN_Phase5_Scale_1B.md` | this file (root) |
| `crates/denote-model` | larger decoder configs; optional MoE/LoRA |
| `crates/denote-train` | `--job count-params`; scale job configs |
| `configs/scale/*.toml` | create |
| `configs/groups_g8_per_instrument.toml` | create if 5.2 |
| `scripts/scale_1b.md` | create |
| `reports/param_count_*.json` | generated |
| `reports/thesis_scale_notes.md` | create |
| `runs/ablation_phase5.md` | generated |
| `models/denote_modular_v1_1b/` | packed |

---

## 9. Implementation order

```text
1. WP5.0–5.1  measure + counter
2. WP5.2      configs ladder
3. WP5.3      scale decoders G=4 (main quality path)
4. Re-eval Job E gate
5. WP5.4 and/or WP5.5/5.6 until N_total ~1B
6. WP5.7      pack product dir
7. WP5.8–5.10 ablations, thesis, docs, 5.6 decision
```

**Critical path:** counter → dec_M all groups → Job E → pack.  
**MoE/LoRA/G=8:** use as capacity fillers if N_total short or quality plateaus.

---

## 10. 1070 envelope (scale jobs)

| Situation | Action |
|-----------|--------|
| N_trainable ≤ ~40–60M | 1070 Job B default |
| OOM on dec_L | reduce batch, seq len, layers; gradient ckpt if available |
| Still OOM | Kaggle overflow that one group |
| Encoder LoRA + full wav | prefer Kaggle or latent of intermediate layer |
| Joint all decoders | not on 1070 — step 5.6 only with 24GB+ |

Latent cache remains default for decoder-only jobs.

---

## 11. Test plan

| Test | How |
|------|-----|
| Param counter matches hand sum on tiny module | unit |
| Load dec_M ckpt in infer | smoke |
| Job E baseline vs scaled JSON diff | script |
| G=8 program routing | unit groups |
| MoE forward shape | unit if implemented |
| Module dir pack roundtrip | manual |

---

## 12. Risks

| Risk | Mitigation |
|------|------------|
| 1B vanity without F1 gain | Job E gate; ablation C7; don't claim SOTA from N_total alone |
| d_model change breaks adapter | scale depth/ffn first; freeze d_model |
| 8× Job B calendar explosion | train priority instruments first; init from group parent |
| MoE unsupported/slow in Burn | defer E5.11 |
| Disk for 8 large decoders | external drive; don't git-commit weights |
| Overfit large decoders on same data | more Phase 3 data; dropout; early stop on val F1 |
| Infer too slow for product | chunk; smaller active groups via presence; optional distill later (out of scope) |

---

## 13. Handoff (project complete for scale story)

After Phase 5:

1. Diploma can report **modular free training** + **system ~1B** with honest `N_trainable_per_job`  
2. Product path still Phase 4 CLI  
3. Best quality ckpt = Phase 5 pack if Job E wins; else keep Phase 3 as “quality” and Phase 5 as “scale demonstration”  
4. No further master phases required  

---

## 14. Checklist

- [ ] WP5.0 gate + baseline params  
- [ ] WP5.1 param counter  
- [ ] WP5.2 decoder ladder configs  
- [ ] WP5.3 step 5.1 train scaled decoders  
- [ ] WP5.4 step 5.2 G=8 (if needed)  
- [ ] WP5.5 step 5.3 MoE (optional)  
- [ ] WP5.6 step 5.4 LoRA (optional)  
- [ ] WP5.7 N_total ~1B + pack  
- [ ] WP5.8 ablations + thesis notes  
- [ ] WP5.9 step 5.6 decision  
- [ ] WP5.10 docs  
- [ ] E5.1–E5.12  

---

## 15. Definition of done (кратко)

> System params **~1B** (or documented near-miss) as **sum of modular pieces**; each piece trained within free/1070 (or noted overflow); product module dir infers; Job E не развален; есть честный accounting и ablation scale vs smaller modular. Master-план не изменялся.

---

*Phase 5 = capacity endgame under modular free constraints. Architecture wins were Phase 1–3; product was Phase 4; here we scale without abandoning C1/C5.*