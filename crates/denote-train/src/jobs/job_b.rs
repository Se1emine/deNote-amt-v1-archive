//! Job B_g: train one modular decoder on latent cache (adapter/presence frozen).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use burn::{
    backend::{Autodiff, NdArray},
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{backend::AutodiffBackend, backend::Backend, ElementConversion, Int, Tensor},
};
use log::info;
use serde_json::json;

use crate::device::{log_backend, resolve_device, DeviceKind};
use crate::metrics_log::{
    copy_config, default_run_dir, log_run_banner, prepare_run_dir, run_dir_from_resume,
    write_hparams, RunLogger, TrainState,
};
use crate::phase2_config::JobBConfig;
use denote_data::{
    encode_group_tokens, filter_curriculum, group_has_notes, load_notes_for_entry,
    mean_n_instruments, CacheManifest, LatentFeatureCache, ManifestEntry, TokenFeatureCache,
};
use denote_model::{
    token_ce_loss, AdapterConfig, DenoteModular, DenoteModularConfig, GroupsConfig,
    PresenceHeadConfig, TokenDecoderConfig,
};
use denote_tokens::{pad_tokens, Vocab};

#[derive(Debug, Clone)]
pub struct JobBOpts {
    pub config: PathBuf,
    pub run_dir: Option<PathBuf>,
    pub resume: Option<PathBuf>,
    pub device: String,
    pub max_steps: Option<u64>,
    pub log_every: Option<u64>,
    pub ckpt_every: Option<u64>,
    pub batch_size: Option<usize>,
    /// Override group id from CLI.
    pub group_id: Option<u32>,
}

pub fn run_job_b(opts: JobBOpts) -> Result<()> {
    let kind = resolve_device(&opts.device)?;
    log_backend(kind, "Phase2 Job B");

    match kind {
        DeviceKind::Cpu => run_job_b_backend::<Autodiff<NdArray>>(opts, kind),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => run_job_b_backend::<Autodiff<burn::backend::Cuda>>(opts, kind),
        #[cfg(feature = "wgpu")]
        DeviceKind::Wgpu => run_job_b_backend::<Autodiff<burn::backend::Wgpu>>(opts, kind),
        #[cfg(not(feature = "cuda"))]
        DeviceKind::Cuda => unreachable!(),
        #[cfg(not(feature = "wgpu"))]
        DeviceKind::Wgpu => unreachable!(),
    }
}

fn run_job_b_backend<B>(opts: JobBOpts, kind: DeviceKind) -> Result<()>
where
    B: AutodiffBackend,
{
    let cfg = JobBConfig::load(&opts.config)?;
    let device = B::Device::default();
    let vocab = cfg.vocab();
    let pad_id = vocab.pad as usize;

    let groups = load_groups(&cfg.groups.path)?;
    let group_id = opts.group_id.unwrap_or(cfg.data.group_id);
    let single = cfg.model.single_decoder;
    if !single && groups.group(group_id).is_none() {
        bail!("group_id={group_id} not found in groups config");
    }
    let train_g = if single { 0u32 } else { group_id };
    let g_idx = train_g as usize;

    let run_dir = resolve_run_dir(&opts, &cfg, train_g, single)?;
    prepare_run_dir(&run_dir)?;
    if opts.config.exists() {
        let _ = copy_config(&opts.config, &run_dir);
    }

    let max_steps = opts.max_steps.unwrap_or(cfg.optim.max_steps);
    let log_every = opts.log_every.unwrap_or(cfg.log.log_every).max(1);
    let ckpt_every = opts.ckpt_every.unwrap_or(cfg.optim.ckpt_every);
    let batch_size = opts.batch_size.unwrap_or(cfg.data.batch_size).max(1);
    let val_every = cfg.log.val_every.max(1);
    let max_val_batches = cfg.log.max_val_batches;
    let lr = cfg.optim.lr;
    let k = cfg.model.presence_k.max(1);
    let max_token_len = cfg.data.max_token_len.max(8);
    let mut d_model = cfg.model.d_model;

    write_hparams(
        &run_dir,
        &json!({
            "phase": "phase2_job_b",
            "job": if single { "job-b-single" } else { "job-b" },
            "device": kind.as_str(),
            "group_id": train_g,
            "single_decoder": single,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "lr": lr,
            "d_model": d_model,
            "presence_k": k,
            "decoder_layers": cfg.model.decoder_layers,
            "freeze_adapter": cfg.model.freeze_adapter,
            "freeze_presence": cfg.model.freeze_presence,
            "trainable": if single {
                vec!["decoder_0".to_string()]
            } else {
                vec![format!("decoder_g{train_g}")]
            },
            "job_a_ckpt": cfg.run.job_a_ckpt,
            "latent_manifest": cfg.data.latent_manifest,
            "max_token_len": max_token_len,
        }),
    )?;
    log_run_banner(&run_dir, kind.as_str());

    let manifest = CacheManifest::from_path(&cfg.data.latent_manifest);
    let cache_root = manifest.root.clone();
    let all_train = manifest.list_by_split("train")?;
    let all_val = manifest.list_by_split("val")?;
    if all_train.is_empty() {
        bail!(
            "no train entries in {} — run Job L0 first",
            cfg.data.latent_manifest
        );
    }

    let latent_cache = LatentFeatureCache::new(cache_root.join("latents"));
    let token_cache = TokenFeatureCache::new(cache_root.join("tokens"));

    // Probe d_model from first latent.
    let probe = latent_cache
        .load(&all_train[0].id)
        .with_context(|| format!("load latent {}", all_train[0].id))?;
    if probe.meta.d_model != d_model {
        info!(
            "d_model config={} but latent has {}; using latent d_model",
            d_model, probe.meta.d_model
        );
        d_model = probe.meta.d_model;
    }

    // Filter samples with group notes (or full tokens for single decoder).
    let group_programs: Vec<u8> = if single {
        (0..k as u8).collect()
    } else {
        groups.programs_of(group_id).to_vec()
    };

    let train_entries = filter_entries_for_group(&all_train, &group_programs, single, &token_cache)?;
    let val_entries = filter_entries_for_group(&all_val, &group_programs, single, &token_cache)?;
    if train_entries.is_empty() {
        bail!(
            "no train samples with notes for group {} (programs={:?})",
            train_g,
            group_programs
        );
    }
    let frac_train = train_entries.len() as f32 / all_train.len() as f32;
    info!(
        "Job B data | group={} single={} train_with_group={}/{} ({:.1}%) val={}/{} batch={}",
        train_g,
        single,
        train_entries.len(),
        all_train.len(),
        frac_train * 100.0,
        val_entries.len(),
        all_val.len(),
        batch_size
    );

    let vocab_size = vocab.size as usize;
    let n_heads = cfg.model.decoder_heads.max(1);
    let d_ff = d_model * cfg.model.decoder_ff_mult.max(1);
    let n_groups_model = if single { 1 } else { groups.n_groups().max(1) };

    let model_cfg = DenoteModularConfig {
        adapter: AdapterConfig::new(d_model, k)
            .with_n_pitches_aux(128)
            .with_dropout(0.0),
        presence: PresenceHeadConfig::new(d_model, k).with_hidden(d_model),
        decoder: TokenDecoderConfig::new(
            vocab_size,
            d_model,
            cfg.model.decoder_layers.max(1),
            n_heads,
            d_ff,
        )
        .with_dropout(cfg.model.decoder_dropout),
        n_groups: n_groups_model,
    };
    let mut model: DenoteModular<B> = model_cfg.init(&device);

    // Load Job A (required for real train; random ok only if path missing for compile smoke).
    let job_a = PathBuf::from(&cfg.run.job_a_ckpt);
    if job_a_has_modules(&job_a) {
        info!("Loading Job A from {}", job_a.display());
        model
            .load_job_a(&job_a, &device)
            .map_err(|e| anyhow::anyhow!(e))?;
    } else {
        info!(
            "Job A ckpt not found at {} — training decoder with random adapter (smoke only)",
            job_a.display()
        );
    }

    // Resume decoder weights if requested.
    let ckpt_dir = run_dir.join("checkpoints");
    let state_path = ckpt_dir.join("state.json");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let mut start_step: u64 = 0;

    if let Some(ref resume) = opts.resume {
        let base = normalize_ckpt_dir(resume);
        let dec_stem = decoder_stem(&base, train_g, single);
        if path_mpk_exists(&dec_stem) {
            info!("Resume decoder from {}", dec_stem.display());
            model
                .load_decoder_g(g_idx, &dec_stem, &device)
                .map_err(|e| anyhow::anyhow!(e))?;
        } else {
            let last = base.join("last");
            let dec_last = decoder_stem(&last, train_g, single);
            if path_mpk_exists(&dec_last) {
                info!("Resume decoder from {}", dec_last.display());
                model
                    .load_decoder_g(g_idx, &dec_last, &device)
                    .map_err(|e| anyhow::anyhow!(e))?;
            }
        }
        if state_path.exists() {
            let st = TrainState::load(&state_path)?;
            start_step = st.step;
            info!("Resume state step={start_step}");
        }
    }

    // Optim only on decoder_g (not adapter/presence).
    let mut optim = AdamWConfig::new()
        .with_weight_decay(cfg.optim.weight_decay as f32)
        .init::<B, denote_model::TokenDecoder<B>>();

    let trainable_name = if single {
        "decoder_0".to_string()
    } else {
        format!("decoder_g{train_g}")
    };
    info!(
        "DenoteModular | n_groups={} d_model={} vocab={} | trainable ONLY {trainable_name} | adapter/presence frozen={}",
        model.n_groups(),
        d_model,
        vocab_size,
        cfg.model.freeze_adapter
    );
    info!(
        "Trainable module path: decoders[{g_idx}] (params≈{})",
        model.decoder(g_idx).num_params()
    );

    let mut logger = RunLogger::new(&run_dir)?;
    let end_step = start_step + max_steps;
    let mut step = start_step;
    let mut train_cursor = 0usize;
    let detach_adapter = cfg.model.freeze_adapter;
    let curriculum_on = cfg.curriculum.enabled;
    let curriculum_sched = cfg.curriculum.schedule();
    if curriculum_on {
        info!(
            "Curriculum ON | stage1_end={} stage2_end={}",
            curriculum_sched.stage1_end, curriculum_sched.stage2_end
        );
    }

    // Log frac once
    logger.log_scalars(
        step.max(1),
        "data",
        &[("frac_samples_with_group", frac_train)],
    )?;

    info!(
        "Job B loop steps {}..={} log_every={log_every} val_every={val_every} \
         max_val_batches={max_val_batches} ckpt_every={ckpt_every}",
        start_step + 1,
        end_step
    );

    while step < end_step {
        step += 1;
        let t0 = Instant::now();

        let batch_entries = if curriculum_on {
            let eligible = filter_curriculum(&train_entries, &curriculum_sched, step);
            if eligible.is_empty() {
                bail!("curriculum filtered all train samples at step {step}");
            }
            let mut out = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                out.push(eligible[train_cursor % eligible.len()]);
                train_cursor = train_cursor.wrapping_add(1);
            }
            out
        } else {
            next_batch(&train_entries, &mut train_cursor, batch_size)
        };
        let n_inst_mean = mean_n_instruments(&batch_entries);
        let batch = load_group_batch::<B>(
            &latent_cache,
            &token_cache,
            &batch_entries,
            &group_programs,
            single,
            k,
            max_token_len,
            &vocab,
            &groups,
            train_g,
            &device,
        )?;

        let out = model.forward_group_tf(
            g_idx,
            batch.z,
            batch.tokens_in,
            batch.multi_hot,
            detach_adapter,
        );
        let loss = token_ce_loss(
            out.token_logits,
            batch.tokens_tgt,
            pad_id,
            cfg.optim.label_smoothing,
            &device,
        );
        let loss_val: f32 = loss.clone().into_scalar().elem();
        if !loss_val.is_finite() {
            bail!("non-finite loss_ce at step {step}: {loss_val}");
        }

        let grads = loss.backward();
        // Gradients only for decoder_g — adapter detached so no adapter grads.
        let grads = GradientsParams::from_grads(grads, model.decoder(g_idx));
        let updated = optim.step(lr, model.decoders[g_idx].clone(), grads);
        model.decoders[g_idx] = updated;

        let step_time_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let samples_per_sec = batch_size as f32 / (step_time_ms / 1000.0).max(1e-6);

        if step % log_every == 0 || step == end_step || step == start_step + 1 {
            info!(
                "step {step}/{end_step} loss_ce={loss_val:.4} lr={lr:.2e} step_time_ms={step_time_ms:.1} trainable={trainable_name}"
            );
            logger.log_scalars(
                step,
                "train",
                &[
                    ("loss_ce", loss_val),
                    ("lr", lr as f32),
                    ("step_time_ms", step_time_ms),
                    ("samples_per_sec", samples_per_sec),
                    ("group_id", train_g as f32),
                    ("data/n_instruments_mean", n_inst_mean),
                    (
                        "data/curriculum_stage",
                        if curriculum_on {
                            curriculum_sched.stage_id(step) as f32
                        } else {
                            0.0
                        },
                    ),
                ],
            )?;
            logger.flush()?;
        }

        if !val_entries.is_empty() && (step % val_every == 0 || step == end_step) {
            let t_val = Instant::now();
            let (v_loss, n_val_batches, n_val_samples) = eval_ce::<B>(
                &model,
                g_idx,
                &latent_cache,
                &token_cache,
                &val_entries,
                &group_programs,
                single,
                k,
                max_token_len,
                &vocab,
                &groups,
                train_g,
                pad_id,
                cfg.optim.label_smoothing,
                batch_size,
                max_val_batches,
                detach_adapter,
                &device,
            )?;
            info!(
                "val step={step} loss_ce={v_loss:.4} batches={n_val_batches} samples={n_val_samples} \
                 val_ms={:.1}",
                t_val.elapsed().as_secs_f32() * 1000.0
            );
            logger.log_scalars(
                step,
                "val",
                &[
                    ("loss_ce", v_loss),
                    ("val_batches", n_val_batches as f32),
                    ("val_samples", n_val_samples as f32),
                ],
            )?;
            logger.flush()?;
        }

        if ckpt_every > 0 && step % ckpt_every == 0 && step != end_step {
            save_job_b_checkpoint(
                &model,
                g_idx,
                train_g,
                single,
                &ckpt_dir,
                step,
                lr as f32,
                kind.as_str(),
                d_model,
                k,
                &trainable_name,
            )?;
        }
    }

    save_job_b_checkpoint(
        &model,
        g_idx,
        train_g,
        single,
        &ckpt_dir,
        step,
        lr as f32,
        kind.as_str(),
        d_model,
        k,
        &trainable_name,
    )?;
    // Keep a frozen copy of Job A modules for infer convenience.
    let last_dir = ckpt_dir.join("last");
    let _ = model.save_job_a_modules(&last_dir);

    logger.flush()?;
    info!(
        "Job B done | group={} final_step={step} metrics={} decoder_ckpt={}",
        train_g,
        run_dir.join("metrics.jsonl").display(),
        last_dir.display()
    );
    info!(
        "TensorBoard: python tools/jsonl_to_tb.py --jsonl {} --out {}",
        run_dir.join("metrics.jsonl").display(),
        run_dir.join("tb").display()
    );
    let _ = recorder; // silence if unused
    Ok(())
}

struct GroupBatch<B: Backend> {
    z: Tensor<B, 3>,
    multi_hot: Tensor<B, 2>,
    tokens_in: Tensor<B, 2, Int>,
    tokens_tgt: Tensor<B, 2, Int>,
}

fn next_batch<'a>(
    entries: &'a [ManifestEntry],
    cursor: &mut usize,
    batch_size: usize,
) -> Vec<&'a ManifestEntry> {
    let n = entries.len();
    let mut out = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        out.push(&entries[*cursor % n]);
        *cursor = (*cursor + 1) % n;
    }
    out
}

fn filter_entries_for_group(
    entries: &[ManifestEntry],
    group_programs: &[u8],
    single: bool,
    token_cache: &TokenFeatureCache,
) -> Result<Vec<ManifestEntry>> {
    let mut out = Vec::new();
    for e in entries {
        if single {
            // Prefer full token stream; skip only if empty sequence.
            if let Ok(tok) = token_cache.load(&e.id) {
                if tok.token_ids.len() > 2 {
                    out.push(e.clone());
                    continue;
                }
            }
            // Fallback: notes exist
            if let Ok(notes) = load_notes_for_entry(
                e.notes_path.as_deref(),
                e.audio_path.as_deref(),
                &e.id,
            ) {
                if !notes.notes.is_empty() {
                    out.push(e.clone());
                }
            }
            continue;
        }
        // Group filter: programs overlap OR notes in group.
        if e.programs.iter().any(|p| group_programs.contains(p)) {
            // Verify notes if available; keep if programs overlap even without notes load.
            if let Ok(notes) = load_notes_for_entry(
                e.notes_path.as_deref(),
                e.audio_path.as_deref(),
                &e.id,
            ) {
                if group_has_notes(&notes.notes, group_programs) {
                    out.push(e.clone());
                }
            } else {
                out.push(e.clone());
            }
        }
    }
    Ok(out)
}

fn load_group_batch<B: Backend>(
    latent_cache: &LatentFeatureCache,
    token_cache: &TokenFeatureCache,
    entries: &[&ManifestEntry],
    group_programs: &[u8],
    single: bool,
    k: usize,
    max_token_len: usize,
    vocab: &Vocab,
    groups: &GroupsConfig,
    group_id: u32,
    device: &B::Device,
) -> Result<GroupBatch<B>> {
    let b = entries.len();
    let mut samples = Vec::with_capacity(b);
    let mut max_t = 1usize;
    let mut d = 0usize;
    let mut token_seqs: Vec<Vec<u32>> = Vec::with_capacity(b);

    for e in entries {
        let s = latent_cache
            .load(&e.id)
            .with_context(|| format!("load latent {}", e.id))?;
        max_t = max_t.max(s.meta.n_frames.max(1));
        d = s.meta.d_model;

        let ids = if single {
            load_full_tokens(e, token_cache, vocab)?
        } else {
            load_group_tokens(e, group_programs, vocab)?
        };
        token_seqs.push(ids);
        samples.push((s, e));
    }

    // Teacher forcing: tokens_in = prefix[:-1], targets = tokens[1:], pad to max_len-1 seq.
    // Sequence includes BOS…EOS; input = all but last, target = all but first.
    let mut max_s = 2usize;
    for ids in &token_seqs {
        max_s = max_s.max(ids.len().saturating_sub(1).max(1));
    }
    max_s = max_s.min(max_token_len.saturating_sub(1).max(1));

    let mut z_flat = vec![0.0f32; b * max_t * d];
    let mut multi = vec![0.0f32; b * k];
    let mut tin = vec![vocab.pad; b * max_s];
    let mut tgt = vec![vocab.pad; b * max_s];

    for (i, (s, e)) in samples.iter().enumerate() {
        let t = s.meta.n_frames.min(max_t);
        let row = &s.z[..t * d];
        z_flat[i * max_t * d..i * max_t * d + t * d].copy_from_slice(row);

        // Program cond: active programs in group (or full multi-hot for single).
        let mh = if single {
            e.multi_hot(k)
        } else {
            groups.group_active_multi_hot(group_id, &e.programs, k)
        };
        multi[i * k..(i + 1) * k].copy_from_slice(&mh);

        let ids = &token_seqs[i];
        let in_ids: Vec<u32> = if ids.len() >= 2 {
            ids[..ids.len() - 1].to_vec()
        } else {
            vec![vocab.bos]
        };
        let tg_ids: Vec<u32> = if ids.len() >= 2 {
            ids[1..].to_vec()
        } else {
            vec![vocab.eos]
        };
        let in_pad = pad_tokens(&in_ids, max_s, vocab.pad);
        let tg_pad = pad_tokens(&tg_ids, max_s, vocab.pad);
        tin[i * max_s..(i + 1) * max_s].copy_from_slice(&in_pad);
        tgt[i * max_s..(i + 1) * max_s].copy_from_slice(&tg_pad);
    }

    let z = Tensor::<B, 1>::from_floats(z_flat.as_slice(), device).reshape([b, max_t, d]);
    let multi_t = Tensor::<B, 1>::from_floats(multi.as_slice(), device).reshape([b, k]);
    let tokens_in = Tensor::<B, 1, Int>::from_ints(tin.as_slice(), device).reshape([b, max_s]);
    let tokens_tgt = Tensor::<B, 1, Int>::from_ints(tgt.as_slice(), device).reshape([b, max_s]);

    Ok(GroupBatch {
        z,
        multi_hot: multi_t,
        tokens_in,
        tokens_tgt,
    })
}

fn load_group_tokens(
    e: &ManifestEntry,
    group_programs: &[u8],
    vocab: &Vocab,
) -> Result<Vec<u32>> {
    let notes = load_notes_for_entry(e.notes_path.as_deref(), e.audio_path.as_deref(), &e.id)?;
    Ok(encode_group_tokens(&notes.notes, group_programs, vocab))
}

fn load_full_tokens(
    e: &ManifestEntry,
    token_cache: &TokenFeatureCache,
    vocab: &Vocab,
) -> Result<Vec<u32>> {
    if let Ok(tok) = token_cache.load(&e.id) {
        return Ok(tok.token_ids);
    }
    let notes = load_notes_for_entry(e.notes_path.as_deref(), e.audio_path.as_deref(), &e.id)?;
    Ok(encode_group_tokens(
        &notes.notes,
        &(0..vocab.cfg.n_programs as u8).collect::<Vec<_>>(),
        vocab,
    ))
}

/// Validation without Autodiff: adapter/decoder `.valid()` (InnerBackend) + batch cap.
/// `DenoteModular` is not a Burn `Module`, so we valid only the pieces used in CE.
/// Returns `(mean_loss, n_batches, n_samples)`.
fn eval_ce<B>(
    model: &DenoteModular<B>,
    g_idx: usize,
    latent_cache: &LatentFeatureCache,
    token_cache: &TokenFeatureCache,
    entries: &[ManifestEntry],
    group_programs: &[u8],
    single: bool,
    k: usize,
    max_token_len: usize,
    vocab: &Vocab,
    groups: &GroupsConfig,
    group_id: u32,
    pad_id: usize,
    label_smoothing: f32,
    batch_size: usize,
    max_val_batches: usize,
    _detach_adapter: bool,
    device: &B::Device,
) -> Result<(f32, usize, usize)>
where
    B: AutodiffBackend,
{
    let adapter = model.adapter.valid();
    let decoder = model.decoders[g_idx].valid();
    eval_ce_inner::<B::InnerBackend>(
        &adapter,
        &decoder,
        latent_cache,
        token_cache,
        entries,
        group_programs,
        single,
        k,
        max_token_len,
        vocab,
        groups,
        group_id,
        pad_id,
        label_smoothing,
        batch_size,
        max_val_batches,
        device,
    )
}

fn eval_ce_inner<B: Backend>(
    adapter: &denote_model::InstrumentAdapter<B>,
    decoder: &denote_model::TokenDecoder<B>,
    latent_cache: &LatentFeatureCache,
    token_cache: &TokenFeatureCache,
    entries: &[ManifestEntry],
    group_programs: &[u8],
    single: bool,
    k: usize,
    max_token_len: usize,
    vocab: &Vocab,
    groups: &GroupsConfig,
    group_id: u32,
    pad_id: usize,
    label_smoothing: f32,
    batch_size: usize,
    max_val_batches: usize,
    device: &B::Device,
) -> Result<(f32, usize, usize)> {
    if entries.is_empty() {
        return Ok((0.0, 0, 0));
    }
    let mut loss_sum = 0.0f32;
    let mut n_batches = 0usize;
    let mut n_samples = 0usize;
    let mut i = 0usize;
    while i < entries.len() {
        if max_val_batches > 0 && n_batches >= max_val_batches {
            break;
        }
        let end = (i + batch_size).min(entries.len());
        let batch_e: Vec<&ManifestEntry> = entries[i..end].iter().collect();
        let bsz = batch_e.len();
        let batch = load_group_batch::<B>(
            latent_cache,
            token_cache,
            &batch_e,
            group_programs,
            single,
            k,
            max_token_len,
            vocab,
            groups,
            group_id,
            device,
        )?;
        // InnerBackend: no autodiff graph; memory need not be detached.
        let adapter_out = adapter.forward_multi_hot(batch.z, batch.multi_hot);
        let token_logits = decoder.forward(batch.tokens_in, adapter_out.memory);
        let loss = token_ce_loss(
            token_logits,
            batch.tokens_tgt,
            pad_id,
            label_smoothing,
            device,
        );
        loss_sum += loss.into_scalar().elem::<f32>();
        n_batches += 1;
        n_samples += bsz;
        i = end;
    }
    let mean = if n_batches == 0 {
        0.0
    } else {
        loss_sum / n_batches as f32
    };
    Ok((mean, n_batches, n_samples))
}

fn save_job_b_checkpoint<B: Backend>(
    model: &DenoteModular<B>,
    g_idx: usize,
    group_id: u32,
    single: bool,
    ckpt_dir: &Path,
    step: u64,
    lr: f32,
    backend: &str,
    d_model: usize,
    k: usize,
    trainable_name: &str,
) -> Result<()> {
    let step_dir = ckpt_dir.join(format!("step_{step:06}"));
    let last_dir = ckpt_dir.join("last");
    fs::create_dir_all(&step_dir)?;
    fs::create_dir_all(&last_dir)?;

    for dir in [&step_dir, &last_dir] {
        let stem = decoder_stem(dir, group_id, single);
        model
            .save_decoder_g(g_idx, &stem)
            .map_err(|e| anyhow::anyhow!(e))?;
        let state = json!({
            "step": step,
            "lr": lr,
            "backend": backend,
            "d_model": d_model,
            "presence_k": k,
            "group_id": group_id,
            "single_decoder": single,
            "modular": true,
            "modules": [trainable_name],
            "trainable": [trainable_name],
            "optim_resume": false,
        });
        fs::write(dir.join("state.json"), serde_json::to_string_pretty(&state)?)?;
    }
    let st = TrainState {
        step,
        lr,
        backend: backend.to_string(),
        optim_resume: false,
    };
    st.save(&ckpt_dir.join("state.json"))?;
    info!(
        "Job B checkpoint step={step} → {}/{}",
        last_dir.display(),
        if single {
            "decoder.mpk".into()
        } else {
            format!("decoder_g{group_id}.mpk")
        }
    );
    Ok(())
}

fn decoder_stem(dir: &Path, group_id: u32, single: bool) -> PathBuf {
    if single {
        dir.join("decoder")
    } else {
        dir.join(format!("decoder_g{group_id}"))
    }
}

fn job_a_has_modules(dir: &Path) -> bool {
    path_mpk_exists(&dir.join("adapter")) && path_mpk_exists(&dir.join("presence"))
}

fn path_mpk_exists(stem: &Path) -> bool {
    stem.with_extension("mpk").exists()
        || PathBuf::from(format!("{}.mpk", stem.display())).exists()
}

fn normalize_ckpt_dir(path: &Path) -> PathBuf {
    if path.extension().and_then(|e| e.to_str()) == Some("mpk") {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

fn load_groups(path: &str) -> Result<GroupsConfig> {
    let p = Path::new(path);
    if p.exists() {
        GroupsConfig::load(p).map_err(|e| anyhow::anyhow!(e))
    } else {
        info!("groups file {path} missing — using challenge8 defaults");
        Ok(GroupsConfig::challenge8())
    }
}

fn resolve_run_dir(
    opts: &JobBOpts,
    cfg: &JobBConfig,
    group_id: u32,
    single: bool,
) -> Result<PathBuf> {
    if let Some(ref d) = opts.run_dir {
        return Ok(d.clone());
    }
    if let Some(ref resume) = opts.resume {
        if let Some(d) = run_dir_from_resume(resume) {
            return Ok(d);
        }
        if let Some(parent) = resume.parent() {
            if parent.file_name().and_then(|s| s.to_str()) == Some("checkpoints") {
                if let Some(run) = parent.parent() {
                    return Ok(run.to_path_buf());
                }
            }
        }
    }
    if let Some(ref d) = cfg.run.run_dir {
        return Ok(PathBuf::from(d));
    }
    let name = if single {
        cfg.run.name.clone()
    } else if cfg.run.name.contains(&format!("g{group_id}")) {
        cfg.run.name.clone()
    } else {
        format!("{}_g{group_id}", cfg.run.name)
    };
    Ok(default_run_dir(&name))
}
