//! Job A: train adapter + presence on latent cache (no encoder in graph).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use burn::{
    backend::{Autodiff, NdArray},
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        activation::sigmoid, backend::AutodiffBackend, backend::Backend, ElementConversion,
        Tensor,
    },
};
use log::info;
use serde_json::json;

use crate::device::{log_backend, resolve_device, DeviceKind};
use crate::metrics_log::{
    copy_config, default_run_dir, log_run_banner, prepare_run_dir, run_dir_from_resume,
    write_hparams, RunLogger, TrainState,
};
use crate::phase1_config::JobAConfig;
use denote_data::{
    filter_curriculum, mean_n_instruments, CacheManifest, LatentFeatureCache, ManifestEntry,
};
use denote_model::{
    presence_bce_logits, presence_f1_from_slices, DenoteJobA, DenoteJobAConfig, PresenceHeadConfig,
    AdapterConfig,
};

#[derive(Debug, Clone)]
pub struct JobAOpts {
    pub config: PathBuf,
    pub run_dir: Option<PathBuf>,
    pub resume: Option<PathBuf>,
    pub device: String,
    pub max_steps: Option<u64>,
    pub log_every: Option<u64>,
    pub ckpt_every: Option<u64>,
    pub batch_size: Option<usize>,
}

pub fn run_job_a(opts: JobAOpts) -> Result<()> {
    let kind = resolve_device(&opts.device)?;
    log_backend(kind, "Phase1 Job A");

    match kind {
        DeviceKind::Cpu => run_job_a_backend::<Autodiff<NdArray>>(opts, kind),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => run_job_a_backend::<Autodiff<burn::backend::Cuda>>(opts, kind),
        #[cfg(feature = "wgpu")]
        DeviceKind::Wgpu => run_job_a_backend::<Autodiff<burn::backend::Wgpu>>(opts, kind),
        #[cfg(not(feature = "cuda"))]
        DeviceKind::Cuda => unreachable!(),
        #[cfg(not(feature = "wgpu"))]
        DeviceKind::Wgpu => unreachable!(),
    }
}

fn run_job_a_backend<B>(opts: JobAOpts, kind: DeviceKind) -> Result<()>
where
    B: AutodiffBackend,
{
    let cfg = JobAConfig::load(&opts.config)?;
    let device = B::Device::default();

    let run_dir = resolve_run_dir(&opts, &cfg)?;
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
    let d_model = cfg.model.d_model;

    write_hparams(
        &run_dir,
        &json!({
            "phase": "phase1_job_a",
            "job": "job-a",
            "device": kind.as_str(),
            "max_steps": max_steps,
            "batch_size": batch_size,
            "lr": lr,
            "d_model": d_model,
            "presence_k": k,
            "mode": cfg.model.mode,
            "use_latent_cache": cfg.model.use_latent_cache,
            "freeze_encoder": cfg.model.freeze_encoder,
            "encoder_loaded": false,
            "trainable": ["adapter", "presence"],
            "latent_manifest": cfg.data.latent_manifest,
        }),
    )?;
    log_run_banner(&run_dir, kind.as_str());

    if !cfg.model.use_latent_cache {
        bail!("Job A requires model.use_latent_cache = true");
    }

    let manifest = CacheManifest::from_path(&cfg.data.latent_manifest);
    // Root is parent of manifest file.
    let cache_root = manifest.root.clone();
    let train_entries = manifest.list_by_split("train")?;
    let val_entries = manifest.list_by_split("val")?;
    if train_entries.is_empty() {
        bail!(
            "no train entries in {} — run Job L0 first",
            cfg.data.latent_manifest
        );
    }
    let curriculum_on = cfg.curriculum.enabled;
    let curriculum_sched = cfg.curriculum.schedule();
    if curriculum_on {
        info!(
            "Curriculum ON | stage1_end={} stage2_end={} max_inst={}",
            curriculum_sched.stage1_end,
            curriculum_sched.stage2_end,
            curriculum_sched.max_instruments_cap
        );
    }
    info!(
        "Job A data | train={} val={} cache_root={} batch={} curriculum={}",
        train_entries.len(),
        val_entries.len(),
        cache_root.display(),
        batch_size,
        curriculum_on
    );

    // Infer d_model from first latent if config mismatches.
    let latent_cache = LatentFeatureCache::new(cache_root.join("latents"));
    let probe = latent_cache
        .load(&train_entries[0].id)
        .with_context(|| format!("load latent {}", train_entries[0].id))?;
    let d_model = if probe.meta.d_model != d_model {
        info!(
            "d_model config={} but latent has {}; using latent d_model",
            d_model, probe.meta.d_model
        );
        probe.meta.d_model
    } else {
        d_model
    };

    if cfg.model.mode.contains("decoder") {
        info!("mode contains decoder: Mode A2 token CE not wired in Phase 1 hard path (adapter+presence only)");
    }
    let model_cfg = DenoteJobAConfig {
        adapter: AdapterConfig::new(d_model, k)
            .with_n_pitches_aux(128)
            .with_dropout(0.0),
        presence: PresenceHeadConfig::new(d_model, k).with_hidden(d_model),
    };
    let mut model: DenoteJobA<B> = model_cfg.init(&device);
    info!(
        "DenoteJobA | params={} d_model={} K={} | encoder NOT constructed",
        model.num_params(),
        d_model,
        k
    );
    info!("Trainable modules: adapter + presence (no encoder in optim)");

    let ckpt_dir = run_dir.join("checkpoints");
    let state_path = ckpt_dir.join("state.json");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    let mut start_step: u64 = 0;
    if let Some(ref resume) = opts.resume {
        // Resume modular layout: resume/adapter + resume/presence OR full stem
        let base = normalize_ckpt_dir(resume);
        let adapter_stem = base.join("adapter");
        let presence_stem = base.join("presence");
        if adapter_stem.with_extension("mpk").exists() || Path::new(&format!("{}.mpk", adapter_stem.display())).exists()
            || fs::metadata(format!("{}.mpk", adapter_stem.display())).is_ok()
        {
            info!("Resume modular adapter from {}", adapter_stem.display());
            model.adapter = model.adapter.load_file(
                adapter_stem.to_string_lossy().as_ref(),
                &recorder,
                &device,
            )?;
            model.presence = model.presence.load_file(
                presence_stem.to_string_lossy().as_ref(),
                &recorder,
                &device,
            )?;
        } else {
            // Try directory last/
            let last = if base.file_name().and_then(|s| s.to_str()) == Some("last") {
                base.clone()
            } else {
                base.join("last")
            };
            let a = last.join("adapter");
            if path_mpk_exists(&a) {
                info!("Resume modular from {}", last.display());
                model.adapter =
                    model
                        .adapter
                        .load_file(a.to_string_lossy().as_ref(), &recorder, &device)?;
                model.presence = model.presence.load_file(
                    last.join("presence").to_string_lossy().as_ref(),
                    &recorder,
                    &device,
                )?;
            } else {
                bail!(
                    "resume path {} has no adapter.mpk (expected modular layout)",
                    resume.display()
                );
            }
        }
        if state_path.exists() {
            let st = TrainState::load(&state_path)?;
            start_step = st.step;
            info!("Resume state step={start_step}");
        }
    }

    let mut optim = AdamWConfig::new()
        .with_weight_decay(cfg.optim.weight_decay as f32)
        .init::<B, DenoteJobA<B>>();

    let mut logger = RunLogger::new(&run_dir)?;
    let end_step = start_step + max_steps;
    let mut step = start_step;
    let mut train_cursor = 0usize;

    info!(
        "Job A loop steps {}..={} log_every={log_every} val_every={val_every} \
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
            // cycle within eligible set
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
        let batch = load_batch::<B>(&latent_cache, &batch_entries, k, &device)?;

        let out = model.forward_from_latents(batch.z, batch.multi_hot.clone());
        let loss_p = presence_bce_logits(out.presence_logits.clone(), batch.targets.clone());
        let loss_presence: f32 = loss_p.clone().into_scalar().elem();
        let loss_total = loss_p * cfg.optim.w_presence;
        let loss_val: f32 = loss_total.clone().into_scalar().elem();
        if !loss_val.is_finite() {
            bail!("non-finite loss at step {step}: {loss_val}");
        }

        let grads = loss_total.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);

        let step_time_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let samples_per_sec = batch_size as f32 / (step_time_ms / 1000.0).max(1e-6);

        if step % log_every == 0 || step == end_step || step == start_step + 1 {
            // micro F1 on train batch
            let probs = sigmoid(out.presence_logits);
            let p_vec = probs.into_data().to_vec::<f32>().unwrap_or_default();
            let y_vec = batch.targets.into_data().to_vec::<f32>().unwrap_or_default();
            let f1 = presence_f1_from_slices(&p_vec, &y_vec, k, 0.5);

            info!(
                "step {step}/{end_step} loss_presence={loss_presence:.4} loss_total={loss_val:.4} \
                 train_f1_micro={:.3} lr={lr:.2e} step_time_ms={step_time_ms:.1}",
                f1.micro
            );
            let mut scalars = vec![
                ("loss_presence", loss_presence),
                ("loss_total", loss_val),
                ("presence_f1_micro", f1.micro),
                ("presence_f1_macro", f1.macro_f1),
                ("lr", lr as f32),
                ("step_time_ms", step_time_ms),
                ("samples_per_sec", samples_per_sec),
                ("data/n_instruments_mean", n_inst_mean),
            ];
            if curriculum_on {
                scalars.push((
                    "data/curriculum_stage",
                    curriculum_sched.stage_id(step) as f32,
                ));
            }
            logger.log_scalars(step, "train", &scalars)?;
            logger.flush()?;
        }

        if !val_entries.is_empty() && (step % val_every == 0 || step == end_step) {
            let t_val = Instant::now();
            let (v_loss, v_f1_micro, v_f1_macro, n_val_batches, n_val_samples) =
                eval_presence::<B>(
                    &model,
                    &latent_cache,
                    &val_entries,
                    k,
                    batch_size,
                    max_val_batches,
                    &device,
                )?;
            info!(
                "val step={step} loss_presence={v_loss:.4} presence_f1_micro={v_f1_micro:.3} \
                 presence_f1_macro={v_f1_macro:.3} batches={n_val_batches} samples={n_val_samples} \
                 val_ms={:.1}",
                t_val.elapsed().as_secs_f32() * 1000.0
            );
            logger.log_scalars(
                step,
                "val",
                &[
                    ("loss_presence", v_loss),
                    ("presence_f1_micro", v_f1_micro),
                    ("presence_f1_macro", v_f1_macro),
                    ("val_batches", n_val_batches as f32),
                    ("val_samples", n_val_samples as f32),
                ],
            )?;
            logger.flush()?;
        }

        if ckpt_every > 0 && step % ckpt_every == 0 && step != end_step {
            save_modular_checkpoint(&model, &recorder, &ckpt_dir, step, lr as f32, kind.as_str(), d_model, k)?;
        }
    }

    save_modular_checkpoint(
        &model,
        &recorder,
        &ckpt_dir,
        step,
        lr as f32,
        kind.as_str(),
        d_model,
        k,
    )?;
    logger.flush()?;
    info!(
        "Job A done | final_step={step} metrics={} modular_ckpts={}",
        run_dir.join("metrics.jsonl").display(),
        ckpt_dir.display()
    );
    info!(
        "TensorBoard: python tools/jsonl_to_tb.py --jsonl {} --out {}",
        run_dir.join("metrics.jsonl").display(),
        run_dir.join("tb").display()
    );
    Ok(())
}

struct LatentBatch<B: Backend> {
    z: Tensor<B, 3>,
    multi_hot: Tensor<B, 2>,
    targets: Tensor<B, 2>,
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

fn load_batch<B: Backend>(
    latent_cache: &LatentFeatureCache,
    entries: &[&ManifestEntry],
    k: usize,
    device: &B::Device,
) -> Result<LatentBatch<B>> {
    let b = entries.len();
    // Pad/truncate time to max T in batch.
    let mut samples = Vec::with_capacity(b);
    let mut max_t = 1usize;
    let mut d = 0usize;
    for e in entries {
        let s = latent_cache
            .load(&e.id)
            .with_context(|| format!("load latent {}", e.id))?;
        max_t = max_t.max(s.meta.n_frames.max(1));
        d = s.meta.d_model;
        samples.push((s, e));
    }
    let mut z_flat = vec![0.0f32; b * max_t * d];
    let mut multi = vec![0.0f32; b * k];
    for (i, (s, e)) in samples.iter().enumerate() {
        let t = s.meta.n_frames.min(max_t);
        let row = &s.z[..t * d];
        let dest = &mut z_flat[i * max_t * d..i * max_t * d + t * d];
        dest.copy_from_slice(row);
        let y = e.multi_hot(k);
        multi[i * k..(i + 1) * k].copy_from_slice(&y);
    }
    let z = Tensor::<B, 1>::from_floats(z_flat.as_slice(), device).reshape([b, max_t, d]);
    let multi_t = Tensor::<B, 1>::from_floats(multi.as_slice(), device).reshape([b, k]);
    Ok(LatentBatch {
        z,
        multi_hot: multi_t.clone(),
        targets: multi_t,
    })
}

/// Validation without Autodiff graph: `model.valid()` → InnerBackend, capped batches,
/// metrics streamed to CPU (no full-val logit buffer on GPU/host).
///
/// Returns `(loss, f1_micro, f1_macro, n_batches, n_samples)`.
fn eval_presence<B>(
    model: &DenoteJobA<B>,
    latent_cache: &LatentFeatureCache,
    entries: &[ManifestEntry],
    k: usize,
    batch_size: usize,
    max_val_batches: usize,
    device: &B::Device,
) -> Result<(f32, f32, f32, usize, usize)>
where
    B: AutodiffBackend,
{
    let model_v = model.valid();
    eval_presence_inner::<B::InnerBackend>(
        &model_v,
        latent_cache,
        entries,
        k,
        batch_size,
        max_val_batches,
        device,
    )
}

fn eval_presence_inner<B: Backend>(
    model: &DenoteJobA<B>,
    latent_cache: &LatentFeatureCache,
    entries: &[ManifestEntry],
    k: usize,
    batch_size: usize,
    max_val_batches: usize,
    device: &B::Device,
) -> Result<(f32, f32, f32, usize, usize)> {
    if entries.is_empty() {
        return Ok((0.0, 0.0, 0.0, 0, 0));
    }

    let mut loss_sum = 0.0f32;
    let mut n_batches = 0usize;
    let mut n_samples = 0usize;
    let mut tp = vec![0u64; k];
    let mut fp = vec![0u64; k];
    let mut fns = vec![0u64; k];

    let mut i = 0usize;
    while i < entries.len() {
        if max_val_batches > 0 && n_batches >= max_val_batches {
            break;
        }
        let end = (i + batch_size).min(entries.len());
        let batch_e: Vec<&ManifestEntry> = entries[i..end].iter().collect();
        let batch = load_batch::<B>(latent_cache, &batch_e, k, device)?;
        let bsz = batch_e.len();

        // No Autodiff graph: InnerBackend forward only.
        let out = model.forward_from_latents(batch.z, batch.multi_hot);
        let loss = presence_bce_logits(out.presence_logits.clone(), batch.targets.clone());
        loss_sum += loss.into_scalar().elem::<f32>();

        // Pull probs + targets to CPU immediately; drop GPU tensors at end of iteration.
        let p = sigmoid(out.presence_logits)
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        let y = batch
            .targets
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        accumulate_presence_counts(&p, &y, k, 0.5, &mut tp, &mut fp, &mut fns);

        n_batches += 1;
        n_samples += bsz;
        i = end;
    }

    let v_loss = if n_batches == 0 {
        0.0
    } else {
        loss_sum / n_batches as f32
    };
    let (f1_micro, f1_macro) = f1_from_counts(&tp, &fp, &fns);
    Ok((v_loss, f1_micro, f1_macro, n_batches, n_samples))
}

fn accumulate_presence_counts(
    probs: &[f32],
    targets: &[f32],
    k: usize,
    thr: f32,
    tp: &mut [u64],
    fp: &mut [u64],
    fns: &mut [u64],
) {
    if k == 0 || probs.is_empty() {
        return;
    }
    let n = probs.len() / k;
    for i in 0..n {
        for c in 0..k {
            let pred = probs[i * k + c] >= thr;
            let gold = targets.get(i * k + c).copied().unwrap_or(0.0) >= 0.5;
            match (pred, gold) {
                (true, true) => tp[c] += 1,
                (true, false) => fp[c] += 1,
                (false, true) => fns[c] += 1,
                _ => {}
            }
        }
    }
}

fn f1_from_counts(tp: &[u64], fp: &[u64], fns: &[u64]) -> (f32, f32) {
    let mut tp_sum = 0u64;
    let mut fp_sum = 0u64;
    let mut fn_sum = 0u64;
    let mut macro_acc = 0.0f32;
    let mut n_cls = 0usize;
    for i in 0..tp.len() {
        let t = tp[i];
        let f = fp[i];
        let n = fns[i];
        tp_sum += t;
        fp_sum += f;
        fn_sum += n;
        let prec = if t + f == 0 {
            0.0
        } else {
            t as f32 / (t + f) as f32
        };
        let rec = if t + n == 0 {
            0.0
        } else {
            t as f32 / (t + n) as f32
        };
        let f1 = if prec + rec == 0.0 {
            0.0
        } else {
            2.0 * prec * rec / (prec + rec)
        };
        // Only average classes that appear in gold or pred (same spirit as macro).
        if t + f + n > 0 {
            macro_acc += f1;
            n_cls += 1;
        }
    }
    let micro_p = if tp_sum + fp_sum == 0 {
        0.0
    } else {
        tp_sum as f32 / (tp_sum + fp_sum) as f32
    };
    let micro_r = if tp_sum + fn_sum == 0 {
        0.0
    } else {
        tp_sum as f32 / (tp_sum + fn_sum) as f32
    };
    let micro = if micro_p + micro_r == 0.0 {
        0.0
    } else {
        2.0 * micro_p * micro_r / (micro_p + micro_r)
    };
    let macro_f1 = if n_cls == 0 {
        0.0
    } else {
        macro_acc / n_cls as f32
    };
    (micro, macro_f1)
}

fn save_modular_checkpoint<B: Backend>(
    model: &DenoteJobA<B>,
    recorder: &NamedMpkFileRecorder<FullPrecisionSettings>,
    ckpt_dir: &Path,
    step: u64,
    lr: f32,
    backend: &str,
    d_model: usize,
    k: usize,
) -> Result<()> {
    let step_dir = ckpt_dir.join(format!("step_{step:06}"));
    let last_dir = ckpt_dir.join("last");
    fs::create_dir_all(&step_dir)?;
    fs::create_dir_all(&last_dir)?;

    for (dir, label) in [(&step_dir, "step"), (&last_dir, "last")] {
        let _ = label;
        model.adapter.clone().save_file(
            dir.join("adapter").to_string_lossy().as_ref(),
            recorder,
        )?;
        model.presence.clone().save_file(
            dir.join("presence").to_string_lossy().as_ref(),
            recorder,
        )?;
        let state = json!({
            "step": step,
            "lr": lr,
            "backend": backend,
            "d_model": d_model,
            "presence_k": k,
            "modular": true,
            "modules": ["adapter", "presence"],
            "optim_resume": false,
        });
        fs::write(dir.join("state.json"), serde_json::to_string_pretty(&state)?)?;
    }
    // Also root state.json for resume step counter (Phase 0 compat).
    let st = TrainState {
        step,
        lr,
        backend: backend.to_string(),
        optim_resume: false,
    };
    st.save(&ckpt_dir.join("state.json"))?;
    info!(
        "Modular checkpoint step={step} → {}/{{adapter,presence}}.mpk",
        last_dir.display()
    );
    Ok(())
}

fn resolve_run_dir(opts: &JobAOpts, cfg: &JobAConfig) -> Result<PathBuf> {
    if let Some(ref d) = opts.run_dir {
        return Ok(d.clone());
    }
    if let Some(ref resume) = opts.resume {
        if let Some(d) = run_dir_from_resume(resume) {
            return Ok(d);
        }
        // resume might be .../checkpoints/last
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
    Ok(default_run_dir(&cfg.run.name))
}

fn normalize_ckpt_dir(path: &Path) -> PathBuf {
    if path.extension().and_then(|e| e.to_str()) == Some("mpk") {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

fn path_mpk_exists(stem: &Path) -> bool {
    stem.with_extension("mpk").exists()
        || PathBuf::from(format!("{}.mpk", stem.display())).exists()
}
