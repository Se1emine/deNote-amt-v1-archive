//! Phase 0 mini training loop: synthetic TF CE steps + checkpoint/resume.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
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
use denote_audio::{musicfm_log_mel, MelConfig, MelStats};
use denote_encoder::mel_to_tensor;
use denote_model::{miros_loss, DenoteMiros, DenoteMirosConfig, MirosLossWeights};
use denote_tokens::{encode_notes, pad_tokens, TokenNote, Vocab};

/// CLI-facing options for `--mini-steps`.
#[derive(Debug, Clone)]
pub struct MiniTrainOpts {
    pub config: PathBuf,
    pub mini_steps: u64,
    pub run_dir: Option<PathBuf>,
    pub resume: Option<PathBuf>,
    pub device: String,
    pub log_every: u64,
    pub ckpt_every: u64,
    pub lr: f64,
}

/// Entry: resolve device feature → dispatch concrete Burn backend.
pub fn run_mini_train(opts: MiniTrainOpts) -> Result<()> {
    let kind = resolve_device(&opts.device)?;
    log_backend(kind, "Phase0 mini-train");

    match kind {
        DeviceKind::Cpu => run_mini_train_backend::<Autodiff<NdArray>>(opts, kind),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => {
            run_mini_train_backend::<Autodiff<burn::backend::Cuda>>(opts, kind)
        }
        #[cfg(feature = "wgpu")]
        DeviceKind::Wgpu => {
            run_mini_train_backend::<Autodiff<burn::backend::Wgpu>>(opts, kind)
        }
        #[cfg(not(feature = "cuda"))]
        DeviceKind::Cuda => unreachable!("feature checked in resolve_device"),
        #[cfg(not(feature = "wgpu"))]
        DeviceKind::Wgpu => unreachable!("feature checked in resolve_device"),
    }
}

fn run_mini_train_backend<B>(opts: MiniTrainOpts, kind: DeviceKind) -> Result<()>
where
    B: AutodiffBackend,
{
    let device = B::Device::default();
    let run_dir = resolve_run_dir(&opts)?;
    prepare_run_dir(&run_dir)?;
    if opts.config.exists() {
        let _ = copy_config(&opts.config, &run_dir);
    }
    write_hparams(
        &run_dir,
        &json!({
            "mini_steps": opts.mini_steps,
            "lr": opts.lr,
            "device": kind.as_str(),
            "log_every": opts.log_every,
            "ckpt_every": opts.ckpt_every,
            "config": opts.config.display().to_string(),
            "phase": "phase0_mini",
            "data": "synthetic",
        }),
    )?;
    log_run_banner(&run_dir, kind.as_str());

    let vocab = Vocab::default_mt3_like();
    let model_cfg = DenoteMirosConfig::smoke(vocab.size as usize);
    let mut model: DenoteMiros<B> = model_cfg.init(&device);
    info!(
        "Denote-MIROS mini | params={} vocab={} run_dir={}",
        model.num_params(),
        vocab.size,
        run_dir.display()
    );

    let ckpt_dir = run_dir.join("checkpoints");
    let state_path = ckpt_dir.join("state.json");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    let mut start_step: u64 = 0;
    if let Some(ref resume) = opts.resume {
        let load_path = normalize_ckpt_path(resume);
        info!("Resume model from {}", load_path.display());
        model = model
            .load_file(load_path.to_string_lossy().as_ref(), &recorder, &device)
            .with_context(|| format!("load checkpoint {}", load_path.display()))?;
        if state_path.exists() {
            let st = TrainState::load(&state_path)?;
            start_step = st.step;
            info!(
                "Resume state step={} (optim not restored — Phase 0)",
                start_step
            );
        } else {
            info!("No state.json next to checkpoints; starting step counter at 0");
        }
    }

    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<B, DenoteMiros<B>>();
    let lr = opts.lr as f32;
    let weights = MirosLossWeights {
        token: 1.0,
        aux_pitch: 0.0,
        aux_onset: 0.0,
        label_smoothing: 0.0,
        pad_id: vocab.pad as usize,
    };

    // Synthetic batch built once (same path as smoke); re-used every step.
    let batch = synthetic_batch::<B>(&vocab, &device)?;
    let mut logger = RunLogger::new(&run_dir)?;

    let end_step = start_step + opts.mini_steps;
    info!(
        "Mini-train loop steps {}..={} ({} steps) log_every={} ckpt_every={}",
        start_step + 1,
        end_step,
        opts.mini_steps,
        opts.log_every,
        opts.ckpt_every
    );

    let mut step = start_step;
    while step < end_step {
        step += 1;
        let t0 = Instant::now();

        let out = model.forward(
            batch.mel.clone(),
            batch.tokens_in.clone(),
            batch.program.clone(),
        );
        let loss = miros_loss(
            out,
            batch.tokens_tgt.clone(),
            None,
            None,
            &weights,
            &device,
        );
        let loss_val: f32 = loss.total.clone().into_scalar().elem();
        if !loss_val.is_finite() {
            bail!("non-finite loss at step {step}: {loss_val}");
        }

        let grads = loss.total.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(opts.lr, model, grads);

        let step_time_ms = t0.elapsed().as_secs_f32() * 1000.0;

        if step % opts.log_every == 0 || step == end_step || step == start_step + 1 {
            info!(
                "step {step}/{end_step} loss={loss_val:.4} lr={lr:.2e} step_time_ms={step_time_ms:.1}"
            );
            logger.log_scalars(
                step,
                "train",
                &[
                    ("loss", loss_val),
                    ("lr", lr),
                    ("step_time_ms", step_time_ms),
                ],
            )?;
            logger.flush()?;
        }

        // Periodic step_NNNNNN + last; final step always saved after the loop.
        if opts.ckpt_every > 0 && step % opts.ckpt_every == 0 && step != end_step {
            save_checkpoint(&model, &recorder, &ckpt_dir, step, lr, kind.as_str())?;
        }
    }

    save_checkpoint(&model, &recorder, &ckpt_dir, step, lr, kind.as_str())?;
    logger.flush()?;

    info!(
        "Mini-train done | final_step={step} metrics={} checkpoints={}",
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

struct SyntheticBatch<B: Backend> {
    mel: Tensor<B, 4>,
    tokens_in: Tensor<B, 2, Int>,
    tokens_tgt: Tensor<B, 2, Int>,
    program: Tensor<B, 1, Int>,
}

fn synthetic_batch<B: Backend>(vocab: &Vocab, device: &B::Device) -> Result<SyntheticBatch<B>> {
    // 1 s of 440 Hz tone @ 24 kHz (same as miros_smoke).
    let sr = 24_000u32;
    let n = sr as usize;
    let audio: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.2
        })
        .collect();
    let mel_cfg = MelConfig::musicfm();
    let stats = MelStats::identity(mel_cfg.n_mels);
    let mel = musicfm_log_mel(&audio, &mel_cfg, &stats);
    let mel_t = mel_to_tensor::<B>(mel.as_slice(), mel.n_mels(), mel.n_frames(), device);

    let notes = [TokenNote {
        start_sec: 0.10,
        end_sec: 0.50,
        pitch_midi: 69,
        velocity: 90,
        program: 0,
    }];
    let ids = encode_notes(&notes, vocab);
    let max_len = 64usize.min(ids.len().max(8));
    let padded = pad_tokens(&ids, max_len, vocab.pad);
    let s = padded.len();
    let input_ids: Vec<i32> = padded[..s - 1].iter().map(|&t| t as i32).collect();
    let target_ids: Vec<i32> = padded[1..].iter().map(|&t| t as i32).collect();
    let seq = input_ids.len();

    let tokens_in = Tensor::<B, 1, Int>::from_ints(input_ids.as_slice(), device).reshape([1, seq]);
    let tokens_tgt =
        Tensor::<B, 1, Int>::from_ints(target_ids.as_slice(), device).reshape([1, seq]);
    let program = Tensor::<B, 1, Int>::from_ints([0i32], device);

    Ok(SyntheticBatch {
        mel: mel_t,
        tokens_in,
        tokens_tgt,
        program,
    })
}

fn save_checkpoint<B: Backend>(
    model: &DenoteMiros<B>,
    recorder: &NamedMpkFileRecorder<FullPrecisionSettings>,
    ckpt_dir: &Path,
    step: u64,
    lr: f32,
    backend: &str,
) -> Result<()> {
    let step_stem = ckpt_dir.join(format!("step_{step:06}"));
    let last_stem = ckpt_dir.join("last");
    model
        .clone()
        .save_file(step_stem.to_string_lossy().as_ref(), recorder)
        .with_context(|| format!("save {}", step_stem.display()))?;
    model
        .clone()
        .save_file(last_stem.to_string_lossy().as_ref(), recorder)
        .with_context(|| format!("save {}", last_stem.display()))?;
    let state = TrainState {
        step,
        lr,
        backend: backend.to_string(),
        optim_resume: false,
    };
    state.save(&ckpt_dir.join("state.json"))?;
    info!("Checkpoint saved step={step} → {}", last_stem.display());
    Ok(())
}

fn resolve_run_dir(opts: &MiniTrainOpts) -> Result<PathBuf> {
    if let Some(ref d) = opts.run_dir {
        return Ok(d.clone());
    }
    if let Some(ref resume) = opts.resume {
        if let Some(d) = run_dir_from_resume(resume) {
            return Ok(d);
        }
    }
    Ok(default_run_dir("phase0_mini"))
}

/// Strip `.mpk` if the user passed a full filename; Burn adds the extension itself.
fn normalize_ckpt_path(path: &Path) -> PathBuf {
    if path.extension().and_then(|e| e.to_str()) == Some("mpk") {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}
