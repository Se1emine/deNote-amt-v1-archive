use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    optim::{
        grad_clipping::GradientClippingConfig, AdamWConfig, GradientsParams, Optimizer,
    },
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::activation::sigmoid,
};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::AppConfig;
use crate::data::{batch_to_tensors, sample_batch, write_synthetic_smoke_cache};
use denote_decode::decode_notes;
use denote_eval::{evaluate_full, RefNote};
use denote_model::{multi_task_loss, DenoteCrnn};

type TrainBackend = Autodiff<NdArray>;

fn build_optimizer(
    cfg: &AppConfig,
) -> impl Optimizer<DenoteCrnn<TrainBackend>, TrainBackend> {
    let mut conf = AdamWConfig::new().with_weight_decay(cfg.train.weight_decay as f32);
    if cfg.train.grad_clip > 0.0 {
        conf = conf.with_grad_clipping(Some(GradientClippingConfig::Norm(
            cfg.train.grad_clip as f32,
        )));
    }
    conf.init::<TrainBackend, DenoteCrnn<TrainBackend>>()
}

/// Quick forward+backward on synthetic data (batch=1 so CPU smoke finishes in seconds).
pub fn run_smoke(cfg: &AppConfig) -> Result<()> {
    write_synthetic_smoke_cache(cfg)?;

    let device = Default::default();
    let model: DenoteCrnn<TrainBackend> = cfg.crnn_config().init(&device);
    info!("Model params: {}", model.num_params());

    let mut rng = StdRng::seed_from_u64(cfg.train.seed);
    // Force batch=1 — full batch=8 CRNN+BiLSTM on CPU can hang for many minutes.
    let batch = sample_batch(cfg, &mut rng, Some(1))?;
    let (mel, onset, frame, offset, frame_w) =
        batch_to_tensors::<TrainBackend>(&batch, &device);

    let pred = model.forward(mel);
    let loss = multi_task_loss(
        pred,
        onset,
        frame,
        offset,
        frame_w,
        &cfg.loss_weights(),
    );
    let loss_val = loss.total.clone().into_scalar();
    info!(
        "Smoke loss={:.4} (onset={:.4} frame={:.4} offset={:.4})",
        loss_val,
        loss.onset.clone().into_scalar(),
        loss.frame.clone().into_scalar(),
        loss.offset.clone().into_scalar()
    );

    let mut optim = build_optimizer(cfg);
    let grads = loss.total.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optim.step(cfg.train.lr, model, grads);
    info!("Smoke optim step OK");

    smoke_decode_check(&model, cfg, &device)?;
    Ok(())
}

fn smoke_decode_check(
    model: &DenoteCrnn<TrainBackend>,
    cfg: &AppConfig,
    device: &burn::tensor::Device<TrainBackend>,
) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(cfg.train.seed);
    let batch = sample_batch(cfg, &mut rng, Some(1))?;
    let (mel, onset_t, _frame_t, _offset_t, _) =
        batch_to_tensors::<TrainBackend>(&batch, device);
    let out = model.forward(mel);
    let onset = sigmoid(out.onset)
        .into_data()
        .to_vec::<f32>()
        .context("onset data")?;
    let frame = sigmoid(out.frame)
        .into_data()
        .to_vec::<f32>()
        .context("frame data")?;
    let offset = sigmoid(out.offset)
        .into_data()
        .to_vec::<f32>()
        .context("offset data")?;

    let dec = cfg.decode_config();
    let notes = decode_notes(
        &onset,
        &frame,
        Some(&offset),
        batch.n_frames,
        batch.n_pitches,
        &dec,
    );
    info!(
        "Smoke decode: {} notes (onset_thr={} min_note_ms={:.1})",
        notes.len(),
        dec.onset_threshold,
        dec.min_note_ms
    );

    let gt_onset = onset_t.into_data().to_vec::<f32>().unwrap_or_default();
    let mut refs = Vec::new();
    for t in 0..batch.n_frames {
        for p in 0..batch.n_pitches {
            if gt_onset
                .get(t * batch.n_pitches + p)
                .copied()
                .unwrap_or(0.0)
                >= 0.9
            {
                let spf = dec.seconds_per_frame();
                refs.push(RefNote {
                    start_sec: t as f64 * spf,
                    end_sec: (t + 10) as f64 * spf,
                    pitch_midi: dec.midi_offset + p as u8,
                });
            }
        }
    }
    let report = evaluate_full(&notes, &refs, 0.05);
    info!(
        "Smoke eval onset_f1={:.3} (refs={}, est={})",
        report.onset_f1.f1,
        refs.len(),
        notes.len()
    );
    Ok(())
}

/// Full training loop on precomputed feature cache (NdArray default).
pub fn run_training(cfg: &AppConfig, device_name: &str) -> Result<()> {
    let _ = device_name;
    let cache = denote_data::FeatureCache::new(&cfg.train.data_dir);
    let ids = cache.list_ids()?;
    if ids.is_empty() {
        anyhow::bail!(
            "empty cache at {} — build features first or pass --smoke",
            cfg.train.data_dir
        );
    }
    info!("{} cached tracks in {}", ids.len(), cfg.train.data_dir);

    let device = Default::default();
    let mut model: DenoteCrnn<TrainBackend> = cfg.crnn_config().init(&device);
    info!("Model params: {}", model.num_params());

    if let Some(ref path) = cfg.train.pretrained {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        model = model
            .load_file(path, &recorder, &device)
            .with_context(|| format!("load pretrained {path}"))?;
        info!("Loaded pretrained from {path}");
    }

    let mut optim = build_optimizer(cfg);
    let mut rng = StdRng::seed_from_u64(cfg.train.seed);

    let ckpt_dir = PathBuf::from(&cfg.train.checkpoint_dir);
    fs::create_dir_all(&ckpt_dir)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    let accum = cfg.grad_accum_steps();
    let micro_batch = cfg.effective_batch_size();
    let steps_per_epoch = ((ids.len().max(1) * 8) / micro_batch.max(1)).max(1);
    // Scale LR per micro-step when accumulating (simple approximation).
    let micro_lr = cfg.train.lr / accum as f64;

    info!(
        "Train: micro_batch={} grad_accum={} effective_batch={} clip={} micro_lr={:.2e}",
        micro_batch,
        accum,
        micro_batch * accum,
        cfg.train.grad_clip,
        micro_lr
    );

    for epoch in 1..=cfg.train.epochs {
        let mut epoch_loss = 0.0f32;
        let mut n = 0u32;

        for step in 0..steps_per_epoch {
            let mut accum_loss_val = 0.0f32;

            for _micro in 0..accum {
                let batch = sample_batch(cfg, &mut rng, Some(micro_batch))?;
                let (mel, onset, frame, offset, frame_w) =
                    batch_to_tensors::<TrainBackend>(&batch, &device);

                let pred = model.forward(mel);
                let loss = multi_task_loss(
                    pred,
                    onset,
                    frame,
                    offset,
                    frame_w,
                    &cfg.loss_weights(),
                );
                let lv = loss.total.clone().into_scalar();
                accum_loss_val += lv / accum as f32;

                let grads = loss.total.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(micro_lr, model, grads);
            }

            epoch_loss += accum_loss_val;
            n += 1;

            if step % 10 == 0 {
                info!("epoch {epoch} step {step}/{steps_per_epoch} loss={accum_loss_val:.4}");
            }
        }

        let mean = epoch_loss / n.max(1) as f32;
        info!("epoch {epoch} mean_loss={mean:.4}");

        let path = ckpt_dir.join(format!("epoch_{epoch}.mpk"));
        model
            .clone()
            .save_file(path.to_string_lossy().as_ref(), &recorder)
            .with_context(|| format!("save {}", path.display()))?;

        let best = ckpt_dir.join("best.mpk");
        model
            .clone()
            .save_file(best.to_string_lossy().as_ref(), &recorder)
            .context("save best")?;
    }

    Ok(())
}
