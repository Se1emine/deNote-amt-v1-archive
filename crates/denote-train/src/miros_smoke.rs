//! Denote-MIROS CPU/GPU smoke: synthetic audio + token CE forward/backward.

use anyhow::{bail, Result};
use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Int, Tensor},
};
use log::info;

use crate::device::DeviceKind;
use denote_audio::{musicfm_log_mel, MelConfig, MelStats};
use denote_encoder::mel_to_tensor;
use denote_model::{miros_loss, DenoteMiros, DenoteMirosConfig, MirosLossWeights};
use denote_tokens::{encode_notes, pad_tokens, TokenNote, Vocab};

/// One teacher-forcing step on a tiny sine + MIDI token sequence.
pub fn run_miros_smoke(kind: DeviceKind) -> Result<()> {
    match kind {
        DeviceKind::Cpu => run_smoke_backend::<Autodiff<NdArray>>(),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => run_smoke_backend::<Autodiff<burn::backend::Cuda>>(),
        #[cfg(feature = "wgpu")]
        DeviceKind::Wgpu => run_smoke_backend::<Autodiff<burn::backend::Wgpu>>(),
        #[cfg(not(feature = "cuda"))]
        DeviceKind::Cuda => bail!("cuda feature not enabled"),
        #[cfg(not(feature = "wgpu"))]
        DeviceKind::Wgpu => bail!("wgpu feature not enabled"),
    }
}

fn run_smoke_backend<B: AutodiffBackend>() -> Result<()> {
    let device = B::Device::default();
    let vocab = Vocab::default_mt3_like();
    let model_cfg = DenoteMirosConfig::smoke(vocab.size as usize);
    let model: DenoteMiros<B> = model_cfg.init(&device);
    info!(
        "Denote-MIROS smoke | params={} vocab={}",
        model.num_params(),
        vocab.size
    );

    // 1 s of 440 Hz-ish tone at 24 kHz
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
    let mel_t = mel_to_tensor::<B>(mel.as_slice(), mel.n_mels(), mel.n_frames(), &device);

    let notes = [TokenNote {
        start_sec: 0.10,
        end_sec: 0.50,
        pitch_midi: 69, // A4
        velocity: 90,
        program: 0,
    }];
    let ids = encode_notes(&notes, &vocab);
    let max_len = 64usize.min(ids.len().max(8));
    let padded = pad_tokens(&ids, max_len, vocab.pad);
    // teacher forcing: in = tokens[..S-1], target = tokens[1..]
    let s = padded.len();
    let input_ids: Vec<i32> = padded[..s - 1].iter().map(|&t| t as i32).collect();
    let target_ids: Vec<i32> = padded[1..].iter().map(|&t| t as i32).collect();
    let seq = input_ids.len();

    let tokens_in = Tensor::<B, 1, Int>::from_ints(input_ids.as_slice(), &device)
        .reshape([1, seq]);
    let tokens_tgt = Tensor::<B, 1, Int>::from_ints(target_ids.as_slice(), &device)
        .reshape([1, seq]);
    let program = Tensor::<B, 1, Int>::from_ints([0i32], &device);

    let out = model.forward(mel_t, tokens_in, program);
    let weights = MirosLossWeights {
        token: 1.0,
        aux_pitch: 0.0,
        aux_onset: 0.0,
        label_smoothing: 0.0,
        pad_id: vocab.pad as usize,
    };
    let loss = miros_loss(out, tokens_tgt, None, None, &weights, &device);
    let loss_val = loss.total.clone().into_scalar();
    info!("MIROS smoke CE loss={loss_val:.4}");

    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<B, DenoteMiros<B>>();
    let grads = loss.total.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let _model = optim.step(3e-4, model, grads);
    info!("MIROS smoke optim step OK");
    Ok(())
}

/// Detect MIROS config by path or contents marker.
pub fn is_miros_config(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|n| n.contains("miros") || n.contains("phase0"))
        .unwrap_or(false)
}
