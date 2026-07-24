//! Token cross-entropy (+ optional aux pitch/onset) for Denote-MIROS.

use burn::{
    nn::loss::CrossEntropyLossConfig,
    tensor::{backend::Backend, activation::sigmoid, Int, Tensor},
};

use crate::miros::MirosOutput;

#[derive(Debug, Clone)]
pub struct MirosLossWeights {
    pub token: f32,
    pub aux_pitch: f32,
    pub aux_onset: f32,
    pub label_smoothing: f32,
    pub pad_id: usize,
}

impl Default for MirosLossWeights {
    fn default() -> Self {
        Self {
            token: 1.0,
            aux_pitch: 0.1,
            aux_onset: 0.1,
            label_smoothing: 0.1,
            pad_id: 0,
        }
    }
}

pub struct MirosLoss<B: Backend> {
    pub total: Tensor<B, 1>,
    pub token: Tensor<B, 1>,
    pub aux: Option<Tensor<B, 1>>,
}

/// Flatten token logits/targets and compute CE with pad ignore + optional smoothing.
pub fn token_ce_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    pad_id: usize,
    label_smoothing: f32,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [b, s, v] = logits.dims();
    let logits_2d = logits.reshape([b * s, v]);
    let targets_1d = targets.reshape([b * s]);
    let mut cfg = CrossEntropyLossConfig::new().with_pad_tokens(Some(vec![pad_id]));
    if label_smoothing > 0.0 {
        cfg = cfg.with_smoothing(Some(label_smoothing));
    }
    cfg.init(device).forward(logits_2d, targets_1d)
}

/// Soft multi-label BCE for aux maps `[B, T, P]`.
pub fn soft_bce_logits<B: Backend>(logits: Tensor<B, 3>, targets: Tensor<B, 3>) -> Tensor<B, 1> {
    let p = sigmoid(logits);
    let eps = 1e-7;
    let pos = targets.clone() * (p.clone().clamp(eps, 1.0 - eps)).log();
    let neg = (targets.neg().add_scalar(1.0)) * ((p.neg().add_scalar(1.0)).clamp(eps, 1.0 - eps)).log();
    (pos + neg).neg().mean()
}

/// Combine token CE with optional aux heads.
pub fn miros_loss<B: Backend>(
    out: MirosOutput<B>,
    token_targets: Tensor<B, 2, Int>,
    aux_pitch: Option<Tensor<B, 3>>,
    aux_onset: Option<Tensor<B, 3>>,
    weights: &MirosLossWeights,
    device: &B::Device,
) -> MirosLoss<B> {
    let token = token_ce_loss(
        out.token_logits,
        token_targets,
        weights.pad_id,
        weights.label_smoothing,
        device,
    ) * weights.token;

    let mut total = token.clone();
    let mut aux_acc: Option<Tensor<B, 1>> = None;

    if let (Some(tp), Some(w)) = (aux_pitch, Some(weights.aux_pitch).filter(|w| *w > 0.0)) {
        let a = soft_bce_logits(out.adapter.pitch_logits, tp) * w;
        total = total + a.clone();
        aux_acc = Some(a);
    }
    if let (Some(to), Some(w)) = (aux_onset, Some(weights.aux_onset).filter(|w| *w > 0.0)) {
        let a = soft_bce_logits(out.adapter.onset_logits, to) * w;
        total = total + a.clone();
        aux_acc = match aux_acc {
            Some(prev) => Some(prev + a),
            None => Some(a),
        };
    }

    MirosLoss {
        total,
        token,
        aux: aux_acc,
    }
}
