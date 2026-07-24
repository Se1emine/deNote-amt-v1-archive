use burn::{
    tensor::{activation::sigmoid, backend::Backend, Tensor},
};

use crate::crnn::CrnnOutput;

/// Weights for multi-task AMT loss.
#[derive(Debug, Clone)]
pub struct LossWeights {
    pub onset: f32,
    pub frame: f32,
    pub offset: f32,
    /// Positive-class weight for sparse onset head.
    pub onset_pos_weight: f32,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            onset: 1.0,
            frame: 1.0,
            offset: 0.5,
            onset_pos_weight: 15.0,
        }
    }
}

/// Soft multi-label BCE with logits (supports fractional targets / onset smear).
pub fn soft_bce_with_logits<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 3>,
    pos_weight: f32,
) -> Tensor<B, 1> {
    let p = sigmoid(logits);
    let eps = 1e-7;
    let pos = targets.clone() * (p.clone().clamp(eps, 1.0 - eps)).log() * pos_weight;
    let neg = (targets.neg().add_scalar(1.0)) * ((p.neg().add_scalar(1.0)).clamp(eps, 1.0 - eps)).log();
    let loss = (pos + neg).neg();
    loss.mean()
}

/// Weighted soft BCE (element-wise weights, e.g. frame onset boost).
pub fn weighted_soft_bce_with_logits<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 3>,
    weights: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let p = sigmoid(logits);
    let eps = 1e-7;
    let pos = targets.clone() * (p.clone().clamp(eps, 1.0 - eps)).log();
    let neg = (targets.neg().add_scalar(1.0)) * ((p.neg().add_scalar(1.0)).clamp(eps, 1.0 - eps)).log();
    let loss = (pos + neg).neg() * weights;
    loss.mean()
}

#[derive(Debug)]
pub struct MultiTaskLoss<B: Backend> {
    pub total: Tensor<B, 1>,
    pub onset: Tensor<B, 1>,
    pub frame: Tensor<B, 1>,
    pub offset: Tensor<B, 1>,
}

/// L = w_on * BCE_onset + w_fr * weighted_BCE_frame + w_off * BCE_offset
pub fn multi_task_loss<B: Backend>(
    pred: CrnnOutput<B>,
    onset_tgt: Tensor<B, 3>,
    frame_tgt: Tensor<B, 3>,
    offset_tgt: Tensor<B, 3>,
    frame_weight: Tensor<B, 3>,
    weights: &LossWeights,
) -> MultiTaskLoss<B> {
    let onset = soft_bce_with_logits(pred.onset, onset_tgt, weights.onset_pos_weight);
    let frame = weighted_soft_bce_with_logits(pred.frame, frame_tgt, frame_weight);
    let offset = soft_bce_with_logits(pred.offset, offset_tgt, 1.0);

    let total = onset
        .clone()
        .mul_scalar(weights.onset)
        .add(frame.clone().mul_scalar(weights.frame))
        .add(offset.clone().mul_scalar(weights.offset));

    MultiTaskLoss {
        total,
        onset,
        frame,
        offset,
    }
}
