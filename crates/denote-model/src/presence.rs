//! Multi-label instrument presence head (Job A).
//!
//! `presence_logits = MLP(mean_pool(adapter.memory))` → `[B, K]`.

use burn::{
    config::Config,
    module::Module,
    nn::{Gelu, Linear, LinearConfig},
    tensor::{activation::sigmoid, backend::Backend, Tensor},
};

#[cfg(test)]
use burn::tensor::ElementConversion;

#[derive(Config, Debug)]
pub struct PresenceHeadConfig {
    pub d_model: usize,
    /// Number of challenge programs K (default 8).
    pub n_programs: usize,
    /// Hidden size of MLP (0 → use `d_model`).
    #[config(default = 0)]
    pub hidden: usize,
}

/// Multi-label presence classifier over pooled adapter memory.
#[derive(Module, Debug)]
pub struct PresenceHead<B: Backend> {
    fc1: Linear<B>,
    fc_out: Linear<B>,
    act: Gelu,
    #[module(skip)]
    pub n_programs: usize,
    #[module(skip)]
    pub d_model: usize,
}

impl PresenceHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PresenceHead<B> {
        let hidden = if self.hidden == 0 {
            self.d_model
        } else {
            self.hidden
        };
        PresenceHead {
            fc1: LinearConfig::new(self.d_model, hidden).init(device),
            fc_out: LinearConfig::new(hidden, self.n_programs).init(device),
            act: Gelu::new(),
            n_programs: self.n_programs,
            d_model: self.d_model,
        }
    }
}

impl<B: Backend> PresenceHead<B> {
    /// # Shapes
    /// - `memory`: `[B, T, D]` adapter output
    /// - returns: `[B, K]` logits (pre-sigmoid)
    pub fn forward(&self, memory: Tensor<B, 3>) -> Tensor<B, 2> {
        // Mean pool over time. `sum_dim(1)` keeps dim → [B, 1, D].
        let [b, t, d] = memory.dims();
        let x = memory.sum_dim(1).reshape([b, d]) / (t as f32).max(1.0);
        let x = self.act.forward(self.fc1.forward(x));
        self.fc_out.forward(x)
    }

    /// Convenience: sigmoid probabilities `[B, K]`.
    pub fn probs(&self, memory: Tensor<B, 3>) -> Tensor<B, 2> {
        sigmoid(self.forward(memory))
    }
}

/// Multi-label BCE with logits for presence `[B, K]`.
pub fn presence_bce_logits<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let p = sigmoid(logits);
    let eps = 1e-7;
    let pos = targets.clone() * (p.clone().clamp(eps, 1.0 - eps)).log();
    let neg =
        (targets.neg().add_scalar(1.0)) * ((p.neg().add_scalar(1.0)).clamp(eps, 1.0 - eps)).log();
    (pos + neg).neg().mean()
}

/// Micro / macro F1 for multi-label presence (CPU-side helper on flat slices).
///
/// `probs` and `targets` are length `B * K`, row-major.
#[derive(Debug, Clone, Copy)]
pub struct PresenceF1 {
    pub micro: f32,
    pub macro_f1: f32,
}

pub fn presence_f1_from_slices(probs: &[f32], targets: &[f32], k: usize, thr: f32) -> PresenceF1 {
    assert_eq!(probs.len(), targets.len());
    assert!(k > 0);
    let n = probs.len() / k;
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_ = 0u64;
    let mut macro_sum = 0.0f32;
    let mut macro_n = 0u32;

    for c in 0..k {
        let mut c_tp = 0u64;
        let mut c_fp = 0u64;
        let mut c_fn = 0u64;
        for b in 0..n {
            let i = b * k + c;
            let pred = probs[i] >= thr;
            let gold = targets[i] >= 0.5;
            match (pred, gold) {
                (true, true) => {
                    c_tp += 1;
                    tp += 1;
                }
                (true, false) => {
                    c_fp += 1;
                    fp += 1;
                }
                (false, true) => {
                    c_fn += 1;
                    fn_ += 1;
                }
                (false, false) => {}
            }
        }
        let p = if c_tp + c_fp == 0 {
            0.0
        } else {
            c_tp as f32 / (c_tp + c_fp) as f32
        };
        let r = if c_tp + c_fn == 0 {
            0.0
        } else {
            c_tp as f32 / (c_tp + c_fn) as f32
        };
        if c_tp + c_fp + c_fn > 0 {
            let f1 = if p + r == 0.0 {
                0.0
            } else {
                2.0 * p * r / (p + r)
            };
            macro_sum += f1;
            macro_n += 1;
        }
    }

    let micro_p = if tp + fp == 0 {
        0.0
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let micro_r = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_) as f32
    };
    let micro = if micro_p + micro_r == 0.0 {
        0.0
    } else {
        2.0 * micro_p * micro_r / (micro_p + micro_r)
    };
    let macro_f1 = if macro_n == 0 {
        0.0
    } else {
        macro_sum / macro_n as f32
    };
    PresenceF1 { micro, macro_f1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn presence_forward_shapes() {
        let device = Default::default();
        let head = PresenceHeadConfig::new(32, 8).with_hidden(16).init::<B>(&device);
        let mem = Tensor::<B, 3>::zeros([2, 10, 32], &device);
        let logits = head.forward(mem);
        assert_eq!(logits.dims(), [2, 8]);
    }

    #[test]
    fn presence_bce_finite() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::zeros([4, 8], &device);
        let targets = Tensor::<B, 2>::zeros([4, 8], &device);
        let loss = presence_bce_logits(logits, targets);
        let v: f32 = loss.into_scalar().elem();
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    fn presence_f1_perfect() {
        let probs = vec![0.9, 0.1, 0.8, 0.2];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        let f1 = presence_f1_from_slices(&probs, &targets, 2, 0.5);
        assert!((f1.micro - 1.0).abs() < 1e-5);
    }
}
