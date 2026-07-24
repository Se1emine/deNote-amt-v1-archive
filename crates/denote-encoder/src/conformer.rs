//! Conformer-style stack used as MusicFM backend.
//!
//! Full bit-exact Wav2Vec2-Conformer parity is a later weight-load step.
//! Here we use Burn's TransformerEncoder with matching depth/dim for a
//! trainable, shape-compatible encoder (freeze/LoRA ready).

use burn::{
    config::Config,
    module::Module,
    nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct ConformerStackConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct ConformerStack<B: Backend> {
    encoder: TransformerEncoder<B>,
    pub d_model: usize,
}

impl ConformerStackConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConformerStack<B> {
        let encoder = TransformerEncoderConfig::new(self.d_model, self.d_ff, self.n_heads, self.n_layers)
            .with_dropout(self.dropout)
            .init(device);
        ConformerStack {
            encoder,
            d_model: self.d_model,
        }
    }
}

impl<B: Backend> ConformerStack<B> {
    /// # Shapes
    /// - `x`: `[batch, time, d_model]`
    /// - returns: `[batch, time, d_model]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(TransformerEncoderInput::new(x))
    }
}
