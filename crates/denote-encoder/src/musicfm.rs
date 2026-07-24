//! MusicFM-compatible encoder: mel → conv frontend → Conformer stack → latents.

use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

use crate::config::MusicFmEncoderConfig;
use crate::conformer::{ConformerStack, ConformerStackConfig};
use crate::conv_frontend::{ConvFrontend, ConvFrontendConfig};

/// Full encoder module (weights may be frozen during adapter/decoder training).
#[derive(Module, Debug)]
pub struct MusicFmEncoder<B: Backend> {
    frontend: ConvFrontend<B>,
    backbone: ConformerStack<B>,
    /// Stored as plain data (not a burn Param).
    #[module(skip)]
    pub d_model: usize,
}

impl MusicFmEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MusicFmEncoder<B> {
        let frontend = ConvFrontendConfig {
            n_mels: self.n_mels,
            conv_dim: self.conv_dim,
            encoder_dim: self.encoder_dim,
        }
        .init(device);

        let backbone = ConformerStackConfig {
            d_model: self.encoder_dim,
            n_layers: self.encoder_depth,
            n_heads: self.n_heads,
            d_ff: self.encoder_dim * self.ffn_mult,
            dropout: self.dropout,
        }
        .init(device);

        MusicFmEncoder {
            frontend,
            backbone,
            d_model: self.encoder_dim,
        }
    }
}

impl<B: Backend> MusicFmEncoder<B> {
    /// Encode log-mel spectrogram to frame latents.
    ///
    /// # Shapes
    /// - `mel`: `[batch, 1, n_mels, time]` (MusicFM: 128 × T @ ~100 Hz)
    /// - returns: `[batch, time', encoder_dim]` (`time' ≈ time/4` @ ~25 Hz)
    pub fn forward_mel(&self, mel: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.frontend.forward(mel);
        self.backbone.forward(x)
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

/// Pack a CPU mel matrix `[n_mels, T]` into a batch-1 tensor `[1, 1, n_mels, T]`.
pub fn mel_to_tensor<B: Backend>(
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    assert_eq!(mel.len(), n_mels * n_frames);
    Tensor::<B, 1>::from_floats(mel, device).reshape([1, 1, n_mels, n_frames])
}
