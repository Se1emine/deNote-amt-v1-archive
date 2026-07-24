//! Conv2d subsampling frontend (MusicFM-style 2-layer residual path simplified).
//!
//! Input mel: `[batch, 1, n_mels, time]`
//! Output sequence: `[batch, time', encoder_dim]` with `time' ≈ time / 4`.

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct ConvFrontendConfig {
    pub n_mels: usize,
    pub conv_dim: usize,
    pub encoder_dim: usize,
}

#[derive(Module, Debug)]
pub struct ConvFrontend<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool1: MaxPool2d,
    pool2: MaxPool2d,
    proj: Linear<B>,
    norm: LayerNorm<B>,
    act: Gelu,
    n_mels: usize,
    encoder_dim: usize,
}

impl ConvFrontendConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvFrontend<B> {
        // Freq pooling only via stride on height; time down by pool (1,2) twice → /4 time.
        ConvFrontend {
            conv1: Conv2dConfig::new([1, self.conv_dim], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([self.conv_dim, self.conv_dim], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool1: MaxPool2dConfig::new([2, 2]).init(),
            pool2: MaxPool2dConfig::new([2, 2]).init(),
            // After two (2,2) pools: freq ≈ n_mels/4
            proj: LinearConfig::new(self.conv_dim * (self.n_mels / 4).max(1), self.encoder_dim)
                .init(device),
            norm: LayerNormConfig::new(self.encoder_dim).init(device),
            act: Gelu::new(),
            n_mels: self.n_mels,
            encoder_dim: self.encoder_dim,
        }
    }
}

impl<B: Backend> ConvFrontend<B> {
    /// # Shapes
    /// - `mel`: `[batch, 1, n_mels, time]`
    /// - returns: `[batch, time', encoder_dim]`
    pub fn forward(&self, mel: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.act.forward(self.conv1.forward(mel));
        let x = self.pool1.forward(x);
        let x = self.act.forward(self.conv2.forward(x));
        let x = self.pool2.forward(x);

        // [B, C, F, T] → [B, T, C*F]
        let [b, c, f, t] = x.dims();
        let x = x.swap_dims(1, 3).reshape([b, t, c * f]);
        let x = self.proj.forward(x);
        self.norm.forward(x)
    }

    pub fn encoder_dim(&self) -> usize {
        self.encoder_dim
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }
}
