use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, BiLstm, BiLstmConfig, Dropout, DropoutConfig, Linear,
        LinearConfig, PaddingConfig2d, Relu,
    },
    tensor::{backend::Backend, Tensor},
};

/// Hyperparameters for Denote-CRNN-v1.
#[derive(Config, Debug)]
pub struct CrnnConfig {
    #[config(default = 229)]
    pub n_mels: usize,
    #[config(default = 88)]
    pub n_pitches: usize,
    #[config(default = 256)]
    pub feat_dim: usize,
    #[config(default = 128)]
    pub onset_hidden: usize,
    #[config(default = 128)]
    pub frame_hidden: usize,
    #[config(default = 64)]
    pub offset_hidden: usize,
    #[config(default = 0.25)]
    pub dropout: f64,
}

/// Multi-head logits, each `[batch, time, n_pitches]`.
#[derive(Debug)]
pub struct CrnnOutput<B: Backend> {
    pub onset: Tensor<B, 3>,
    pub frame: Tensor<B, 3>,
    pub offset: Tensor<B, 3>,
}

/// Compact Onsets-and-Frames style CRNN.
#[derive(Module, Debug)]
pub struct DenoteCrnn<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    pool1: MaxPool2d,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B>,
    conv4: Conv2d<B>,
    bn4: BatchNorm<B>,
    pool2: MaxPool2d,
    conv5: Conv2d<B>,
    bn5: BatchNorm<B>,
    pool3: MaxPool2d,
    fc: Linear<B>,
    dropout: Dropout,
    onset_lstm: BiLstm<B>,
    onset_head: Linear<B>,
    frame_lstm: BiLstm<B>,
    frame_head: Linear<B>,
    offset_lstm: BiLstm<B>,
    offset_head: Linear<B>,
    activation: Relu,
}

impl CrnnConfig {
    /// Frequency bins after the three max-pools on mel axis: 229 → 114 → 57 → 19.
    pub fn freq_after_pools(&self) -> usize {
        let mut f = self.n_mels;
        f /= 2; // pool (2,1)
        f /= 2; // pool (2,1)
        f /= 3; // pool (3,1)
        f
    }

    pub fn flat_dim(&self) -> usize {
        96 * self.freq_after_pools()
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DenoteCrnn<B> {
        let flat = self.flat_dim();
        DenoteCrnn {
            conv1: Conv2dConfig::new([1, 48], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn1: BatchNormConfig::new(48).init(device),
            conv2: Conv2dConfig::new([48, 48], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn2: BatchNormConfig::new(48).init(device),
            pool1: MaxPool2dConfig::new([2, 1]).init(),
            conv3: Conv2dConfig::new([48, 96], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn3: BatchNormConfig::new(96).init(device),
            conv4: Conv2dConfig::new([96, 96], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn4: BatchNormConfig::new(96).init(device),
            pool2: MaxPool2dConfig::new([2, 1]).init(),
            conv5: Conv2dConfig::new([96, 96], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn5: BatchNormConfig::new(96).init(device),
            pool3: MaxPool2dConfig::new([3, 1]).init(),
            fc: LinearConfig::new(flat, self.feat_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            onset_lstm: BiLstmConfig::new(self.feat_dim, self.onset_hidden, true).init(device),
            onset_head: LinearConfig::new(self.onset_hidden * 2, self.n_pitches).init(device),
            frame_lstm: BiLstmConfig::new(self.feat_dim + self.n_pitches, self.frame_hidden, true)
                .init(device),
            frame_head: LinearConfig::new(self.frame_hidden * 2, self.n_pitches).init(device),
            offset_lstm: BiLstmConfig::new(self.feat_dim, self.offset_hidden, true).init(device),
            offset_head: LinearConfig::new(self.offset_hidden * 2, self.n_pitches).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> DenoteCrnn<B> {
    /// Forward.
    ///
    /// # Shapes
    /// - `x`: `[batch, 1, n_mels, time]` log-mel
    /// - outputs: each `[batch, time, n_pitches]` logits
    pub fn forward(&self, x: Tensor<B, 4>) -> CrnnOutput<B> {
        let [batch, _c, _mels, time] = x.dims();

        let x = self.activation.forward(self.bn1.forward(self.conv1.forward(x)));
        let x = self.activation.forward(self.bn2.forward(self.conv2.forward(x)));
        let x = self.pool1.forward(x);

        let x = self.activation.forward(self.bn3.forward(self.conv3.forward(x)));
        let x = self.activation.forward(self.bn4.forward(self.conv4.forward(x)));
        let x = self.pool2.forward(x);

        let x = self.activation.forward(self.bn5.forward(self.conv5.forward(x)));
        let x = self.pool3.forward(x);

        // [B, C, F, T] → [B, T, C*F]
        let [b, c, f, t] = x.dims();
        debug_assert_eq!(b, batch);
        debug_assert_eq!(t, time);
        let x = x.swap_dims(1, 3); // [B, T, F, C]
        let x = x.reshape([b, t, c * f]);
        let x = self.activation.forward(self.fc.forward(x));
        let feat = self.dropout.forward(x); // [B, T, feat_dim]

        let (onset_h, _) = self.onset_lstm.forward(feat.clone(), None);
        let onset = self.onset_head.forward(onset_h); // [B, T, P]

        let frame_in = Tensor::cat(vec![feat.clone(), onset.clone()], 2);
        let (frame_h, _) = self.frame_lstm.forward(frame_in, None);
        let frame = self.frame_head.forward(frame_h);

        let (offset_h, _) = self.offset_lstm.forward(feat, None);
        let offset = self.offset_head.forward(offset_h);

        CrnnOutput {
            onset,
            frame,
            offset,
        }
    }
}
