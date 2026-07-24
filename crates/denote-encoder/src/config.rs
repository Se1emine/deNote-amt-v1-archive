use serde::{Deserialize, Serialize};

/// MusicFM-compatible encoder hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicFmEncoderConfig {
    pub n_mels: usize,
    pub conv_dim: usize,
    pub encoder_dim: usize,
    pub encoder_depth: usize,
    pub n_heads: usize,
    pub ffn_mult: usize,
    pub dropout: f64,
    /// Temporal downsample factor of the conv frontend (MusicFM: 2×2 = 4).
    pub temporal_downsample: usize,
}

impl Default for MusicFmEncoderConfig {
    fn default() -> Self {
        Self {
            n_mels: 128,
            conv_dim: 512,
            encoder_dim: 1024,
            encoder_depth: 12,
            n_heads: 16,
            ffn_mult: 4,
            dropout: 0.1,
            temporal_downsample: 4,
        }
    }
}

impl MusicFmEncoderConfig {
    /// Tiny config for unit tests / ultra-light CPU only (not MusicFM-width).
    /// Production / phase train path uses configs with encoder_dim=1024.
    pub fn smoke() -> Self {
        Self::smoke_tiny()
    }

    /// Alias: deliberately non-MusicFM width for fast tests.
    pub fn smoke_tiny() -> Self {
        Self {
            n_mels: 128,
            conv_dim: 64,
            encoder_dim: 128,
            encoder_depth: 2,
            n_heads: 4,
            ffn_mult: 2,
            dropout: 0.0,
            temporal_downsample: 4,
        }
    }

    /// MusicFM-width scaffold without pretrained weights (depth reduced for 1070 L0).
    pub fn musicfm_width_scaffold() -> Self {
        Self {
            n_mels: 128,
            conv_dim: 512,
            encoder_dim: 1024,
            encoder_depth: 4,
            n_heads: 16,
            ffn_mult: 4,
            dropout: 0.0,
            temporal_downsample: 4,
        }
    }

    /// Full MusicFM-MSD layout (requires converted checkpoint).
    pub fn musicfm_msd() -> Self {
        Self::default()
    }
}
