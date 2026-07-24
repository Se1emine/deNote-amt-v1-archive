//! MusicFM-compatible audio encoder for Denote-MIROS.
//!
//! Pipeline:
//! ```text
//! wav 24 kHz → musicfm_log_mel (denote-audio) → ConvFrontend → ConformerStack → latents [B,T,D]
//! ```
//!
//! Weight conversion from official PyTorch MusicFM checkpoints is handled by
//! `tools/convert_musicfm.py` (offline). Runtime stays pure Rust/Burn.

mod config;
mod conformer;
mod conv_frontend;
mod load_weights;
mod musicfm;

pub use config::MusicFmEncoderConfig;
pub use conformer::{ConformerStack, ConformerStackConfig};
pub use conv_frontend::{ConvFrontend, ConvFrontendConfig};
pub use load_weights::{
    load_musicfm_numpy_dir, log_weight_inventory, musicfm_weights_available, read_npy_f32,
    StateKeyEntry, WeightLoadReport,
};
pub use musicfm::{mel_to_tensor, MusicFmEncoder};

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use denote_audio::{musicfm_log_mel, MelConfig, MelStats};

    use super::*;

    type B = NdArray;

    #[test]
    fn smoke_encoder_shapes() {
        let device = Default::default();
        let cfg = MusicFmEncoderConfig::smoke();
        let enc = cfg.init::<B>(&device);

        let mel_cfg = MelConfig::musicfm();
        let audio = vec![0.05f32; mel_cfg.sample_rate as usize]; // 1 s
        let stats = MelStats::identity(mel_cfg.n_mels);
        let mel = musicfm_log_mel(&audio, &mel_cfg, &stats);
        let n_mels = mel.n_mels();
        let n_frames = mel.n_frames();
        let tensor = mel_to_tensor::<B>(mel.as_slice(), n_mels, n_frames, &device);
        let latents = enc.forward_mel(tensor);
        let [b, t, d] = latents.dims();
        assert_eq!(b, 1);
        assert_eq!(d, cfg.encoder_dim);
        // temporal /4 (pools), allow off-by-one from padding
        let expected_t = (n_frames / 4).max(1);
        assert!(
            (t as i64 - expected_t as i64).abs() <= 2,
            "t={t} expected~{expected_t} (mel frames={n_frames})"
        );
    }
}
