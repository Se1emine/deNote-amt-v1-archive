//! Audio loading, resampling, STFT and log-mel spectrograms.
//! Shared by train and infer so feature parity is guaranteed.

mod load;
mod mel;
mod resample;
mod write;

pub use load::{load_mono_f32, load_mono_resampled, load_wav_mono_f32, load_wav_mono_f32_max_sec};
pub use mel::{
    apply_mel_stats, log_mel_spectrogram, log_mel_spectrogram_unnormalized, musicfm_log_mel,
    MelConfig, MelSpectrogram, MelStats,
};
pub use resample::{resample_linear, resample_sinc};
pub use write::{write_wav_mono_i16, write_wav_mono_normalized};

/// Frame rate (annotations per second) for a given hop / sample rate.
pub fn frames_per_second(sample_rate: u32, hop_length: usize) -> f64 {
    sample_rate as f64 / hop_length as f64
}

/// Seconds per frame.
pub fn seconds_per_frame(sample_rate: u32, hop_length: usize) -> f64 {
    hop_length as f64 / sample_rate as f64
}

/// Number of frames for a given audio length in samples.
pub fn n_frames(n_samples: usize, hop_length: usize) -> usize {
    if n_samples == 0 {
        return 0;
    }
    1 + n_samples.saturating_sub(1) / hop_length
}
