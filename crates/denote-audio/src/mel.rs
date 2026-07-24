use ndarray::Array2;
use realfft::RealFftPlanner;
use serde::{Deserialize, Serialize};

/// Log-mel configuration — single source of truth for train/infer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub mel_eps: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            n_fft: 2048,
            hop_length: 256,
            n_mels: 229,
            fmin: 30.0,
            fmax: 11025.0,
            mel_eps: 1e-5,
        }
    }
}

impl MelConfig {
    /// MusicFM frontend defaults: 24 kHz, 128 mels, hop 240 → ~100 Hz frames
    /// (after 2×2 conv subsampling → 25 Hz latents).
    pub fn musicfm() -> Self {
        Self {
            sample_rate: 24_000,
            n_fft: 2048,
            hop_length: 240,
            n_mels: 128,
            fmin: 0.0,
            fmax: 12_000.0,
            mel_eps: 1e-5,
        }
    }

    pub fn frames_per_second(&self) -> f64 {
        self.sample_rate as f64 / self.hop_length as f64
    }

    pub fn seconds_per_frame(&self) -> f64 {
        self.hop_length as f64 / self.sample_rate as f64
    }

    pub fn n_frames(&self, n_samples: usize) -> usize {
        if n_samples < self.n_fft {
            return if n_samples == 0 { 0 } else { 1 };
        }
        1 + (n_samples - self.n_fft) / self.hop_length
    }
}

/// Dataset-level mean/std for MusicFM-style normalization (from `msd_stats.json` / `fma_stats.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl MelStats {
    /// Identity stats (no-op normalize) for smoke tests without the official JSON.
    pub fn identity(n_mels: usize) -> Self {
        Self {
            mean: vec![0.0; n_mels],
            std: vec![1.0; n_mels],
        }
    }
}

/// Mel spectrogram matrix: shape `[n_mels, n_frames]` (freq × time).
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    pub data: Array2<f32>,
    pub config: MelConfig,
}

impl MelSpectrogram {
    pub fn n_mels(&self) -> usize {
        self.data.nrows()
    }

    pub fn n_frames(&self) -> usize {
        self.data.ncols()
    }

    /// Flatten to row-major `[n_mels * n_frames]` for serialization.
    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice().expect("mel contiguous")
    }
}

/// Compute log-mel spectrogram of mono audio with **per-clip** mean/std normalize.
/// Output shape: `[n_mels, T]`. Used by the legacy CRNN path.
pub fn log_mel_spectrogram(audio: &[f32], cfg: &MelConfig) -> MelSpectrogram {
    let mut mel = log_mel_spectrogram_unnormalized(audio, cfg);
    per_clip_normalize(&mut mel.data);
    MelSpectrogram {
        data: mel.data,
        config: cfg.clone(),
    }
}

/// Log-mel without normalization (MusicFM applies dataset stats separately).
pub fn log_mel_spectrogram_unnormalized(audio: &[f32], cfg: &MelConfig) -> MelSpectrogram {
    let n_fft = cfg.n_fft;
    let hop = cfg.hop_length;
    let n_freqs = n_fft / 2 + 1;

    let window = hann_window(n_fft);
    let mel_fb = mel_filterbank(
        cfg.sample_rate,
        n_fft,
        cfg.n_mels,
        cfg.fmin,
        cfg.fmax.min(cfg.sample_rate as f32 * 0.5),
    );

    let n_frames = if audio.len() < n_fft {
        1
    } else {
        1 + (audio.len() - n_fft) / hop
    };

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut indata = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();

    let mut power = Array2::<f32>::zeros((n_freqs, n_frames));

    for t in 0..n_frames {
        let start = t * hop;
        indata.fill(0.0);
        for i in 0..n_fft {
            let idx = start + i;
            if idx < audio.len() {
                indata[i] = audio[idx] * window[i];
            }
        }
        r2c.process(&mut indata, &mut spectrum).expect("fft");
        for (k, c) in spectrum.iter().enumerate() {
            // power spectrum
            power[[k, t]] = c.re * c.re + c.im * c.im;
        }
    }

    // mel = mel_fb [n_mels, n_freqs] @ power [n_freqs, T]
    let mut mel = Array2::<f32>::zeros((cfg.n_mels, n_frames));
    for m in 0..cfg.n_mels {
        for t in 0..n_frames {
            let mut s = 0.0f32;
            for k in 0..n_freqs {
                s += mel_fb[[m, k]] * power[[k, t]];
            }
            mel[[m, t]] = (s + cfg.mel_eps).ln();
        }
    }

    MelSpectrogram {
        data: mel,
        config: cfg.clone(),
    }
}

/// MusicFM-style: log-mel + per-band dataset mean/std.
pub fn musicfm_log_mel(audio: &[f32], cfg: &MelConfig, stats: &MelStats) -> MelSpectrogram {
    let mut mel = log_mel_spectrogram_unnormalized(audio, cfg);
    apply_mel_stats(&mut mel.data, stats);
    mel
}

fn per_clip_normalize(mel: &mut Array2<f32>) {
    let mean = mel.mean().unwrap_or(0.0);
    let var = mel
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / (mel.len() as f32).max(1.0);
    let std = (var + 1e-6).sqrt();
    mel.mapv_inplace(|v| (v - mean) / std);
}

/// Apply per-mel-bin stats: `x[m,t] = (x[m,t] - mean[m]) / std[m]`.
pub fn apply_mel_stats(mel: &mut Array2<f32>, stats: &MelStats) {
    let n_mels = mel.nrows();
    let t = mel.ncols();
    assert_eq!(stats.mean.len(), n_mels, "stats.mean len");
    assert_eq!(stats.std.len(), n_mels, "stats.std len");
    for m in 0..n_mels {
        let mean = stats.mean[m];
        let std = stats.std[m].max(1e-6);
        for ti in 0..t {
            mel[[m, ti]] = (mel[[m, ti]] - mean) / std;
        }
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * 2.0 * i as f32 / n as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Slaney-style triangular mel filterbank: `[n_mels, n_fft/2+1]`.
fn mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Array2<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mut fb = Array2::<f32>::zeros((n_mels, n_freqs));

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bins: Vec<usize> = hz_points
        .iter()
        .map(|&hz| {
            ((n_fft as f32 + 1.0) * hz / sample_rate as f32)
                .floor()
                .clamp(0.0, (n_freqs - 1) as f32) as usize
        })
        .collect();

    for m in 0..n_mels {
        let left = bins[m];
        let center = bins[m + 1];
        let right = bins[m + 2];
        if center == left || right == center {
            continue;
        }
        for k in left..center {
            fb[[m, k]] = (k - left) as f32 / (center - left) as f32;
        }
        for k in center..right {
            fb[[m, k]] = (right - k) as f32 / (right - center) as f32;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_shape_for_two_seconds() {
        let cfg = MelConfig::default();
        let n = cfg.sample_rate as usize * 2;
        let audio = vec![0.0f32; n];
        let mel = log_mel_spectrogram(&audio, &cfg);
        assert_eq!(mel.n_mels(), 229);
        assert!(mel.n_frames() > 100);
    }

    #[test]
    fn musicfm_mel_shape_and_25hz_latent_rate() {
        let cfg = MelConfig::musicfm();
        let n = cfg.sample_rate as usize * 2; // 2 s
        let audio = vec![0.1f32; n];
        let stats = MelStats::identity(cfg.n_mels);
        let mel = musicfm_log_mel(&audio, &cfg, &stats);
        assert_eq!(mel.n_mels(), 128);
        // ~100 Hz mel frames; after 4× temporal downsample → ~25 Hz
        let mel_hz = cfg.frames_per_second();
        assert!((mel_hz - 100.0).abs() < 1.0, "mel_hz={mel_hz}");
        let latent_hz = mel_hz / 4.0;
        assert!((latent_hz - 25.0).abs() < 0.5, "latent_hz={latent_hz}");
        assert!(mel.n_frames() > 150);
    }
}
