use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub audio: AudioSection,
    pub model: ModelSection,
    pub labels: LabelsSection,
    pub train: TrainSection,
    pub decode: DecodeSection,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioSection {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub mel_eps: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSection {
    pub n_pitches: usize,
    pub midi_offset: u8,
    pub feat_dim: usize,
    pub onset_hidden: usize,
    pub frame_hidden: usize,
    pub offset_hidden: usize,
    pub dropout: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LabelsSection {
    pub onset_smear_frames: usize,
    pub frame_onset_boost_frames: usize,
    pub frame_onset_boost: f32,
    pub min_note_frames: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainSection {
    pub chunk_sec: f64,
    pub batch_size: usize,
    pub grad_accum: usize,
    pub epochs: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub seed: u64,
    pub checkpoint_dir: String,
    pub data_dir: String,
    pub onset_weight: f32,
    pub frame_weight: f32,
    pub offset_weight: f32,
    pub onset_pos_weight: f32,
    #[serde(default)]
    pub pretrained: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecodeSection {
    pub onset_threshold: f32,
    pub frame_threshold: f32,
    pub offset_threshold: f32,
    pub min_note_ms: f64,
    pub merge_gap_ms: f64,
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref())?;
        let cfg: Self = toml::from_str(&text).context("parse toml")?;
        Ok(cfg)
    }

    pub fn chunk_frames(&self) -> usize {
        let fps = self.audio.sample_rate as f64 / self.audio.hop_length as f64;
        (self.train.chunk_sec * fps).round() as usize
    }

    pub fn mel_config(&self) -> denote_audio::MelConfig {
        denote_audio::MelConfig {
            sample_rate: self.audio.sample_rate,
            n_fft: self.audio.n_fft,
            hop_length: self.audio.hop_length,
            n_mels: self.audio.n_mels,
            fmin: self.audio.fmin,
            fmax: self.audio.fmax,
            mel_eps: self.audio.mel_eps,
        }
    }

    pub fn label_config(&self) -> denote_data::LabelConfig {
        denote_data::LabelConfig {
            n_pitches: self.model.n_pitches,
            midi_offset: self.model.midi_offset,
            sample_rate: self.audio.sample_rate,
            hop_length: self.audio.hop_length,
            onset_smear_frames: self.labels.onset_smear_frames,
            frame_onset_boost_frames: self.labels.frame_onset_boost_frames,
            frame_onset_boost: self.labels.frame_onset_boost,
        }
    }

    /// Decode thresholds; `min_note_ms` falls back from `labels.min_note_frames` if needed.
    pub fn decode_config(&self) -> denote_decode::DecodeConfig {
        let spf = self.audio.hop_length as f64 / self.audio.sample_rate as f64;
        let from_frames_ms = self.labels.min_note_frames as f64 * spf * 1000.0;
        let min_note_ms = if self.decode.min_note_ms > 0.0 {
            self.decode.min_note_ms
        } else {
            from_frames_ms
        };
        // Prefer the stricter of config ms vs label frame floor.
        let min_note_ms = min_note_ms.max(from_frames_ms * 0.5);

        denote_decode::DecodeConfig {
            sample_rate: self.audio.sample_rate,
            hop_length: self.audio.hop_length,
            midi_offset: self.model.midi_offset,
            onset_threshold: self.decode.onset_threshold,
            frame_threshold: self.decode.frame_threshold,
            offset_threshold: self.decode.offset_threshold,
            min_note_ms,
            merge_gap_ms: self.decode.merge_gap_ms,
        }
    }

    pub fn crnn_config(&self) -> denote_model::CrnnConfig {
        denote_model::CrnnConfig::new()
            .with_n_mels(self.audio.n_mels)
            .with_n_pitches(self.model.n_pitches)
            .with_feat_dim(self.model.feat_dim)
            .with_onset_hidden(self.model.onset_hidden)
            .with_frame_hidden(self.model.frame_hidden)
            .with_offset_hidden(self.model.offset_hidden)
            .with_dropout(self.model.dropout)
    }

    pub fn loss_weights(&self) -> denote_model::LossWeights {
        denote_model::LossWeights {
            onset: self.train.onset_weight,
            frame: self.train.frame_weight,
            offset: self.train.offset_weight,
            onset_pos_weight: self.train.onset_pos_weight,
        }
    }

    pub fn effective_batch_size(&self) -> usize {
        self.train.batch_size.max(1)
    }

    pub fn grad_accum_steps(&self) -> usize {
        self.train.grad_accum.max(1)
    }
}
