//! Phase 1 TOML configs: `free_1070_cache.toml`, `free_1070_jobA.toml`.

#![allow(dead_code)] // config mirrors TOML surface; not every field is read yet

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct CacheJobConfig {
    pub run: RunSection,
    pub data: CacheDataSection,
    pub encoder: EncoderSection,
    #[serde(default)]
    pub vocab: VocabSection,
    #[serde(default)]
    pub audio: AudioSection,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobAConfig {
    pub run: RunSection,
    pub data: JobADataSection,
    pub model: JobAModelSection,
    pub optim: OptimSection,
    #[serde(default)]
    pub log: LogSection,
    #[serde(default)]
    pub vocab: VocabSection,
    /// Phase 3: curriculum by n_instruments (1→2→3).
    #[serde(default)]
    pub curriculum: CurriculumToml,
}

/// Optional curriculum schedule (Phase 3). Disabled by default for Phase 1 configs.
#[derive(Debug, Clone, Deserialize)]
pub struct CurriculumToml {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_curr_s1")]
    pub stage1_end: u64,
    #[serde(default = "default_curr_s2")]
    pub stage2_end: u64,
    #[serde(default = "default_curr_max")]
    pub max_instruments: usize,
}

impl Default for CurriculumToml {
    fn default() -> Self {
        Self {
            enabled: false,
            stage1_end: default_curr_s1(),
            stage2_end: default_curr_s2(),
            max_instruments: default_curr_max(),
        }
    }
}

fn default_curr_s1() -> u64 {
    500
}
fn default_curr_s2() -> u64 {
    1500
}
fn default_curr_max() -> usize {
    3
}

impl CurriculumToml {
    pub fn schedule(&self) -> denote_data::CurriculumSchedule {
        denote_data::CurriculumSchedule {
            stage1_end: self.stage1_end,
            stage2_end: self.stage2_end,
            max_instruments_cap: self.max_instruments.max(1),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RunSection {
    #[serde(default = "default_run_name")]
    pub name: String,
    #[serde(default)]
    pub run_dir: Option<String>,
}

fn default_run_name() -> String {
    "jobA_presence".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheDataSection {
    /// Directory of raw clips: `{id}.wav` + `{id}.notes.json` (or list JSON).
    pub input_dir: String,
    /// Cache root (manifest + latents/ + tokens/).
    pub cache_dir: String,
    #[serde(default = "default_manifest_name")]
    pub manifest_name: String,
    #[serde(default = "default_train_ratio")]
    pub train_ratio: f64,
    #[serde(default = "default_chunk_sec")]
    pub chunk_sec: f64,
    /// Phase 3: skip clips that already have latents+tokens on disk.
    #[serde(default)]
    pub skip_existing: bool,
}

fn default_manifest_name() -> String {
    "manifest.jsonl".into()
}
fn default_train_ratio() -> f64 {
    0.8
}
fn default_chunk_sec() -> f64 {
    4.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobADataSection {
    pub latent_manifest: String,
    #[serde(default = "default_batch")]
    pub batch_size: usize,
    #[serde(default = "default_chunk_sec")]
    pub chunk_sec: f64,
}

fn default_batch() -> usize {
    2
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderSection {
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,
    #[serde(default = "default_conv_dim")]
    pub conv_dim: usize,
    #[serde(default = "default_d_model")]
    pub encoder_dim: usize,
    #[serde(default = "default_depth")]
    pub encoder_depth: usize,
    #[serde(default = "default_heads")]
    pub n_heads: usize,
    #[serde(default = "default_ffn")]
    pub ffn_mult: usize,
    #[serde(default)]
    pub dropout: f64,
    /// Optional path to `assets/musicfm_msd` (numpy dump). Empty = random smoke init.
    #[serde(default)]
    pub weights_dir: Option<String>,
    /// If true and weights missing, still run with random encoder (pipeline smoke).
    #[serde(default = "default_true")]
    pub allow_random_encoder: bool,
}

fn default_n_mels() -> usize {
    128
}
fn default_conv_dim() -> usize {
    64
}
fn default_d_model() -> usize {
    128
}
fn default_depth() -> usize {
    2
}
fn default_heads() -> usize {
    4
}
fn default_ffn() -> usize {
    2
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobAModelSection {
    #[serde(default = "default_true")]
    pub use_latent_cache: bool,
    #[serde(default = "default_true")]
    pub freeze_encoder: bool,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_presence_k")]
    pub presence_k: usize,
    /// `presence_adapter` | `presence_adapter_decoder`
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_n_programs")]
    pub n_programs: usize,
}

fn default_presence_k() -> usize {
    8
}
fn default_mode() -> String {
    "presence_adapter".into()
}
fn default_n_programs() -> usize {
    8
}

#[derive(Debug, Clone, Deserialize)]
pub struct OptimSection {
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default = "default_wd")]
    pub weight_decay: f64,
    #[serde(default = "default_max_steps")]
    pub max_steps: u64,
    #[serde(default = "default_ckpt_every")]
    pub ckpt_every: u64,
    #[serde(default = "default_w_presence")]
    pub w_presence: f32,
    #[serde(default)]
    pub w_aux: f32,
    #[serde(default)]
    pub w_ce: f32,
}

fn default_lr() -> f64 {
    3e-4
}
fn default_wd() -> f64 {
    0.01
}
fn default_max_steps() -> u64 {
    2000
}
fn default_ckpt_every() -> u64 {
    500
}
fn default_w_presence() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LogSection {
    #[serde(default = "default_true")]
    pub jsonl: bool,
    #[serde(default = "default_log_every")]
    pub log_every: u64,
    #[serde(default = "default_val_every")]
    pub val_every: u64,
    /// Cap validation forward passes (anti-OOM / anti-leak on Autodiff). 0 = no cap.
    #[serde(default = "default_max_val_batches")]
    pub max_val_batches: usize,
}

fn default_log_every() -> u64 {
    10
}
fn default_val_every() -> u64 {
    50
}
fn default_max_val_batches() -> usize {
    32
}

#[derive(Debug, Clone, Deserialize)]
pub struct VocabSection {
    #[serde(default = "default_time_bin")]
    pub time_bin_sec: f64,
    #[serde(default = "default_n_time")]
    pub n_time_bins: u32,
    #[serde(default = "default_128u")]
    pub n_pitches: u32,
    #[serde(default = "default_128u")]
    pub n_velocity: u32,
    #[serde(default = "default_8u")]
    pub n_programs: u32,
    #[serde(default = "default_poly")]
    pub max_polyphony: u32,
}

impl Default for VocabSection {
    fn default() -> Self {
        Self {
            time_bin_sec: default_time_bin(),
            n_time_bins: default_n_time(),
            n_pitches: default_128u(),
            n_velocity: default_128u(),
            n_programs: default_8u(),
            max_polyphony: default_poly(),
        }
    }
}

fn default_time_bin() -> f64 {
    0.01
}
fn default_n_time() -> u32 {
    205
}
fn default_128u() -> u32 {
    128
}
fn default_8u() -> u32 {
    8
}
fn default_poly() -> u32 {
    16
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioSection {
    #[serde(default = "default_sr")]
    pub sample_rate: u32,
}

impl Default for AudioSection {
    fn default() -> Self {
        Self {
            sample_rate: default_sr(),
        }
    }
}

fn default_sr() -> u32 {
    24_000
}

impl CacheJobConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read {}", path.as_ref().display()))?;
        toml::from_str(&text).context("parse cache job toml")
    }

    pub fn encoder_config(&self) -> denote_encoder::MusicFmEncoderConfig {
        denote_encoder::MusicFmEncoderConfig {
            n_mels: self.encoder.n_mels,
            conv_dim: self.encoder.conv_dim,
            encoder_dim: self.encoder.encoder_dim,
            encoder_depth: self.encoder.encoder_depth,
            n_heads: self.encoder.n_heads,
            ffn_mult: self.encoder.ffn_mult,
            dropout: self.encoder.dropout,
            temporal_downsample: 4,
        }
    }

    pub fn vocab(&self) -> denote_tokens::Vocab {
        use denote_tokens::{Vocab, VocabConfig};
        Vocab::new(VocabConfig {
            time_bin_sec: self.vocab.time_bin_sec,
            n_time_bins: self.vocab.n_time_bins,
            n_pitches: self.vocab.n_pitches,
            n_velocity: self.vocab.n_velocity,
            n_programs: self.vocab.n_programs,
            max_polyphony: self.vocab.max_polyphony as usize,
        })
    }
}

impl JobAConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read {}", path.as_ref().display()))?;
        toml::from_str(&text).context("parse jobA toml")
    }

    pub fn vocab(&self) -> denote_tokens::Vocab {
        use denote_tokens::{Vocab, VocabConfig};
        Vocab::new(VocabConfig {
            time_bin_sec: self.vocab.time_bin_sec,
            n_time_bins: self.vocab.n_time_bins,
            n_pitches: self.vocab.n_pitches,
            n_velocity: self.vocab.n_velocity,
            n_programs: self.vocab.n_programs,
            max_polyphony: self.vocab.max_polyphony as usize,
        })
    }
}
