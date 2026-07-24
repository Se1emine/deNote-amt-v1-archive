//! Phase 2 TOML configs: Job B_g / single-decoder ablation.

#![allow(dead_code)]

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::phase1_config::{CurriculumToml, LogSection, VocabSection};

#[derive(Debug, Clone, Deserialize)]
pub struct JobBConfig {
    pub run: JobBRunSection,
    pub data: JobBDataSection,
    pub model: JobBModelSection,
    pub optim: JobBOptimSection,
    #[serde(default)]
    pub log: LogSection,
    #[serde(default)]
    pub vocab: VocabSection,
    #[serde(default)]
    pub groups: JobBGroupsSection,
    /// Phase 3 curriculum (n_instruments 1→2→3).
    #[serde(default)]
    pub curriculum: CurriculumToml,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobBRunSection {
    #[serde(default = "default_jobb_name")]
    pub name: String,
    #[serde(default)]
    pub run_dir: Option<String>,
    /// Path to Job A modular checkpoint dir (`.../checkpoints/last`).
    pub job_a_ckpt: String,
}

fn default_jobb_name() -> String {
    "jobB_g0".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobBDataSection {
    pub latent_manifest: String,
    /// Group id to train (ignored if `single_decoder = true`).
    #[serde(default)]
    pub group_id: u32,
    #[serde(default = "default_batch")]
    pub batch_size: usize,
    #[serde(default = "default_max_token_len")]
    pub max_token_len: usize,
    #[serde(default = "default_chunk_sec")]
    pub chunk_sec: f64,
}

fn default_batch() -> usize {
    2
}
fn default_max_token_len() -> usize {
    256
}
fn default_chunk_sec() -> f64 {
    4.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobBModelSection {
    #[serde(default = "default_true")]
    pub freeze_adapter: bool,
    #[serde(default = "default_true")]
    pub freeze_presence: bool,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_presence_k")]
    pub presence_k: usize,
    #[serde(default = "default_n_programs")]
    pub n_programs: usize,
    #[serde(default = "default_decoder_layers")]
    pub decoder_layers: usize,
    #[serde(default = "default_decoder_heads")]
    pub decoder_heads: usize,
    #[serde(default = "default_decoder_ff_mult")]
    pub decoder_ff_mult: usize,
    #[serde(default)]
    pub decoder_dropout: f64,
    /// Ablation: one decoder on full multi-program tokens (not per-group).
    #[serde(default)]
    pub single_decoder: bool,
}

fn default_true() -> bool {
    true
}
fn default_d_model() -> usize {
    128
}
fn default_presence_k() -> usize {
    8
}
fn default_n_programs() -> usize {
    8
}
fn default_decoder_layers() -> usize {
    2
}
fn default_decoder_heads() -> usize {
    4
}
fn default_decoder_ff_mult() -> usize {
    2
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobBOptimSection {
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default = "default_wd")]
    pub weight_decay: f64,
    #[serde(default = "default_max_steps")]
    pub max_steps: u64,
    #[serde(default = "default_ckpt_every")]
    pub ckpt_every: u64,
    #[serde(default)]
    pub label_smoothing: f32,
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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct JobBGroupsSection {
    /// Path to groups TOML (default challenge8).
    #[serde(default = "default_groups_path")]
    pub path: String,
}

fn default_groups_path() -> String {
    "configs/groups_challenge8.toml".into()
}

impl JobBConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read {}", path.as_ref().display()))?;
        toml::from_str(&text).context("parse jobB toml")
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
