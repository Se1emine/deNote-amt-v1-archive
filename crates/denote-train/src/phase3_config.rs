//! Phase 3 TOML: corpus build (`phase3_corpus.toml`).

#![allow(dead_code)]

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// `configs/phase3_corpus.toml` — FluidR3 + BabySlakh → flat wav/notes for L0.
#[derive(Debug, Clone, Deserialize)]
pub struct Phase3CorpusConfig {
    pub run: Phase3RunSection,
    pub data: Phase3CorpusData,
    #[serde(default)]
    pub render: Phase3RenderSection,
    #[serde(default)]
    pub stem: Phase3StemSection,
    #[serde(default)]
    pub chunk: Phase3ChunkSection,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Phase3RunSection {
    #[serde(default = "default_corpus_name")]
    pub name: String,
    #[serde(default)]
    pub run_dir: Option<String>,
}

fn default_corpus_name() -> String {
    "phase3_corpus".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct Phase3CorpusData {
    /// Output flat dir: `{id}.wav` + `{id}.notes.json` for Job L0.
    pub out_dir: String,
    /// Provenance index path.
    #[serde(default = "default_render_index")]
    pub render_index: String,
    #[serde(default = "default_train_ratio")]
    pub train_ratio: f64,
    /// Repo-relative root (for bundled fluidsynth / sf2).
    #[serde(default = "default_repo")]
    pub repo_root: String,
    /// clean_midi (or any MIDI tree) for FluidR3.
    #[serde(default)]
    pub midi_dir: Option<String>,
    /// Max MIDI files to render (hard min path ~50).
    #[serde(default = "default_n_midi")]
    pub max_midi: usize,
    /// BabySlakh root (`.../babyslakh_16k`).
    #[serde(default)]
    pub babyslakh_dir: Option<String>,
    /// RNG seed for stem mixes / MIDI sample.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_render_index() -> String {
    "data/raw/phase3/render_index.jsonl".into()
}
fn default_train_ratio() -> f64 {
    0.8
}
fn default_repo() -> String {
    ".".into()
}
fn default_n_midi() -> usize {
    50
}
fn default_seed() -> u64 {
    42
}

#[derive(Debug, Clone, Deserialize)]
pub struct Phase3RenderSection {
    #[serde(default)]
    pub fluidsynth: Option<String>,
    #[serde(default)]
    pub sf2: Option<String>,
    #[serde(default = "default_sr")]
    pub sample_rate: u32,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for Phase3RenderSection {
    fn default() -> Self {
        Self {
            fluidsynth: None,
            sf2: None,
            sample_rate: default_sr(),
            enabled: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Phase3StemSection {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_mixes")]
    pub mixes_per_track: usize,
    #[serde(default = "default_max_inst")]
    pub max_instruments: usize,
    #[serde(default = "default_min_inst")]
    pub min_instruments: usize,
}

impl Default for Phase3StemSection {
    fn default() -> Self {
        Self {
            enabled: true,
            mixes_per_track: 4,
            max_instruments: 3,
            min_instruments: 1,
        }
    }
}

fn default_mixes() -> usize {
    4
}
fn default_max_inst() -> usize {
    3
}
fn default_min_inst() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub struct Phase3ChunkSection {
    #[serde(default = "default_chunk_sec")]
    pub chunk_sec: f64,
    #[serde(default = "default_chunk_sec")]
    pub hop_sec: f64,
    #[serde(default = "default_min_notes")]
    pub min_notes: usize,
    #[serde(default = "default_max_inst")]
    pub max_instruments: usize,
}

impl Default for Phase3ChunkSection {
    fn default() -> Self {
        Self {
            chunk_sec: 4.0,
            hop_sec: 4.0,
            min_notes: 1,
            max_instruments: 3,
        }
    }
}

fn default_chunk_sec() -> f64 {
    4.0
}
fn default_min_notes() -> usize {
    1
}
fn default_sr() -> u32 {
    24_000
}
fn default_true() -> bool {
    true
}

impl Phase3CorpusConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read {}", path.as_ref().display()))?;
        toml::from_str(&text).context("parse phase3 corpus toml")
    }
}
