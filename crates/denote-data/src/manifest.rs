//! Cache manifest (`manifest.jsonl`) for Phase 1 L0 / Job A.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One line in `manifest.jsonl` (master §6 / Phase 1 layout).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestEntry {
    pub id: String,
    /// Relative path from cache root, e.g. `latents/clip_0001.lat.bin`.
    pub latent: String,
    /// Relative path from cache root, e.g. `tokens/clip_0001.tok.bin`.
    pub tokens: String,
    /// Active challenge program indices (K=8).
    pub programs: Vec<u8>,
    pub n_instruments: usize,
    /// `"train"` | `"val"` | `"test"`.
    pub split: String,
    #[serde(default)]
    pub duration_sec: f64,
    #[serde(default = "default_sr")]
    pub sample_rate: u32,
    #[serde(default)]
    pub d_model: usize,
    #[serde(default)]
    pub n_frames: usize,
    /// Optional source audio path (for L0 rebuild / debug).
    #[serde(default)]
    pub audio_path: Option<String>,
    /// Optional notes JSON path.
    #[serde(default)]
    pub notes_path: Option<String>,
    /// Optional per-group token paths: group id string → relative path.
    #[serde(default)]
    pub tokens_by_group: std::collections::HashMap<String, String>,
    /// Groups present in the clip (optional, Phase 2).
    #[serde(default)]
    pub groups_present: Vec<u8>,
}

fn default_sr() -> u32 {
    24_000
}

impl ManifestEntry {
    pub fn multi_hot(&self, k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; k];
        for &p in &self.programs {
            let i = p as usize;
            if i < k {
                y[i] = 1.0;
            }
        }
        y
    }
}

/// Read/write helper for a cache root that contains `manifest.jsonl`.
#[derive(Debug, Clone)]
pub struct CacheManifest {
    pub root: PathBuf,
    pub path: PathBuf,
}

impl CacheManifest {
    pub fn new(cache_root: impl Into<PathBuf>) -> Self {
        let root = cache_root.into();
        let path = root.join("manifest.jsonl");
        Self { root, path }
    }

    pub fn from_path(manifest_path: impl Into<PathBuf>) -> Self {
        let path = manifest_path.into();
        let root = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        Self { root, path }
    }

    pub fn ensure_parent(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(())
    }

    /// Append one entry (creates file if missing).
    pub fn append(&self, entry: &ManifestEntry) -> Result<()> {
        self.ensure_parent()?;
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("open {}", self.path.display()))?;
        let mut w = BufWriter::new(f);
        let line = serde_json::to_string(entry).context("serialize manifest entry")?;
        writeln!(w, "{line}").context("write manifest line")?;
        w.flush()?;
        Ok(())
    }

    /// Overwrite entire manifest.
    pub fn write_all(&self, entries: &[ManifestEntry]) -> Result<()> {
        self.ensure_parent()?;
        let f = File::create(&self.path).with_context(|| format!("create {}", self.path.display()))?;
        let mut w = BufWriter::new(f);
        for e in entries {
            let line = serde_json::to_string(e).context("serialize manifest entry")?;
            writeln!(w, "{line}")?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(&self) -> Result<Vec<ManifestEntry>> {
        if !self.path.exists() {
            return Ok(vec![]);
        }
        let f = File::open(&self.path).with_context(|| format!("open {}", self.path.display()))?;
        let reader = BufReader::new(f);
        let mut out = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("read line {} of {}", i + 1, self.path.display()))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let e: ManifestEntry = serde_json::from_str(line).with_context(|| {
                format!("parse manifest line {} in {}", i + 1, self.path.display())
            })?;
            out.push(e);
        }
        Ok(out)
    }

    pub fn list_by_split(&self, split: &str) -> Result<Vec<ManifestEntry>> {
        let all = self.load()?;
        Ok(all.into_iter().filter(|e| e.split == split).collect())
    }

    pub fn resolve(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
}

/// Stable train/val split by hashing id (80/20 default).
pub fn split_for_id(id: &str, train_ratio: f64) -> &'static str {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in id.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    // Avoid u64→f64 precision collapse; use fixed buckets.
    let bucket = (h % 10_000) as f64 / 10_000.0;
    if bucket < train_ratio.clamp(0.0, 1.0) {
        "train"
    } else {
        "val"
    }
}

/// Build multi-hot presence vector from program list (K programs).
pub fn programs_multi_hot(programs: &[u8], k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; k];
    for &p in programs {
        let i = p as usize;
        if i < k {
            y[i] = 1.0;
        }
    }
    y
}

/// Challenge K=8 program names (master §2.6 / Phase 1 §5.2).
pub const CHALLENGE_PROGRAM_NAMES: [&str; 8] = [
    "Piano",
    "Violin",
    "Viola",
    "Cello",
    "Flute",
    "Bassoon",
    "Trombone",
    "Oboe",
];

pub fn challenge_k() -> usize {
    CHALLENGE_PROGRAM_NAMES.len()
}

/// Unique sorted program ids from a list.
pub fn unique_programs(programs: impl IntoIterator<Item = u8>) -> Vec<u8> {
    let mut v: Vec<u8> = programs.into_iter().collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn manifest_append_load_split() {
        let dir = std::env::temp_dir().join(format!(
            "denote_manifest_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&dir);
        let m = CacheManifest::new(&dir);
        let e1 = ManifestEntry {
            id: "a".into(),
            latent: "latents/a.lat.bin".into(),
            tokens: "tokens/a.tok.bin".into(),
            programs: vec![0, 2],
            n_instruments: 2,
            split: "train".into(),
            duration_sec: 4.0,
            sample_rate: 24_000,
            d_model: 128,
            n_frames: 25,
            audio_path: None,
            notes_path: None,
            tokens_by_group: Default::default(),
            groups_present: vec![],
        };
        let e2 = ManifestEntry {
            id: "b".into(),
            latent: "latents/b.lat.bin".into(),
            tokens: "tokens/b.tok.bin".into(),
            programs: vec![1],
            n_instruments: 1,
            split: "val".into(),
            duration_sec: 4.0,
            sample_rate: 24_000,
            d_model: 128,
            n_frames: 25,
            audio_path: None,
            notes_path: None,
            tokens_by_group: Default::default(),
            groups_present: vec![],
        };
        m.append(&e1).unwrap();
        m.append(&e2).unwrap();
        let all = m.load().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(m.list_by_split("train").unwrap().len(), 1);
        assert_eq!(m.list_by_split("val").unwrap().len(), 1);
        assert_eq!(e1.multi_hot(8)[0], 1.0);
        assert_eq!(e1.multi_hot(8)[2], 1.0);
        assert_eq!(e1.multi_hot(8)[1], 0.0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_for_id_is_stable() {
        let s1 = split_for_id("clip_0001", 0.8);
        let s2 = split_for_id("clip_0001", 0.8);
        assert_eq!(s1, s2);
    }
}
