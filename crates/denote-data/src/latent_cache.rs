//! Latent feature cache for Job L0 / Job A (MusicFM encoder output on disk).
//!
//! File format (`.lat.bin`):
//! ```text
//! magic: b"DNL1" (4)
//! version: u32 LE (=1)
//! n_frames: u32 LE (T)
//! d_model: u32 LE (D)
//! dtype: u32 LE (0 = f32)
//! body: T*D f32 LE row-major [t0_d0..t0_dD, t1_d0, ...]
//! ```

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Metadata for one latent sample (also stored in header + optional meta JSON).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatentMeta {
    pub id: String,
    pub n_frames: usize,
    pub d_model: usize,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default)]
    pub duration_sec: f64,
}

fn default_sample_rate() -> u32 {
    24_000
}

/// One precomputed latent clip: `z` is row-major `[T * D]` f32.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentSample {
    pub meta: LatentMeta,
    pub z: Vec<f32>,
}

impl LatentSample {
    pub fn new(id: impl Into<String>, n_frames: usize, d_model: usize, z: Vec<f32>) -> Result<Self> {
        if z.len() != n_frames * d_model {
            bail!(
                "latent size mismatch: z.len()={} expected T*D={}*{}={}",
                z.len(),
                n_frames,
                d_model,
                n_frames * d_model
            );
        }
        Ok(Self {
            meta: LatentMeta {
                id: id.into(),
                n_frames,
                d_model,
                sample_rate: default_sample_rate(),
                duration_sec: 0.0,
            },
            z,
        })
    }

    pub fn with_audio_meta(mut self, sample_rate: u32, duration_sec: f64) -> Self {
        self.meta.sample_rate = sample_rate;
        self.meta.duration_sec = duration_sec;
        self
    }
}

const MAGIC: &[u8; 4] = b"DNL1";
const VERSION: u32 = 1;
const DTYPE_F32: u32 = 0;

/// Directory cache for latents (`data/cache/.../latents/`).
pub struct LatentFeatureCache {
    pub root: PathBuf,
}

impl LatentFeatureCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn ensure_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        Ok(())
    }

    pub fn sample_path(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.lat.bin"))
    }

    pub fn list_ids(&self) -> Result<Vec<String>> {
        if !self.root.exists() {
            return Ok(vec![]);
        }
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if let Some(stem) = name.strip_suffix(".lat.bin") {
                ids.push(stem.to_string());
            }
        }
        ids.sort();
        Ok(ids)
    }

    pub fn save(&self, sample: &LatentSample) -> Result<()> {
        self.ensure_dir()?;
        let path = self.sample_path(&sample.meta.id);
        write_latent_bin(&path, sample)?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<LatentSample> {
        let path = self.sample_path(id);
        let mut sample = read_latent_bin(&path)?;
        // Prefer path stem if header id empty (should not happen).
        if sample.meta.id.is_empty() {
            sample.meta.id = id.to_string();
        }
        Ok(sample)
    }
}

fn write_latent_bin(path: &Path, sample: &LatentSample) -> Result<()> {
    let expected = sample.meta.n_frames * sample.meta.d_model;
    if sample.z.len() != expected {
        bail!(
            "cannot write {}: z.len()={} != T*D={}",
            path.display(),
            sample.z.len(),
            expected
        );
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::new(f);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(sample.meta.n_frames as u32).to_le_bytes())?;
    w.write_all(&(sample.meta.d_model as u32).to_le_bytes())?;
    w.write_all(&DTYPE_F32.to_le_bytes())?;
    for &v in &sample.z {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

fn read_latent_bin(path: &Path) -> Result<LatentSample> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .with_context(|| format!("read magic {}", path.display()))?;
    if &magic != MAGIC {
        bail!(
            "bad latent magic in {}: got {:?} expected {:?}",
            path.display(),
            magic,
            MAGIC
        );
    }
    let version = read_u32_le(&mut r)?;
    if version != VERSION {
        bail!(
            "unsupported latent version {} in {} (want {})",
            version,
            path.display(),
            VERSION
        );
    }
    let n_frames = read_u32_le(&mut r)? as usize;
    let d_model = read_u32_le(&mut r)? as usize;
    let dtype = read_u32_le(&mut r)?;
    if dtype != DTYPE_F32 {
        bail!("unsupported dtype {} in {}", dtype, path.display());
    }
    let n = n_frames * d_model;
    let mut z = Vec::with_capacity(n);
    let mut buf = [0u8; 4];
    for _ in 0..n {
        r.read_exact(&mut buf)
            .with_context(|| format!("read body {}", path.display()))?;
        z.push(f32::from_le_bytes(buf));
    }
    let id = path
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(|n| n.strip_suffix(".lat.bin"))
        .unwrap_or("unknown")
        .to_string();
    Ok(LatentSample {
        meta: LatentMeta {
            id,
            n_frames,
            d_model,
            sample_rate: default_sample_rate(),
            duration_sec: 0.0,
        },
        z,
    })
}

fn read_u32_le(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn latent_cache_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "denote_lat_cache_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&dir);
        let cache = LatentFeatureCache::new(&dir);
        let t = 10usize;
        let d = 8usize;
        let z: Vec<f32> = (0..t * d).map(|i| i as f32 * 0.1).collect();
        let sample = LatentSample::new("clip_0001", t, d, z.clone())
            .unwrap()
            .with_audio_meta(24_000, 0.4);
        cache.save(&sample).unwrap();
        let loaded = cache.load("clip_0001").unwrap();
        assert_eq!(loaded.meta.id, "clip_0001");
        assert_eq!(loaded.meta.n_frames, t);
        assert_eq!(loaded.meta.d_model, d);
        assert_eq!(loaded.z, z);
        let ids = cache.list_ids().unwrap();
        assert_eq!(ids, vec!["clip_0001".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn latent_size_mismatch_errors() {
        let err = LatentSample::new("x", 2, 3, vec![1.0, 2.0]).unwrap_err();
        assert!(err.to_string().contains("mismatch"));
    }
}
