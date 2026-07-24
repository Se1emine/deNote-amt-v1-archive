use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::labels::{notes_to_targets, FrameTargets, LabelConfig};
use crate::note::NoteList;
use denote_audio::{log_mel_spectrogram, MelConfig};

/// Metadata for a cached training sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleMeta {
    pub track_id: String,
    pub n_frames: usize,
    pub n_mels: usize,
    pub n_pitches: usize,
    pub sample_rate: u32,
    pub hop_length: usize,
}

/// One precomputed sample: log-mel + frame targets.
#[derive(Debug, Clone)]
pub struct CachedSample {
    pub meta: SampleMeta,
    /// `[n_mels, T]`
    pub mel: Array2<f32>,
    pub targets: FrameTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SampleDisk {
    meta: SampleMeta,
    mel: Vec<f32>,
    onset: Vec<f32>,
    frame: Vec<f32>,
    offset: Vec<f32>,
    frame_weight: Vec<f32>,
}

/// Directory-based feature cache (`data/cache/...`).
pub struct FeatureCache {
    pub root: PathBuf,
}

impl FeatureCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn ensure_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        Ok(())
    }

    pub fn sample_path(&self, track_id: &str) -> PathBuf {
        self.root.join(format!("{track_id}.bin"))
    }

    pub fn list_ids(&self) -> Result<Vec<String>> {
        if !self.root.exists() {
            return Ok(vec![]);
        }
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("bin") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    ids.push(stem.to_string());
                }
            }
        }
        ids.sort();
        Ok(ids)
    }

    pub fn save(&self, sample: &CachedSample) -> Result<()> {
        self.ensure_dir()?;
        let disk = SampleDisk {
            meta: sample.meta.clone(),
            mel: sample.mel.as_slice().unwrap().to_vec(),
            onset: sample.targets.onset.as_slice().unwrap().to_vec(),
            frame: sample.targets.frame.as_slice().unwrap().to_vec(),
            offset: sample.targets.offset.as_slice().unwrap().to_vec(),
            frame_weight: sample.targets.frame_weight.as_slice().unwrap().to_vec(),
        };
        let path = self.sample_path(&sample.meta.track_id);
        let f = File::create(&path).with_context(|| format!("create {}", path.display()))?;
        bincode::serialize_into(BufWriter::new(f), &disk)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    pub fn load(&self, track_id: &str) -> Result<CachedSample> {
        let path = self.sample_path(track_id);
        let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let disk: SampleDisk = bincode::deserialize_from(BufReader::new(f))
            .with_context(|| format!("read {}", path.display()))?;
        let t = disk.meta.n_frames;
        let m = disk.meta.n_mels;
        let p = disk.meta.n_pitches;
        Ok(CachedSample {
            meta: disk.meta,
            mel: Array2::from_shape_vec((m, t), disk.mel)?,
            targets: FrameTargets {
                onset: Array2::from_shape_vec((t, p), disk.onset)?,
                frame: Array2::from_shape_vec((t, p), disk.frame)?,
                offset: Array2::from_shape_vec((t, p), disk.offset)?,
                frame_weight: Array2::from_shape_vec((t, p), disk.frame_weight)?,
            },
        })
    }

    /// Build cache entry from mono audio + note list.
    pub fn build_from_audio(
        track_id: &str,
        audio: &[f32],
        notes: &NoteList,
        mel_cfg: &MelConfig,
        label_cfg: &LabelConfig,
    ) -> CachedSample {
        let mel = log_mel_spectrogram(audio, mel_cfg);
        let n_frames = mel.n_frames();
        let targets = notes_to_targets(&notes.notes, n_frames, label_cfg);
        CachedSample {
            meta: SampleMeta {
                track_id: track_id.to_string(),
                n_frames,
                n_mels: mel.n_mels(),
                n_pitches: label_cfg.n_pitches,
                sample_rate: mel_cfg.sample_rate,
                hop_length: mel_cfg.hop_length,
            },
            mel: mel.data,
            targets,
        }
    }

    /// Random time crop of length `chunk_frames` (or full if shorter).
    pub fn crop(
        sample: &CachedSample,
        chunk_frames: usize,
        start_frame: usize,
    ) -> (Array2<f32>, FrameTargets) {
        let t = sample.meta.n_frames;
        let m = sample.meta.n_mels;
        let p = sample.meta.n_pitches;
        if t <= chunk_frames {
            return (sample.mel.clone(), sample.targets.clone());
        }
        let start = start_frame.min(t - chunk_frames);
        let end = start + chunk_frames;
        let mel = sample.mel.slice(ndarray::s![.., start..end]).to_owned();
        let onset = sample.targets.onset.slice(ndarray::s![start..end, ..]).to_owned();
        let frame = sample.targets.frame.slice(ndarray::s![start..end, ..]).to_owned();
        let offset = sample.targets.offset.slice(ndarray::s![start..end, ..]).to_owned();
        let frame_weight = sample
            .targets
            .frame_weight
            .slice(ndarray::s![start..end, ..])
            .to_owned();
        let _ = (m, p);
        (
            mel,
            FrameTargets {
                onset,
                frame,
                offset,
                frame_weight,
            },
        )
    }
}

/// Save a note list next to cache for debugging.
pub fn save_notes_json(path: impl AsRef<Path>, notes: &NoteList) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(notes)?;
    fs::write(path, text)?;
    Ok(())
}
