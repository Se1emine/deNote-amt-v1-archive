//! Seq2seq token sample cache for Denote-MIROS (primary training path).

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::note::{NoteAnnotation, NoteList};
use denote_tokens::{encode_notes, TokenId, TokenNote, Vocab, VocabConfig};

/// Metadata for a tokenized training sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSampleMeta {
    pub track_id: String,
    /// Audio sample rate used when the clip was recorded / will be loaded.
    pub sample_rate: u32,
    /// Duration of the clip in seconds (for windowing).
    pub duration_sec: f64,
    /// Number of token ids (including BOS/EOS, excluding pad).
    pub n_tokens: usize,
    /// Vocab fingerprint so train fails fast on mismatch.
    pub vocab_n_time_bins: u32,
    pub vocab_n_pitches: u32,
    pub vocab_n_velocity: u32,
    pub vocab_n_programs: u32,
    pub time_bin_sec: f64,
}

/// One precomputed seq2seq sample: token ids (+ optional path to audio).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenCachedSample {
    pub meta: TokenSampleMeta,
    /// Full sequence: BOS … EOS (no PAD).
    pub token_ids: Vec<TokenId>,
    /// Optional relative path or absolute path to source audio.
    #[serde(default)]
    pub audio_path: Option<String>,
    /// Optional raw mono PCM (smoke / small clips only).
    #[serde(default)]
    pub audio_pcm: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenSampleDisk {
    meta: TokenSampleMeta,
    token_ids: Vec<TokenId>,
    audio_path: Option<String>,
    audio_pcm: Option<Vec<f32>>,
}

/// Directory cache for tokenized samples (`data/cache/tokens/...`).
pub struct TokenFeatureCache {
    pub root: PathBuf,
}

impl TokenFeatureCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn ensure_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        Ok(())
    }

    pub fn sample_path(&self, track_id: &str) -> PathBuf {
        self.root.join(format!("{track_id}.tok.bin"))
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
            if let Some(stem) = name.strip_suffix(".tok.bin") {
                ids.push(stem.to_string());
            }
        }
        ids.sort();
        Ok(ids)
    }

    pub fn save(&self, sample: &TokenCachedSample) -> Result<()> {
        self.ensure_dir()?;
        let disk = TokenSampleDisk {
            meta: sample.meta.clone(),
            token_ids: sample.token_ids.clone(),
            audio_path: sample.audio_path.clone(),
            audio_pcm: sample.audio_pcm.clone(),
        };
        let path = self.sample_path(&sample.meta.track_id);
        let f = File::create(&path).with_context(|| format!("create {}", path.display()))?;
        bincode::serialize_into(BufWriter::new(f), &disk)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    pub fn load(&self, track_id: &str) -> Result<TokenCachedSample> {
        let path = self.sample_path(track_id);
        let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let disk: TokenSampleDisk = bincode::deserialize_from(BufReader::new(f))
            .with_context(|| format!("read {}", path.display()))?;
        Ok(TokenCachedSample {
            meta: disk.meta,
            token_ids: disk.token_ids,
            audio_path: disk.audio_path,
            audio_pcm: disk.audio_pcm,
        })
    }

    /// Build a token sample from notes (+ optional audio bytes).
    pub fn build_from_notes(
        track_id: &str,
        notes: &NoteList,
        vocab: &Vocab,
        sample_rate: u32,
        duration_sec: f64,
        audio_path: Option<String>,
        audio_pcm: Option<Vec<f32>>,
    ) -> TokenCachedSample {
        let token_notes: Vec<TokenNote> = notes
            .notes
            .iter()
            .map(note_annotation_to_token_note)
            .collect();
        let token_ids = encode_notes(&token_notes, vocab);
        let duration = if duration_sec > 0.0 {
            duration_sec
        } else {
            notes
                .notes
                .iter()
                .map(|n| n.end_sec)
                .fold(0.0_f64, f64::max)
        };
        TokenCachedSample {
            meta: TokenSampleMeta {
                track_id: track_id.to_string(),
                sample_rate,
                duration_sec: duration,
                n_tokens: token_ids.len(),
                vocab_n_time_bins: vocab.cfg.n_time_bins,
                vocab_n_pitches: vocab.cfg.n_pitches,
                vocab_n_velocity: vocab.cfg.n_velocity,
                vocab_n_programs: vocab.cfg.n_programs,
                time_bin_sec: vocab.cfg.time_bin_sec,
            },
            token_ids,
            audio_path,
            audio_pcm,
        }
    }
}

pub fn note_annotation_to_token_note(n: &NoteAnnotation) -> TokenNote {
    TokenNote {
        start_sec: n.start_sec,
        end_sec: n.end_sec,
        pitch_midi: n.pitch_midi,
        velocity: if n.velocity == 0 { 80 } else { n.velocity },
        program: n.program,
    }
}

pub fn token_note_to_annotation(n: &TokenNote) -> NoteAnnotation {
    NoteAnnotation {
        start_sec: n.start_sec,
        end_sec: n.end_sec,
        pitch_midi: n.pitch_midi,
        velocity: n.velocity,
        confidence: 1.0,
        program: n.program,
    }
}

/// Reconstruct vocab config stored in sample meta (for load-time checks).
pub fn vocab_config_from_meta(meta: &TokenSampleMeta) -> VocabConfig {
    VocabConfig {
        time_bin_sec: meta.time_bin_sec,
        n_time_bins: meta.vocab_n_time_bins,
        n_pitches: meta.vocab_n_pitches,
        n_velocity: meta.vocab_n_velocity,
        n_programs: meta.vocab_n_programs,
        max_polyphony: 16,
    }
}

/// Save token ids as JSON list (debug).
pub fn save_tokens_json(path: impl AsRef<Path>, token_ids: &[TokenId]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(token_ids)?;
    fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use denote_tokens::{decode_tokens, sequence_is_valid, Vocab};

    #[test]
    fn token_cache_roundtrip() {
        let dir = std::env::temp_dir().join(format!("denote_tok_cache_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cache = TokenFeatureCache::new(&dir);
        let vocab = Vocab::default_mt3_like();
        let notes = NoteList {
            track_id: "t1".into(),
            sr: 24_000,
            notes: vec![NoteAnnotation {
                start_sec: 0.1,
                end_sec: 0.5,
                pitch_midi: 60,
                velocity: 100,
                confidence: 1.0,
                program: 0,
            }],
        };
        let sample = TokenFeatureCache::build_from_notes(
            "t1",
            &notes,
            &vocab,
            24_000,
            1.0,
            None,
            Some(vec![0.0f32; 24000]),
        );
        assert!(sequence_is_valid(&sample.token_ids, &vocab));
        cache.save(&sample).unwrap();
        let loaded = cache.load("t1").unwrap();
        assert_eq!(loaded.token_ids, sample.token_ids);
        let back = decode_tokens(&loaded.token_ids, &vocab);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].pitch_midi, 60);
        let _ = fs::remove_dir_all(&dir);
    }
}
