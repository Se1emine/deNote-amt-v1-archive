//! Time-window chunking of long wav+notes into train clips (Phase 3).

use crate::note::{NoteAnnotation, NoteList};
use crate::programs::gm_to_challenge;
use crate::manifest::unique_programs;

/// One chunk of audio + notes with relative times in `[0, chunk_sec]`.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub id: String,
    /// Sample slice of mono audio at `sample_rate`.
    pub audio: Vec<f32>,
    pub notes: NoteList,
    pub programs: Vec<u8>,
    pub n_instruments: usize,
    pub start_sec: f64,
    pub duration_sec: f64,
    /// Piece-level id for train/val split (no leakage).
    pub piece_id: String,
}

/// Non-overlapping (or hop-controlled) windows.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub chunk_sec: f64,
    /// Hop between chunk starts; default = chunk_sec (no overlap).
    pub hop_sec: f64,
    pub sample_rate: u32,
    /// Drop chunks with fewer than this many notes after filter.
    pub min_notes: usize,
    /// Cap instruments per chunk (challenge-like ≤3).
    pub max_instruments: usize,
    /// If true, remap GM programs to challenge K before counting.
    pub remap_gm: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_sec: 4.0,
            hop_sec: 4.0,
            sample_rate: 24_000,
            min_notes: 1,
            max_instruments: 3,
            remap_gm: true,
        }
    }
}

/// Slice notes whose onset falls in `[t0, t1)` and shift times by `-t0`.
/// Notes spanning the boundary keep clipped end if they started in window.
pub fn slice_notes(notes: &[NoteAnnotation], t0: f64, t1: f64) -> Vec<NoteAnnotation> {
    let mut out = Vec::new();
    for n in notes {
        if n.start_sec >= t1 || n.end_sec <= t0 {
            continue;
        }
        // Keep if onset in window (standard AMT chunking) OR any overlap with onset clip
        if n.start_sec < t0 {
            // started before window — only keep if we want sustain; Phase 3: onset-in-window
            continue;
        }
        if n.start_sec >= t1 {
            continue;
        }
        let start = n.start_sec - t0;
        let end = (n.end_sec - t0).min(t1 - t0).max(start + 1e-3);
        out.push(NoteAnnotation {
            start_sec: start,
            end_sec: end,
            pitch_midi: n.pitch_midi,
            velocity: n.velocity,
            confidence: n.confidence,
            program: n.program,
        });
    }
    out
}

/// Chunk long mono audio + notes into fixed windows.
pub fn chunk_audio_notes(
    piece_id: &str,
    audio: &[f32],
    notes: &NoteList,
    cfg: &ChunkConfig,
) -> Vec<AudioChunk> {
    let sr = cfg.sample_rate as f64;
    let chunk_samples = (cfg.chunk_sec * sr).round() as usize;
    let hop_samples = (cfg.hop_sec * sr).round().max(1.0) as usize;
    if chunk_samples == 0 || audio.is_empty() {
        return vec![];
    }

    let mut base_notes = notes.notes.clone();
    if cfg.remap_gm {
        base_notes.retain_mut(|n| {
            if let Some(c) = gm_to_challenge(n.program) {
                n.program = c;
                true
            } else {
                false
            }
        });
    }

    let mut out = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while start + chunk_samples <= audio.len() || (start < audio.len() && idx == 0 && audio.len() < chunk_samples) {
        let end = (start + chunk_samples).min(audio.len());
        if end <= start {
            break;
        }
        // Pad short final? Only full windows for train volume unless piece shorter than chunk.
        if end - start < chunk_samples && audio.len() >= chunk_samples {
            break;
        }

        let t0 = start as f64 / sr;
        let t1 = t0 + cfg.chunk_sec;
        let mut sliced = slice_notes(&base_notes, t0, t1);

        // Limit to max instruments: keep most frequent programs
        let mut progs = unique_programs(sliced.iter().map(|n| n.program));
        if progs.len() > cfg.max_instruments {
            // Count notes per program
            let mut counts: Vec<(u8, usize)> = progs
                .iter()
                .map(|&p| (p, sliced.iter().filter(|n| n.program == p).count()))
                .collect();
            counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let keep: std::collections::HashSet<u8> = counts
                .into_iter()
                .take(cfg.max_instruments)
                .map(|(p, _)| p)
                .collect();
            sliced.retain(|n| keep.contains(&n.program));
            progs = unique_programs(sliced.iter().map(|n| n.program));
        }

        if sliced.len() < cfg.min_notes {
            start += hop_samples;
            idx += 1;
            continue;
        }

        let mut samples = audio[start..end].to_vec();
        if samples.len() < chunk_samples {
            samples.resize(chunk_samples, 0.0);
        }

        let id = format!("{piece_id}_c{idx:04}");
        out.push(AudioChunk {
            id: id.clone(),
            audio: samples,
            notes: NoteList {
                track_id: id,
                sr: cfg.sample_rate,
                notes: sliced,
            },
            programs: progs.clone(),
            n_instruments: progs.len().max(1),
            start_sec: t0,
            duration_sec: cfg.chunk_sec,
            piece_id: piece_id.to_string(),
        });

        if audio.len() < chunk_samples {
            break; // single short piece
        }
        start += hop_samples;
        idx += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::note::NoteAnnotation;

    #[test]
    fn slice_keeps_onset_in_window() {
        let notes = vec![
            NoteAnnotation {
                start_sec: 0.5,
                end_sec: 1.0,
                pitch_midi: 60,
                velocity: 80,
                confidence: 1.0,
                program: 0,
            },
            NoteAnnotation {
                start_sec: 3.5,
                end_sec: 4.5,
                pitch_midi: 64,
                velocity: 80,
                confidence: 1.0,
                program: 0,
            },
        ];
        let s = slice_notes(&notes, 0.0, 4.0);
        assert_eq!(s.len(), 2);
        assert!((s[0].start_sec - 0.5).abs() < 1e-9);
        assert!((s[1].start_sec - 3.5).abs() < 1e-9);
        assert!(s[1].end_sec <= 4.0 + 1e-9);
    }
}
