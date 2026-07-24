//! Cross-stem mixing (BabySlakh / Slakh-style) — Phase 3 Stem-A.
//!
//! Sample 1..=max stems from a track (non-drum), sum audio, union notes with programs.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use log::info;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::chunk::{chunk_audio_notes, AudioChunk, ChunkConfig};
use crate::note::NoteList;
use crate::piano_roll::load_midi_notes;
use crate::programs::{gm_to_challenge, is_drum_program};
use denote_audio::{load_mono_resampled, write_wav_mono_normalized};

#[derive(Debug, Clone)]
pub struct StemInfo {
    pub stem_id: String,
    pub wav: PathBuf,
    pub midi: PathBuf,
    pub gm_program: u8,
}

#[derive(Debug, Clone)]
pub struct BabySlakhTrack {
    pub track_id: String,
    pub root: PathBuf,
    pub stems: Vec<StemInfo>,
    pub mix_wav: Option<PathBuf>,
}

/// Discover BabySlakh-style track dirs under `babyslakh_16k/TrackXXXXX/`.
pub fn discover_babyslakh(root: impl AsRef<Path>) -> Result<Vec<BabySlakhTrack>> {
    let root = root.as_ref();
    if !root.is_dir() {
        bail!("babyslakh root not a dir: {}", root.display());
    }
    let mut tracks = Vec::new();
    let mut dirs: Vec<PathBuf> = fs::read_dir(root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();

    for dir in dirs {
        let track_id = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("track")
            .to_string();
        let stems_dir = dir.join("stems");
        let midi_dir = dir.join("MIDI");
        if !stems_dir.is_dir() {
            continue;
        }
        let mut stems = Vec::new();
        let mut wavs: Vec<PathBuf> = fs::read_dir(&stems_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("wav"))
                    .unwrap_or(false)
            })
            .collect();
        wavs.sort();

        for wav in wavs {
            let stem_id = wav
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("S")
                .to_string();
            let midi = midi_dir.join(format!("{stem_id}.mid"));
            if !midi.exists() {
                continue;
            }
            // Infer GM program from MIDI ProgramChange (first note's program)
            let gm_program = load_midi_notes(&midi)
                .ok()
                .and_then(|n| n.notes.first().map(|x| x.program))
                .unwrap_or(0);
            if is_drum_program(gm_program) {
                continue;
            }
            if gm_to_challenge(gm_program).is_none() {
                continue;
            }
            stems.push(StemInfo {
                stem_id,
                wav,
                midi,
                gm_program,
            });
        }
        if stems.is_empty() {
            continue;
        }
        let mix = dir.join("mix.wav");
        tracks.push(BabySlakhTrack {
            track_id,
            root: dir,
            stems,
            mix_wav: if mix.exists() { Some(mix) } else { None },
        });
    }
    info!("babyslakh tracks with usable stems: {}", tracks.len());
    Ok(tracks)
}

#[derive(Debug, Clone)]
pub struct StemMixConfig {
    pub sample_rate: u32,
    pub max_instruments: usize,
    pub min_instruments: usize,
    pub mixes_per_track: usize,
    pub chunk: ChunkConfig,
}

impl Default for StemMixConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            max_instruments: 3,
            min_instruments: 1,
            mixes_per_track: 4,
            chunk: ChunkConfig::default(),
        }
    }
}

/// Mix selected stems → mono audio + union notes (challenge programs).
pub fn mix_stems(
    track: &BabySlakhTrack,
    stem_indices: &[usize],
    sample_rate: u32,
) -> Result<(Vec<f32>, NoteList)> {
    if stem_indices.is_empty() {
        bail!("no stems to mix");
    }
    let mut audio: Vec<f32> = Vec::new();
    let mut all_notes = Vec::new();

    for &si in stem_indices {
        let stem = &track.stems[si];
        let samples = load_mono_resampled(&stem.wav, sample_rate)
            .with_context(|| format!("load stem {}", stem.wav.display()))?;
        if audio.len() < samples.len() {
            audio.resize(samples.len(), 0.0);
        }
        for (i, &s) in samples.iter().enumerate() {
            audio[i] += s;
        }
        let mut nl = load_midi_notes(&stem.midi)?;
        let ch = gm_to_challenge(stem.gm_program).unwrap_or(0);
        for n in &mut nl.notes {
            // Prefer stem-level program if note had drum/0
            if let Some(c) = gm_to_challenge(n.program) {
                n.program = c;
            } else {
                n.program = ch;
            }
        }
        // Drop still-unmapped (shouldn't happen)
        nl.notes.retain(|n| n.program < 8);
        all_notes.extend(nl.notes);
    }

    all_notes.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap()
            .then_with(|| a.pitch_midi.cmp(&b.pitch_midi))
    });

    let notes = NoteList {
        track_id: track.track_id.clone(),
        sr: sample_rate,
        notes: all_notes,
    };
    Ok((audio, notes))
}

/// Generate chunked mixed clips from BabySlakh tracks.
///
/// Prefers stems with **distinct** challenge programs so curriculum stages
/// 1→2→3 actually see multi-instrument clips (guitar+piano both map to 0 otherwise).
pub fn generate_stem_mix_chunks(
    tracks: &[BabySlakhTrack],
    cfg: &StemMixConfig,
    rng: &mut impl Rng,
) -> Result<Vec<AudioChunk>> {
    let mut out = Vec::new();
    for track in tracks {
        let n_stems = track.stems.len();
        if n_stems == 0 {
            continue;
        }
        // Group stem indices by challenge program for diversity.
        let mut by_prog: std::collections::BTreeMap<u8, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, s) in track.stems.iter().enumerate() {
            if let Some(c) = gm_to_challenge(s.gm_program) {
                by_prog.entry(c).or_default().push(i);
            }
        }
        let distinct_progs: Vec<u8> = by_prog.keys().copied().collect();
        if distinct_progs.is_empty() {
            continue;
        }

        for m in 0..cfg.mixes_per_track {
            let max_pick = cfg.max_instruments.min(distinct_progs.len()).max(1);
            let min_pick = cfg.min_instruments.min(max_pick).max(1);
            let n_pick = if min_pick == max_pick {
                min_pick
            } else {
                rng.gen_range(min_pick..=max_pick)
            };

            let mut prog_pool = distinct_progs.clone();
            prog_pool.shuffle(rng);
            prog_pool.truncate(n_pick);

            let mut idxs = Vec::with_capacity(n_pick);
            for p in prog_pool {
                if let Some(list) = by_prog.get_mut(&p) {
                    list.shuffle(rng);
                    if let Some(si) = list.first().copied() {
                        idxs.push(si);
                    }
                }
            }
            idxs.sort_unstable();
            if idxs.is_empty() {
                continue;
            }

            let (audio, notes) = match mix_stems(track, &idxs, cfg.sample_rate) {
                Ok(x) => x,
                Err(e) => {
                    log::warn!("mix {} failed: {e:#}", track.track_id);
                    continue;
                }
            };
            let piece_id = format!("{}_mix{m}", track.track_id);
            let mut chunk_cfg = cfg.chunk.clone();
            chunk_cfg.sample_rate = cfg.sample_rate;
            chunk_cfg.remap_gm = false; // already challenge
            let chunks = chunk_audio_notes(&piece_id, &audio, &notes, &chunk_cfg);
            out.extend(chunks);
        }
    }
    info!("stem-mix chunks: {}", out.len());
    Ok(out)
}

/// Write chunk list as flat `{id}.wav` + `{id}.notes.json` for Job L0.
pub fn write_chunks_flat(chunks: &[AudioChunk], out_dir: &Path) -> Result<usize> {
    fs::create_dir_all(out_dir)?;
    let mut n = 0usize;
    for c in chunks {
        let wav = out_dir.join(format!("{}.wav", c.id));
        let notes = out_dir.join(format!("{}.notes.json", c.id));
        if wav.exists() && notes.exists() {
            n += 1;
            continue;
        }
        write_wav_mono_normalized(&wav, &c.audio, c.notes.sr, 0.9)?;
        fs::write(&notes, serde_json::to_string_pretty(&c.notes)?)?;
        n += 1;
    }
    Ok(n)
}
