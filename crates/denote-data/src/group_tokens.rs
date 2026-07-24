//! Per-group token sequences for Job B_g (Phase 2).
//!
//! Full multi-program streams are filtered to programs in group `g`, then
//! re-encoded with the shared vocab (`encode_notes`).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::note::{NoteAnnotation, NoteList};
use crate::piano_roll::notes_from_json;
use crate::token_cache::{
    note_annotation_to_token_note, TokenCachedSample, TokenFeatureCache, TokenSampleMeta,
};
use denote_tokens::{encode_notes, TokenId, Vocab};

/// Filter notes whose `program` is in `programs`.
pub fn filter_notes_by_programs(
    notes: &[NoteAnnotation],
    programs: &[u8],
) -> Vec<NoteAnnotation> {
    let set: HashSet<u8> = programs.iter().copied().collect();
    notes
        .iter()
        .filter(|n| set.contains(&n.program))
        .cloned()
        .collect()
}

/// Encode filtered notes for group `g`. Empty notes → `[BOS, EOS]` (caller may skip).
pub fn encode_group_tokens(
    notes: &[NoteAnnotation],
    programs: &[u8],
    vocab: &Vocab,
) -> Vec<TokenId> {
    let filtered = filter_notes_by_programs(notes, programs);
    let token_notes: Vec<_> = filtered
        .iter()
        .map(note_annotation_to_token_note)
        .collect();
    encode_notes(&token_notes, vocab)
}

/// True if the clip has at least one note in the group programs.
pub fn group_has_notes(notes: &[NoteAnnotation], programs: &[u8]) -> bool {
    let set: HashSet<u8> = programs.iter().copied().collect();
    notes.iter().any(|n| set.contains(&n.program))
}

/// Build a token cache sample for one group (for optional disk materialization).
pub fn build_group_token_sample(
    track_id: &str,
    notes: &NoteList,
    programs: &[u8],
    vocab: &Vocab,
    sample_rate: u32,
    duration_sec: f64,
) -> TokenCachedSample {
    let token_ids = encode_group_tokens(&notes.notes, programs, vocab);
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
        audio_path: None,
        audio_pcm: None,
    }
}

/// Relative path convention: `tokens_g/{id}.g{g}.tok.bin`.
pub fn group_token_rel_path(id: &str, group_id: u32) -> String {
    format!("tokens_g/{id}.g{group_id}.tok.bin")
}

/// Disk helper: `tokens_g/` under cache root.
pub struct GroupTokenCache {
    pub root: PathBuf,
}

impl GroupTokenCache {
    pub fn new(cache_root: impl Into<PathBuf>) -> Self {
        Self {
            root: cache_root.into().join("tokens_g"),
        }
    }

    pub fn sample_path(&self, track_id: &str, group_id: u32) -> PathBuf {
        self.root
            .join(format!("{track_id}.g{group_id}.tok.bin"))
    }

    pub fn save(
        &self,
        track_id: &str,
        group_id: u32,
        sample: &TokenCachedSample,
    ) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        // Reuse TokenFeatureCache binary format with a temp id layout.
        let cache = TokenFeatureCache::new(&self.root);
        let mut s = sample.clone();
        s.meta.track_id = format!("{track_id}.g{group_id}");
        cache.save(&s)?;
        // TokenFeatureCache writes `{track_id}.tok.bin` — rename to `.g{g}.tok.bin` if needed.
        let written = self.root.join(format!("{}.tok.bin", s.meta.track_id));
        let target = self.sample_path(track_id, group_id);
        if written != target && written.exists() {
            if target.exists() {
                fs::remove_file(&target)?;
            }
            fs::rename(&written, &target)?;
        }
        Ok(())
    }

    pub fn load(&self, track_id: &str, group_id: u32) -> Result<TokenCachedSample> {
        let path = self.sample_path(track_id, group_id);
        let f = fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        // Same bincode layout as TokenFeatureCache.
        #[derive(Deserialize)]
        struct TokenSampleDisk {
            meta: TokenSampleMeta,
            token_ids: Vec<TokenId>,
            audio_path: Option<String>,
            audio_pcm: Option<Vec<f32>>,
        }
        let disk: TokenSampleDisk = bincode::deserialize_from(std::io::BufReader::new(f))
            .with_context(|| format!("read {}", path.display()))?;
        Ok(TokenCachedSample {
            meta: disk.meta,
            token_ids: disk.token_ids,
            audio_path: disk.audio_path,
            audio_pcm: disk.audio_pcm,
        })
    }
}

/// Load notes for a manifest entry (from `notes_path` or sibling of audio).
pub fn load_notes_for_entry(
    notes_path: Option<&str>,
    audio_path: Option<&str>,
    id: &str,
) -> Result<NoteList> {
    if let Some(p) = notes_path {
        let path = Path::new(p);
        if path.exists() {
            return notes_from_json(path);
        }
    }
    if let Some(audio) = audio_path {
        let p = Path::new(audio).with_extension("notes.json");
        if p.exists() {
            return notes_from_json(&p);
        }
    }
    // Fallback empty
    Ok(NoteList {
        track_id: id.to_string(),
        sr: 24_000,
        notes: vec![],
    })
}

/// Materialize per-group token files for all entries that have group notes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupTokensBuildReport {
    pub n_entries: usize,
    pub n_written: usize,
    pub n_skipped_empty: usize,
    pub group_id: u32,
}

pub fn build_group_tokens_for_manifest(
    entries: &[(String, NoteList)],
    programs: &[u8],
    group_id: u32,
    vocab: &Vocab,
    cache_root: impl AsRef<Path>,
) -> Result<GroupTokensBuildReport> {
    let cache = GroupTokenCache::new(cache_root.as_ref());
    let mut n_written = 0usize;
    let mut n_skipped = 0usize;
    for (id, notes) in entries {
        if !group_has_notes(&notes.notes, programs) {
            n_skipped += 1;
            continue;
        }
        let sample = build_group_token_sample(
            id,
            notes,
            programs,
            vocab,
            notes.sr,
            0.0,
        );
        cache.save(id, group_id, &sample)?;
        n_written += 1;
    }
    Ok(GroupTokensBuildReport {
        n_entries: entries.len(),
        n_written,
        n_skipped_empty: n_skipped,
        group_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use denote_tokens::Vocab;

    fn note(program: u8, pitch: u8) -> NoteAnnotation {
        NoteAnnotation {
            start_sec: 0.1,
            end_sec: 0.4,
            pitch_midi: pitch,
            velocity: 80,
            confidence: 1.0,
            program,
        }
    }

    #[test]
    fn piano_only_filters_to_g0() {
        let notes = vec![note(0, 60), note(1, 64), note(0, 67)];
        let g0 = filter_notes_by_programs(&notes, &[0]);
        assert_eq!(g0.len(), 2);
        assert!(g0.iter().all(|n| n.program == 0));
        let g1 = filter_notes_by_programs(&notes, &[1, 2, 3]);
        assert_eq!(g1.len(), 1);
        assert_eq!(g1[0].program, 1);
    }

    #[test]
    fn encode_group_tokens_non_empty() {
        let v = Vocab::default_mt3_like();
        let notes = vec![note(0, 60)];
        let ids = encode_group_tokens(&notes, &[0], &v);
        assert_eq!(ids.first().copied(), Some(v.bos));
        assert_eq!(ids.last().copied(), Some(v.eos));
        assert!(ids.len() > 2);
    }

    #[test]
    fn empty_group_bos_eos() {
        let v = Vocab::default_mt3_like();
        let notes = vec![note(1, 60)];
        let ids = encode_group_tokens(&notes, &[0], &v);
        assert_eq!(ids, vec![v.bos, v.eos]);
        assert!(!group_has_notes(&notes, &[0]));
    }
}
