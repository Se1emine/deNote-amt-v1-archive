//! Multi-stream note merge + global FSM rules (Phase 2).

use serde::{Deserialize, Serialize};

use denote_tokens::TokenNote;

use crate::NoteEvent;

/// Note with instrument program (group-stream decode output).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProgramNoteEvent {
    pub start_sec: f64,
    pub end_sec: f64,
    pub duration_sec: f64,
    pub pitch_midi: u8,
    pub confidence: f32,
    pub velocity: u8,
    pub program: u8,
    /// Source group id (if known).
    #[serde(default)]
    pub group_id: Option<u32>,
}

impl ProgramNoteEvent {
    pub fn from_token_note(n: &TokenNote, group_id: Option<u32>) -> Self {
        Self {
            start_sec: n.start_sec,
            end_sec: n.end_sec,
            duration_sec: (n.end_sec - n.start_sec).max(0.0),
            pitch_midi: n.pitch_midi,
            confidence: 1.0,
            velocity: n.velocity,
            program: n.program,
            group_id,
        }
    }

    pub fn to_note_event(&self) -> NoteEvent {
        NoteEvent {
            start_sec: self.start_sec,
            end_sec: self.end_sec,
            duration_sec: self.duration_sec,
            pitch_midi: self.pitch_midi,
            confidence: self.confidence,
            velocity: self.velocity,
            program: self.program,
        }
    }
}

/// Merge options for multi-group detokenized streams.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Drop notes that share program+pitch with onset within ±ε seconds.
    pub onset_eps_sec: f64,
    /// Optional max distinct programs kept (by earliest onset first). None = unlimited.
    pub max_programs: Option<usize>,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            onset_eps_sec: 0.03,
            max_programs: None,
        }
    }
}

/// Concatenate group streams, sort by onset, drop near-duplicates, optional program budget.
pub fn merge_note_events(
    streams: impl IntoIterator<Item = Vec<ProgramNoteEvent>>,
    cfg: &MergeConfig,
) -> Vec<ProgramNoteEvent> {
    let mut all: Vec<ProgramNoteEvent> = streams.into_iter().flatten().collect();
    all.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.program.cmp(&b.program))
            .then_with(|| a.pitch_midi.cmp(&b.pitch_midi))
    });

    // Drop duplicates: same program + pitch + onset±ε
    let mut deduped: Vec<ProgramNoteEvent> = Vec::with_capacity(all.len());
    for n in all {
        let is_dup = deduped.iter().any(|p| {
            p.program == n.program
                && p.pitch_midi == n.pitch_midi
                && (p.start_sec - n.start_sec).abs() <= cfg.onset_eps_sec
        });
        if !is_dup {
            deduped.push(n);
        }
    }

    if let Some(max_p) = cfg.max_programs {
        let mut kept_programs = Vec::new();
        let mut out = Vec::new();
        for n in deduped {
            if !kept_programs.contains(&n.program) {
                if kept_programs.len() >= max_p {
                    continue;
                }
                kept_programs.push(n.program);
            }
            out.push(n);
        }
        return out;
    }
    deduped
}

/// Convert TokenNote lists from each group into merged ProgramNoteEvents.
pub fn detokenize_merge(
    group_notes: &[(u32, Vec<TokenNote>)],
    cfg: &MergeConfig,
) -> Vec<ProgramNoteEvent> {
    let streams = group_notes.iter().map(|(g, notes)| {
        notes
            .iter()
            .map(|n| ProgramNoteEvent::from_token_note(n, Some(*g)))
            .collect::<Vec<_>>()
    });
    merge_note_events(streams, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pn(start: f64, pitch: u8, program: u8, group: u32) -> ProgramNoteEvent {
        ProgramNoteEvent {
            start_sec: start,
            end_sec: start + 0.2,
            duration_sec: 0.2,
            pitch_midi: pitch,
            confidence: 1.0,
            velocity: 80,
            program,
            group_id: Some(group),
        }
    }

    #[test]
    fn merge_two_groups_sorted() {
        let g0 = vec![pn(0.5, 60, 0, 0)];
        let g1 = vec![pn(0.1, 64, 1, 1), pn(0.3, 67, 1, 1)];
        let merged = merge_note_events([g0, g1], &MergeConfig::default());
        assert_eq!(merged.len(), 3);
        assert!((merged[0].start_sec - 0.1).abs() < 1e-9);
        assert_eq!(merged[0].program, 1);
        assert!((merged[2].start_sec - 0.5).abs() < 1e-9);
        assert_eq!(merged[2].program, 0);
    }

    #[test]
    fn drops_near_duplicate() {
        let a = vec![pn(0.10, 60, 0, 0)];
        let b = vec![pn(0.12, 60, 0, 0)]; // same program+pitch, onset within ε
        let merged = merge_note_events([a, b], &MergeConfig::default());
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn program_budget() {
        let notes = vec![
            pn(0.0, 60, 0, 0),
            pn(0.1, 64, 1, 1),
            pn(0.2, 67, 6, 3),
        ];
        let cfg = MergeConfig {
            onset_eps_sec: 0.03,
            max_programs: Some(2),
        };
        let merged = merge_note_events([notes], &cfg);
        assert_eq!(merged.len(), 2);
        let progs: Vec<u8> = merged.iter().map(|n| n.program).collect();
        assert!(progs.contains(&0));
        assert!(progs.contains(&1));
        assert!(!progs.contains(&6));
    }
}
