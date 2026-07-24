//! Decoding activations / tokens into note events.
//!
//! - **Primary (MIROS):** [`tokens_to_note_events`] — MIDI-event tokens → notes
//! - **Phase 2 modular:** [`ar`], [`merge`] — constrained AR helpers + multi-group merge
//! - **Legacy (CRNN):** [`decode_notes`] — onset/frame/offset maps → notes

mod ar;
mod merge;

pub use ar::{ar_decode_loop, greedy_next_token, group_active_from_presence};
pub use merge::{detokenize_merge, merge_note_events, MergeConfig, ProgramNoteEvent};

use serde::{Deserialize, Serialize};

use denote_tokens::{decode_tokens, TokenId, TokenNote, Vocab};

/// One decoded note (compatible with deNote `NoteEvent` fields).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NoteEvent {
    pub start_sec: f64,
    pub end_sec: f64,
    pub duration_sec: f64,
    pub pitch_midi: u8,
    pub confidence: f32,
    /// Mean frame activation * 127 (product velocity stand-in).
    pub velocity: u8,
    /// Challenge program index (0..K-1). Default 0 for legacy CRNN path.
    #[serde(default)]
    pub program: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeConfig {
    pub sample_rate: u32,
    pub hop_length: usize,
    pub midi_offset: u8,
    pub onset_threshold: f32,
    pub frame_threshold: f32,
    pub offset_threshold: f32,
    pub min_note_ms: f64,
    pub merge_gap_ms: f64,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            hop_length: 256,
            midi_offset: 21,
            onset_threshold: 0.5,
            frame_threshold: 0.3,
            offset_threshold: 0.5,
            min_note_ms: 58.0,
            merge_gap_ms: 40.0,
        }
    }
}

impl DecodeConfig {
    pub fn seconds_per_frame(&self) -> f64 {
        self.hop_length as f64 / self.sample_rate as f64
    }

    pub fn min_note_frames(&self) -> usize {
        let frames = (self.min_note_ms / 1000.0) / self.seconds_per_frame();
        frames.ceil().max(1.0) as usize
    }
}

/// Decode onset / frame / offset maps `[T][P]` (row-major time × pitch) to notes.
///
/// Values should be probabilities in `[0, 1]` (apply sigmoid to logits first).
pub fn decode_notes(
    onset: &[f32],
    frame: &[f32],
    offset: Option<&[f32]>,
    n_frames: usize,
    n_pitches: usize,
    cfg: &DecodeConfig,
) -> Vec<NoteEvent> {
    assert_eq!(onset.len(), n_frames * n_pitches);
    assert_eq!(frame.len(), n_frames * n_pitches);

    let spf = cfg.seconds_per_frame();
    let min_len = cfg.min_note_frames();
    let mut notes = Vec::new();

    for p in 0..n_pitches {
        let mut t = 0usize;
        while t < n_frames {
            let o = onset[t * n_pitches + p];
            let is_peak = o >= cfg.onset_threshold
                && (t == 0 || o >= onset[(t - 1) * n_pitches + p])
                && (t + 1 >= n_frames || o >= onset[(t + 1) * n_pitches + p]);

            if !is_peak {
                t += 1;
                continue;
            }

            let start = t;
            let mut end = t + 1;
            let mut energy_sum = frame[t * n_pitches + p];
            let mut energy_n = 1usize;

            while end < n_frames {
                let fr = frame[end * n_pitches + p];
                let off = offset
                    .map(|off| off[end * n_pitches + p] >= cfg.offset_threshold)
                    .unwrap_or(false);
                if fr < cfg.frame_threshold || off {
                    if off {
                        end += 1;
                    }
                    break;
                }
                energy_sum += fr;
                energy_n += 1;
                end += 1;
            }

            if end - start >= min_len {
                let conf = o.max(energy_sum / energy_n as f32);
                let start_sec = start as f64 * spf;
                let end_sec = end as f64 * spf;
                notes.push(NoteEvent {
                    start_sec,
                    end_sec,
                    duration_sec: end_sec - start_sec,
                    pitch_midi: cfg.midi_offset + p as u8,
                    confidence: conf.clamp(0.0, 1.0),
                    velocity: (conf.clamp(0.0, 1.0) * 127.0) as u8,
                    program: 0,
                });
            }
            // jump past this note to avoid re-trigger spam
            t = end.max(t + 1);
        }
    }

    notes.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap()
            .then_with(|| a.pitch_midi.cmp(&b.pitch_midi))
    });

    merge_close_same_pitch(notes, cfg)
}

fn merge_close_same_pitch(notes: Vec<NoteEvent>, cfg: &DecodeConfig) -> Vec<NoteEvent> {
    if notes.is_empty() {
        return notes;
    }
    let gap = cfg.merge_gap_ms / 1000.0;
    let mut out: Vec<NoteEvent> = Vec::with_capacity(notes.len());
    for n in notes {
        if let Some(last) = out.last_mut() {
            if last.pitch_midi == n.pitch_midi && n.start_sec - last.end_sec <= gap {
                last.end_sec = last.end_sec.max(n.end_sec);
                last.duration_sec = last.end_sec - last.start_sec;
                last.confidence = last.confidence.max(n.confidence);
                last.velocity = last.velocity.max(n.velocity);
                continue;
            }
        }
        out.push(n);
    }
    out
}

/// Apply sigmoid to a flat logit buffer in-place.
pub fn sigmoid_inplace(xs: &mut [f32]) {
    for x in xs {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

/// Convert seq2seq MIDI tokens into deNote-compatible [`NoteEvent`]s.
pub fn tokens_to_note_events(tokens: &[TokenId], vocab: &Vocab) -> Vec<NoteEvent> {
    decode_tokens(tokens, vocab)
        .into_iter()
        .map(token_note_to_event)
        .collect()
}

fn token_note_to_event(n: TokenNote) -> NoteEvent {
    NoteEvent {
        start_sec: n.start_sec,
        end_sec: n.end_sec,
        duration_sec: (n.end_sec - n.start_sec).max(0.0),
        pitch_midi: n.pitch_midi,
        confidence: 1.0,
        velocity: n.velocity,
        program: n.program,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_simple_note() {
        let t = 50;
        let p = 88;
        let mut onset = vec![0.0f32; t * p];
        let mut frame = vec![0.0f32; t * p];
        let pitch = 39; // MIDI 60
        onset[10 * p + pitch] = 0.9;
        for i in 10..30 {
            frame[i * p + pitch] = 0.8;
        }
        let notes = decode_notes(&onset, &frame, None, t, p, &DecodeConfig::default());
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch_midi, 60);
        assert!(notes[0].duration_sec > 0.1);
    }
}
