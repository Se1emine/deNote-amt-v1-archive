//! Note annotations → MIDI-like token sequences.

use crate::vocab::{TokenId, Vocab};

/// Minimal note input for encoding (no dependency on denote-data).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenNote {
    pub start_sec: f64,
    pub end_sec: f64,
    pub pitch_midi: u8,
    pub velocity: u8,
    /// Optional program / instrument group (default 0).
    pub program: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvKind {
    Off,
    On,
}

#[derive(Debug, Clone)]
struct TimedEvent {
    time_sec: f64,
    kind: EvKind,
    pitch: u8,
    velocity: u8,
}

/// Encode notes into a token sequence: `BOS [Program?] …events… EOS`.
///
/// Ordering at equal timestamps: NoteOff before NoteOn (MIDI-safe).
/// Relative `TimeShift` tokens advance the clock; long gaps are split into
/// multiple shifts of at most `n_time_bins - 1` bins.
pub fn encode_notes(notes: &[TokenNote], vocab: &Vocab) -> Vec<TokenId> {
    let mut events: Vec<TimedEvent> = Vec::with_capacity(notes.len() * 2);
    for n in notes {
        if n.end_sec <= n.start_sec {
            continue;
        }
        if (n.pitch_midi as u32) >= vocab.cfg.n_pitches {
            continue;
        }
        events.push(TimedEvent {
            time_sec: n.start_sec.max(0.0),
            kind: EvKind::On,
            pitch: n.pitch_midi,
            velocity: n.velocity.max(1), // never silent on
        });
        events.push(TimedEvent {
            time_sec: n.end_sec.max(0.0),
            kind: EvKind::Off,
            pitch: n.pitch_midi,
            velocity: 0,
        });
    }

    events.sort_by(|a, b| {
        a.time_sec
            .partial_cmp(&b.time_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                // Off before On at same time
                match (a.kind, b.kind) {
                    (EvKind::Off, EvKind::On) => std::cmp::Ordering::Less,
                    (EvKind::On, EvKind::Off) => std::cmp::Ordering::Greater,
                    _ => a.pitch.cmp(&b.pitch),
                }
            })
    });

    let mut out = Vec::with_capacity(events.len() * 3 + 4);
    out.push(vocab.bos);

    // Emit program once from first note if multi-program vocab is used.
    if let Some(first) = notes.first() {
        if let Some(pid) = vocab.program(first.program.min((vocab.cfg.n_programs - 1) as u8)) {
            out.push(pid);
        }
    }

    let mut t_sec = 0.0_f64;
    let bin = vocab.cfg.time_bin_sec.max(1e-9);
    let max_bins = vocab.cfg.n_time_bins.saturating_sub(1).max(1);

    for ev in events {
        let delta = (ev.time_sec - t_sec).max(0.0);
        // quantize to bins (round)
        let mut remaining_bins = (delta / bin).round() as u32;
        while remaining_bins > 0 {
            let step = remaining_bins.min(max_bins);
            if let Some(ts) = vocab.time_shift(step) {
                out.push(ts);
            }
            remaining_bins -= step;
            t_sec += step as f64 * bin;
        }
        // snap clock to event time after shifts (avoid drift accumulation for exact ends)
        t_sec = ev.time_sec.max(t_sec);

        match ev.kind {
            EvKind::On => {
                out.push(vocab.velocity(ev.velocity));
                if let Some(on) = vocab.note_on(ev.pitch) {
                    out.push(on);
                }
            }
            EvKind::Off => {
                if let Some(off) = vocab.note_off(ev.pitch) {
                    out.push(off);
                }
            }
        }
    }

    out.push(vocab.eos);
    out
}

/// Pad / truncate sequence to fixed length for batching (`PAD` fill).
pub fn pad_tokens(tokens: &[TokenId], max_len: usize, pad: TokenId) -> Vec<TokenId> {
    if tokens.len() >= max_len {
        tokens[..max_len].to_vec()
    } else {
        let mut v = tokens.to_vec();
        v.resize(max_len, pad);
        v
    }
}

/// Attention / loss mask: 1 for real tokens, 0 for pad.
pub fn token_mask(tokens: &[TokenId], pad: TokenId) -> Vec<f32> {
    tokens
        .iter()
        .map(|&t| if t == pad { 0.0 } else { 1.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;

    #[test]
    fn encodes_single_note_with_bos_eos() {
        let v = Vocab::default_mt3_like();
        let notes = [TokenNote {
            start_sec: 0.5,
            end_sec: 1.0,
            pitch_midi: 60,
            velocity: 80,
            program: 0,
        }];
        let ids = encode_notes(&notes, &v);
        assert_eq!(ids.first().copied(), Some(v.bos));
        assert_eq!(ids.last().copied(), Some(v.eos));
        assert!(ids.iter().any(|&t| v.note_on_pitch(t) == Some(60)));
        assert!(ids.iter().any(|&t| v.note_off_pitch(t) == Some(60)));
    }
}
