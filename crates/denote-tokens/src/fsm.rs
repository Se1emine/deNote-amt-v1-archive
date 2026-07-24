//! Finite-state machine for constrained autoregressive decoding.
//!
//! Blocks illegal MIDI event sequences (e.g. NoteOff without open note,
//! NoteOn without prior Velocity, tokens after EOS).

use crate::vocab::{TokenId, TokenKind, Vocab};

/// Decoder FSM state between tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeState {
    /// Whether BOS has been seen / we are inside a sequence.
    pub started: bool,
    /// Sequence finished (EOS emitted).
    pub finished: bool,
    /// Next token must be NoteOn (after Velocity).
    pub expect_note_on: bool,
    /// Open pitches (bitset for 0..127).
    open: [bool; 128],
    open_count: usize,
    /// Last program id (for bookkeeping).
    pub program: u8,
}

impl Default for DecodeState {
    fn default() -> Self {
        Self {
            started: false,
            finished: false,
            expect_note_on: false,
            open: [false; 128],
            open_count: 0,
            program: 0,
        }
    }
}

impl DecodeState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_open(&self, pitch: u8) -> bool {
        self.open[pitch as usize]
    }

    pub fn open_count(&self) -> usize {
        self.open_count
    }
}

/// Returns true if `token` is legal given `state` and `vocab`.
pub fn valid_next(state: &DecodeState, token: TokenId, vocab: &Vocab) -> bool {
    if state.finished {
        return token == vocab.pad;
    }

    match vocab.kind(token) {
        None => false,
        Some(TokenKind::Pad) => {
            // PAD only allowed before start (batch pad) or we treat as illegal mid-sequence
            !state.started
        }
        Some(TokenKind::Bos) => !state.started && !state.expect_note_on,
        Some(TokenKind::Eos) => state.started && !state.expect_note_on,
        Some(TokenKind::Sep) => state.started && !state.expect_note_on,
        Some(TokenKind::TimeShift) => state.started && !state.expect_note_on,
        Some(TokenKind::Velocity) => {
            state.started
                && !state.expect_note_on
                && state.open_count < vocab.cfg.max_polyphony
        }
        Some(TokenKind::NoteOn) => {
            if !state.expect_note_on {
                return false;
            }
            if let Some(p) = vocab.note_on_pitch(token) {
                // allow re-trigger of same pitch
                let new_open = if state.is_open(p) {
                    state.open_count
                } else {
                    state.open_count + 1
                };
                new_open <= vocab.cfg.max_polyphony
            } else {
                false
            }
        }
        Some(TokenKind::NoteOff) => {
            if !state.started || state.expect_note_on {
                return false;
            }
            vocab
                .note_off_pitch(token)
                .map(|p| state.is_open(p))
                .unwrap_or(false)
        }
        Some(TokenKind::Program) => state.started && !state.expect_note_on,
    }
}

/// Apply `token` to `state` (caller should check [`valid_next`] first).
pub fn step(state: &mut DecodeState, token: TokenId, vocab: &Vocab) {
    match vocab.kind(token) {
        Some(TokenKind::Bos) => {
            state.started = true;
        }
        Some(TokenKind::Eos) => {
            state.finished = true;
            state.expect_note_on = false;
        }
        Some(TokenKind::Velocity) => {
            state.expect_note_on = true;
        }
        Some(TokenKind::NoteOn) => {
            if let Some(p) = vocab.note_on_pitch(token) {
                if !state.open[p as usize] {
                    state.open[p as usize] = true;
                    state.open_count += 1;
                }
            }
            state.expect_note_on = false;
        }
        Some(TokenKind::NoteOff) => {
            if let Some(p) = vocab.note_off_pitch(token) {
                if state.open[p as usize] {
                    state.open[p as usize] = false;
                    state.open_count = state.open_count.saturating_sub(1);
                }
            }
        }
        Some(TokenKind::Program) => {
            if let Some(p) = vocab.program_id(token) {
                state.program = p;
            }
        }
        Some(TokenKind::TimeShift) | Some(TokenKind::Sep) | Some(TokenKind::Pad) | None => {}
    }
}

/// Build a boolean mask over the full vocab: `true` = allowed next token.
pub fn allowed_mask(state: &DecodeState, vocab: &Vocab) -> Vec<bool> {
    let mut mask = vec![false; vocab.size as usize];
    for id in 0..vocab.size {
        if valid_next(state, id, vocab) {
            mask[id as usize] = true;
        }
    }
    mask
}

/// In-place: set logits of forbidden tokens to `neg_inf` (default -1e9).
pub fn apply_constraint_mask(logits: &mut [f32], state: &DecodeState, vocab: &Vocab, neg_inf: f32) {
    let n = logits.len().min(vocab.size as usize);
    for id in 0..n as u32 {
        if !valid_next(state, id, vocab) {
            logits[id as usize] = neg_inf;
        }
    }
}

/// Simulate a full sequence; returns false if any token is illegal.
pub fn sequence_is_valid(tokens: &[TokenId], vocab: &Vocab) -> bool {
    let mut state = DecodeState::new();
    for &t in tokens {
        if t == vocab.pad {
            continue;
        }
        if !valid_next(&state, t, vocab) {
            return false;
        }
        step(&mut state, t, vocab);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode_notes, TokenNote};
    use crate::vocab::Vocab;

    #[test]
    fn encoded_sequences_are_valid() {
        let v = Vocab::default_mt3_like();
        let notes = vec![
            TokenNote {
                start_sec: 0.0,
                end_sec: 0.3,
                pitch_midi: 60,
                velocity: 100,
                program: 0,
            },
            TokenNote {
                start_sec: 0.1,
                end_sec: 0.5,
                pitch_midi: 67,
                velocity: 90,
                program: 0,
            },
        ];
        let ids = encode_notes(&notes, &v);
        assert!(sequence_is_valid(&ids, &v), "ids={ids:?}");
    }

    #[test]
    fn rejects_note_off_without_on() {
        let v = Vocab::default_mt3_like();
        let mut state = DecodeState::new();
        step(&mut state, v.bos, &v);
        let off = v.note_off(60).unwrap();
        assert!(!valid_next(&state, off, &v));
    }

    #[test]
    fn rejects_note_on_without_velocity() {
        let v = Vocab::default_mt3_like();
        let mut state = DecodeState::new();
        step(&mut state, v.bos, &v);
        let on = v.note_on(60).unwrap();
        assert!(!valid_next(&state, on, &v));
        step(&mut state, v.velocity(80), &v);
        assert!(valid_next(&state, on, &v));
    }

    #[test]
    fn after_eos_only_pad() {
        let v = Vocab::default_mt3_like();
        let mut state = DecodeState::new();
        step(&mut state, v.bos, &v);
        step(&mut state, v.eos, &v);
        assert!(valid_next(&state, v.pad, &v));
        assert!(!valid_next(&state, v.time_shift(1).unwrap(), &v));
    }
}
