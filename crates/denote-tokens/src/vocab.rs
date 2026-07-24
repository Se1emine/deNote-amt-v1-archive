//! MT3 / MIDI-like vocabulary for seq2seq transcription.

use serde::{Deserialize, Serialize};

/// Integer token id used by the model.
pub type TokenId = u32;

/// Configuration for the event vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VocabConfig {
    /// Relative time-shift bin size in seconds (default 10 ms).
    pub time_bin_sec: f64,
    /// Number of relative time-shift bins (covers `0 .. n_time_bins * time_bin_sec`).
    pub n_time_bins: u32,
    /// MIDI pitch range size (0..127 → 128).
    pub n_pitches: u32,
    /// Velocity bins (128 = full MIDI; use fewer to shrink vocab).
    pub n_velocity: u32,
    /// Instrument / program groups (MIROS-style multi-group).
    pub n_programs: u32,
    /// Max simultaneous open notes (FSM constraint).
    pub max_polyphony: usize,
}

impl Default for VocabConfig {
    fn default() -> Self {
        Self {
            time_bin_sec: 0.01,
            n_time_bins: 205, // 0..2.04 s per TimeShift token
            n_pitches: 128,
            n_velocity: 128,
            n_programs: 8,
            max_polyphony: 16,
        }
    }
}

/// Token class for FSM / debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Pad,
    Bos,
    Eos,
    Sep,
    TimeShift,
    NoteOn,
    NoteOff,
    Velocity,
    Program,
}

/// Vocabulary layout (contiguous id ranges).
///
/// ```text
/// [PAD BOS EOS SEP] [TimeShift × N] [NoteOn × P] [NoteOff × P] [Velocity × V] [Program × G]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Vocab {
    pub cfg: VocabConfig,
    pub pad: TokenId,
    pub bos: TokenId,
    pub eos: TokenId,
    pub sep: TokenId,
    pub time_shift_base: TokenId,
    pub note_on_base: TokenId,
    pub note_off_base: TokenId,
    pub velocity_base: TokenId,
    pub program_base: TokenId,
    pub size: TokenId,
}

impl Vocab {
    pub fn new(cfg: VocabConfig) -> Self {
        let mut next: TokenId = 0;
        let pad = next;
        next += 1;
        let bos = next;
        next += 1;
        let eos = next;
        next += 1;
        let sep = next;
        next += 1;

        let time_shift_base = next;
        next += cfg.n_time_bins;
        let note_on_base = next;
        next += cfg.n_pitches;
        let note_off_base = next;
        next += cfg.n_pitches;
        let velocity_base = next;
        next += cfg.n_velocity;
        let program_base = next;
        next += cfg.n_programs;

        Self {
            cfg,
            pad,
            bos,
            eos,
            sep,
            time_shift_base,
            note_on_base,
            note_off_base,
            velocity_base,
            program_base,
            size: next,
        }
    }

    pub fn default_mt3_like() -> Self {
        Self::new(VocabConfig::default())
    }

    pub fn kind(&self, id: TokenId) -> Option<TokenKind> {
        if id == self.pad {
            return Some(TokenKind::Pad);
        }
        if id == self.bos {
            return Some(TokenKind::Bos);
        }
        if id == self.eos {
            return Some(TokenKind::Eos);
        }
        if id == self.sep {
            return Some(TokenKind::Sep);
        }
        if id >= self.time_shift_base && id < self.note_on_base {
            return Some(TokenKind::TimeShift);
        }
        if id >= self.note_on_base && id < self.note_off_base {
            return Some(TokenKind::NoteOn);
        }
        if id >= self.note_off_base && id < self.velocity_base {
            return Some(TokenKind::NoteOff);
        }
        if id >= self.velocity_base && id < self.program_base {
            return Some(TokenKind::Velocity);
        }
        if id >= self.program_base && id < self.size {
            return Some(TokenKind::Program);
        }
        None
    }

    pub fn time_shift(&self, bins: u32) -> Option<TokenId> {
        if bins < self.cfg.n_time_bins {
            Some(self.time_shift_base + bins)
        } else {
            None
        }
    }

    pub fn time_shift_bins(&self, id: TokenId) -> Option<u32> {
        if id >= self.time_shift_base && id < self.note_on_base {
            Some(id - self.time_shift_base)
        } else {
            None
        }
    }

    pub fn note_on(&self, pitch: u8) -> Option<TokenId> {
        let p = pitch as u32;
        if p < self.cfg.n_pitches {
            Some(self.note_on_base + p)
        } else {
            None
        }
    }

    pub fn note_on_pitch(&self, id: TokenId) -> Option<u8> {
        if id >= self.note_on_base && id < self.note_off_base {
            Some((id - self.note_on_base) as u8)
        } else {
            None
        }
    }

    pub fn note_off(&self, pitch: u8) -> Option<TokenId> {
        let p = pitch as u32;
        if p < self.cfg.n_pitches {
            Some(self.note_off_base + p)
        } else {
            None
        }
    }

    pub fn note_off_pitch(&self, id: TokenId) -> Option<u8> {
        if id >= self.note_off_base && id < self.velocity_base {
            Some((id - self.note_off_base) as u8)
        } else {
            None
        }
    }

    pub fn velocity(&self, vel: u8) -> TokenId {
        let bins = self.cfg.n_velocity.max(1);
        let v = if bins >= 128 {
            (vel as u32).min(bins - 1)
        } else {
            // quantize 0..127 → 0..bins-1
            ((vel as u32) * bins / 128).min(bins - 1)
        };
        self.velocity_base + v
    }

    pub fn velocity_value(&self, id: TokenId) -> Option<u8> {
        if id >= self.velocity_base && id < self.program_base {
            let bin = id - self.velocity_base;
            let bins = self.cfg.n_velocity.max(1);
            if bins >= 128 {
                Some(bin.min(127) as u8)
            } else {
                // expand bin center back to 0..127
                let v = ((bin as u64 * 127) / (bins as u64).max(1)) as u8;
                Some(v)
            }
        } else {
            None
        }
    }

    pub fn program(&self, program: u8) -> Option<TokenId> {
        let p = program as u32;
        if p < self.cfg.n_programs {
            Some(self.program_base + p)
        } else {
            None
        }
    }

    pub fn program_id(&self, id: TokenId) -> Option<u8> {
        if id >= self.program_base && id < self.size {
            Some((id - self.program_base) as u8)
        } else {
            None
        }
    }

    /// Max seconds representable by a single TimeShift token.
    pub fn max_shift_sec(&self) -> f64 {
        (self.cfg.n_time_bins.saturating_sub(1) as f64) * self.cfg.time_bin_sec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_ranges_are_disjoint() {
        let v = Vocab::default_mt3_like();
        assert_eq!(v.pad, 0);
        assert_eq!(v.bos, 1);
        assert_eq!(v.eos, 2);
        assert_eq!(v.kind(v.note_on(60).unwrap()), Some(TokenKind::NoteOn));
        assert_eq!(v.kind(v.note_off(60).unwrap()), Some(TokenKind::NoteOff));
        assert_eq!(v.kind(v.velocity(100)), Some(TokenKind::Velocity));
        assert_eq!(v.kind(v.time_shift(10).unwrap()), Some(TokenKind::TimeShift));
        assert_eq!(v.kind(v.program(0).unwrap()), Some(TokenKind::Program));
        assert!(v.size > 500);
    }
}
