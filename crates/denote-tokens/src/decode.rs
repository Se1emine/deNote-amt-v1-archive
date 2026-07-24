//! Token sequences → note annotations.

use std::collections::HashMap;

use crate::encode::TokenNote;
use crate::vocab::{TokenId, TokenKind, Vocab};

/// Decode tokens produced by [`crate::encode::encode_notes`] (or the model).
///
/// Stops at the first `EOS`. Ignores `PAD`. Applies open-note close at EOS.
pub fn decode_tokens(tokens: &[TokenId], vocab: &Vocab) -> Vec<TokenNote> {
    let bin = vocab.cfg.time_bin_sec.max(1e-9);
    let mut t_sec = 0.0_f64;
    let mut pending_vel: Option<u8> = None;
    let mut program: u8 = 0;
    // pitch -> (start_sec, velocity, program)
    let mut open: HashMap<u8, (f64, u8, u8)> = HashMap::new();
    let mut notes: Vec<TokenNote> = Vec::new();

    for &id in tokens {
        match vocab.kind(id) {
            Some(TokenKind::Pad) | Some(TokenKind::Bos) | Some(TokenKind::Sep) => {}
            Some(TokenKind::Eos) => {
                // close remaining notes at current time
                for (pitch, (start, vel, prog)) in open.drain() {
                    if t_sec > start {
                        notes.push(TokenNote {
                            start_sec: start,
                            end_sec: t_sec,
                            pitch_midi: pitch,
                            velocity: vel,
                            program: prog,
                        });
                    }
                }
                break;
            }
            Some(TokenKind::TimeShift) => {
                if let Some(bins) = vocab.time_shift_bins(id) {
                    t_sec += bins as f64 * bin;
                }
            }
            Some(TokenKind::Velocity) => {
                pending_vel = vocab.velocity_value(id);
            }
            Some(TokenKind::Program) => {
                if let Some(p) = vocab.program_id(id) {
                    program = p;
                }
            }
            Some(TokenKind::NoteOn) => {
                if let Some(pitch) = vocab.note_on_pitch(id) {
                    let vel = pending_vel.unwrap_or(80).max(1);
                    pending_vel = None;
                    // re-trigger: close previous if still open
                    if let Some((start, v, prog)) = open.insert(pitch, (t_sec, vel, program)) {
                        if t_sec > start {
                            notes.push(TokenNote {
                                start_sec: start,
                                end_sec: t_sec,
                                pitch_midi: pitch,
                                velocity: v,
                                program: prog,
                            });
                        }
                    }
                }
            }
            Some(TokenKind::NoteOff) => {
                if let Some(pitch) = vocab.note_off_pitch(id) {
                    if let Some((start, vel, prog)) = open.remove(&pitch) {
                        let end = t_sec.max(start + bin); // at least one bin
                        notes.push(TokenNote {
                            start_sec: start,
                            end_sec: end,
                            pitch_midi: pitch,
                            velocity: vel,
                            program: prog,
                        });
                    }
                }
            }
            None => {}
        }
    }

    // if sequence ended without EOS, still close opens
    for (pitch, (start, vel, prog)) in open.drain() {
        if t_sec > start {
            notes.push(TokenNote {
                start_sec: start,
                end_sec: t_sec,
                pitch_midi: pitch,
                velocity: vel,
                program: prog,
            });
        }
    }

    notes.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch_midi.cmp(&b.pitch_midi))
    });
    notes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode_notes;
    use crate::vocab::Vocab;

    #[test]
    fn roundtrip_monophonic() {
        let v = Vocab::default_mt3_like();
        let notes = vec![TokenNote {
            start_sec: 0.20,
            end_sec: 0.80,
            pitch_midi: 64,
            velocity: 90,
            program: 0,
        }];
        let ids = encode_notes(&notes, &v);
        let back = decode_tokens(&ids, &v);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].pitch_midi, 64);
        assert!((back[0].start_sec - 0.20).abs() < 0.015);
        assert!((back[0].end_sec - 0.80).abs() < 0.015);
        assert_eq!(back[0].velocity, 90);
    }

    #[test]
    fn roundtrip_polyphonic() {
        let v = Vocab::default_mt3_like();
        let notes = vec![
            TokenNote {
                start_sec: 0.0,
                end_sec: 0.5,
                pitch_midi: 60,
                velocity: 70,
                program: 0,
            },
            TokenNote {
                start_sec: 0.1,
                end_sec: 0.6,
                pitch_midi: 64,
                velocity: 80,
                program: 0,
            },
            TokenNote {
                start_sec: 0.1,
                end_sec: 0.4,
                pitch_midi: 67,
                velocity: 85,
                program: 0,
            },
        ];
        let ids = encode_notes(&notes, &v);
        let back = decode_tokens(&ids, &v);
        assert_eq!(back.len(), 3);
        let pitches: Vec<u8> = {
            let mut p: Vec<_> = back.iter().map(|n| n.pitch_midi).collect();
            p.sort();
            p
        };
        assert_eq!(pitches, vec![60, 64, 67]);
    }

    #[test]
    fn empty_notes_only_bos_eos() {
        let v = Vocab::default_mt3_like();
        let ids = encode_notes(&[], &v);
        assert_eq!(ids, vec![v.bos, v.eos]);
        let back = decode_tokens(&ids, &v);
        assert!(back.is_empty());
    }
}
