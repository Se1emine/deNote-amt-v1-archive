//! MIDI-event tokenization for Denote-MIROS seq2seq transcription.
//!
//! Unlike frame-level piano-roll (Basic Pitch / Onsets-and-Frames), the model
//! emits a sequence of discrete MIDI-like tokens:
//! `BOS → [Program] → (TimeShift | Velocity+NoteOn | NoteOff)* → EOS`.
//!
//! The FSM in [`fsm`] constrains autoregressive decoding to legal sequences
//! (improvement over unconstrained MT3/MIROS decoding).

mod decode;
mod encode;
mod fsm;
mod vocab;

pub use decode::decode_tokens;
pub use encode::{encode_notes, pad_tokens, token_mask, TokenNote};
pub use fsm::{
    allowed_mask, apply_constraint_mask, sequence_is_valid, step, valid_next, DecodeState,
};
pub use vocab::{TokenId, TokenKind, Vocab, VocabConfig};
