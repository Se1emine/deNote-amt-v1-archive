//! Constrained greedy AR helpers (FSM mask + argmax).
//!
//! Model forward lives in train/infer; this module only applies FSM constraints
//! to a logits vector and advances decode state.

use denote_tokens::{
    apply_constraint_mask, step, valid_next, DecodeState, TokenId, Vocab,
};

/// Pick greedy next token among FSM-legal candidates.
///
/// Returns `None` if no legal token exists (should not happen with PAD/EOS).
pub fn greedy_next_token(
    logits: &[f32],
    state: &DecodeState,
    vocab: &Vocab,
) -> Option<TokenId> {
    let mut masked = logits.to_vec();
    if masked.len() < vocab.size as usize {
        masked.resize(vocab.size as usize, f32::NEG_INFINITY);
    }
    apply_constraint_mask(&mut masked, state, vocab, -1e9);
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in masked.iter().enumerate().take(vocab.size as usize) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    if !best_v.is_finite() {
        // Fallback: EOS if legal else PAD
        if valid_next(state, vocab.eos, vocab) {
            return Some(vocab.eos);
        }
        return Some(vocab.pad);
    }
    Some(best_i as TokenId)
}

/// Run greedy selection loop given a step function that scores the current prefix.
///
/// `score_prefix(tokens) -> logits for last position` over vocab.
pub fn ar_decode_loop<F>(
    vocab: &Vocab,
    max_len: usize,
    primary_program: Option<u8>,
    mut score_prefix: F,
) -> Vec<TokenId>
where
    F: FnMut(&[TokenId]) -> Vec<f32>,
{
    let mut state = DecodeState::new();
    if let Some(p) = primary_program {
        state.program = p;
    }
    let mut tokens = vec![vocab.bos];
    step(&mut state, vocab.bos, vocab);

    // Optional: emit Program token if legal and known
    if let Some(p) = primary_program {
        if let Some(pid) = vocab.program(p) {
            if valid_next(&state, pid, vocab) {
                tokens.push(pid);
                step(&mut state, pid, vocab);
            }
        }
    }

    while tokens.len() < max_len && !state.finished {
        let logits = score_prefix(&tokens);
        let next = match greedy_next_token(&logits, &state, vocab) {
            Some(t) => t,
            None => break,
        };
        if !valid_next(&state, next, vocab) {
            break;
        }
        tokens.push(next);
        step(&mut state, next, vocab);
        if next == vocab.eos {
            break;
        }
    }
    if !tokens.iter().any(|&t| t == vocab.eos) {
        if valid_next(&state, vocab.eos, vocab) {
            tokens.push(vocab.eos);
        }
    }
    tokens
}

/// Presence gate: group active if any program in group has p > threshold.
pub fn group_active_from_presence(
    presence_probs: &[f32],
    group_programs: &[u8],
    threshold: f32,
) -> bool {
    group_programs.iter().any(|&p| {
        let i = p as usize;
        i < presence_probs.len() && presence_probs[i] > threshold
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use denote_tokens::{sequence_is_valid, Vocab};

    #[test]
    fn fsm_rejects_illegal_note_off_in_greedy() {
        let v = Vocab::default_mt3_like();
        let mut state = DecodeState::new();
        step(&mut state, v.bos, &v);
        // Logits that prefer illegal NoteOff
        let mut logits = vec![0.0f32; v.size as usize];
        let off = v.note_off(60).unwrap() as usize;
        logits[off] = 100.0;
        // Legal time-shift
        if let Some(ts) = v.time_shift(1) {
            logits[ts as usize] = 1.0;
        }
        let next = greedy_next_token(&logits, &state, &v).unwrap();
        assert_ne!(next, v.note_off(60).unwrap());
        assert!(valid_next(&state, next, &v));
    }

    #[test]
    fn ar_loop_produces_valid_sequence() {
        let v = Vocab::default_mt3_like();
        // Always push high mass on EOS when legal, else TimeShift, else Velocity path
        let tokens = ar_decode_loop(&v, 16, Some(0), |prefix| {
            let mut logits = vec![-10.0f32; v.size as usize];
            // Prefer finishing quickly after BOS (+ optional Program)
            if prefix.len() >= 2 {
                logits[v.eos as usize] = 50.0;
            } else if let Some(ts) = v.time_shift(1) {
                logits[ts as usize] = 50.0;
            }
            logits
        });
        assert!(sequence_is_valid(&tokens, &v), "tokens={tokens:?}");
        assert_eq!(tokens[0], v.bos);
        assert!(tokens.contains(&v.eos));
    }

    #[test]
    fn presence_gate() {
        let p = vec![0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!group_active_from_presence(&p, &[0], 0.5));
        assert!(group_active_from_presence(&p, &[1, 2, 3], 0.5));
        assert!(!group_active_from_presence(&p, &[6], 0.5));
    }
}
