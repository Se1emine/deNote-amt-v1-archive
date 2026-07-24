//! GM program → challenge K=8 mapping (Phase 3 / master §2.6).

/// Challenge program names (same as `manifest::CHALLENGE_PROGRAM_NAMES`).
pub const CHALLENGE_K: usize = 8;

/// Map General MIDI program (0–127) to challenge index 0..7.
///
/// Returns `None` for drums / unmapped SFX that should be dropped from stem mixes.
///
/// | Challenge | Name     | GM families (approx) |
/// |-----------|----------|----------------------|
/// | 0         | Piano    | 0–7, 8–15, 16–23, 24–31 (keys/guitar→keys group) |
/// | 1         | Violin   | 40, 44–47, 48–51 |
/// | 2         | Viola    | 41 |
/// | 3         | Cello    | 42–43, 32–39 (bass→cello proxy) |
/// | 4         | Flute    | 72–79 |
/// | 5         | Bassoon  | 70–71, 64–67 (reed proxy) |
/// | 6         | Trombone | 56–63 |
/// | 7         | Oboe     | 68–69 |
pub fn gm_to_challenge(gm: u8) -> Option<u8> {
    if gm >= 128 {
        return None; // drum channel sentinel
    }
    match gm {
        0..=31 => Some(0),  // piano / chrom / organ / guitar → keys
        32..=39 => Some(3), // bass → cello proxy
        40 => Some(1),      // violin
        41 => Some(2),      // viola
        42..=43 => Some(3), // cello / contrabass
        44..=51 => Some(1), // ensemble strings → violin
        56..=63 => Some(6), // brass → trombone
        64..=67 => Some(5), // sax → wood proxy (bassoon group)
        68..=69 => Some(7), // oboe / english horn
        70..=71 => Some(5), // bassoon / clarinet
        72..=79 => Some(4), // flute / recorder / pipe
        // Choir / pads / FX: drop for challenge-like training
        _ => None,
    }
}

/// True if GM program is drums (channel 9 or program 128 convention in Slakh).
pub fn is_drum_program(gm: u8) -> bool {
    gm >= 128
}

/// Map challenge index back to a representative GM program (for FluidR3 re-synth notes).
pub fn challenge_to_gm(challenge: u8) -> u8 {
    match challenge {
        0 => 0,  // Acoustic Grand Piano
        1 => 40, // Violin
        2 => 41, // Viola
        3 => 42, // Cello
        4 => 73, // Flute
        5 => 70, // Bassoon
        6 => 57, // Trombone
        7 => 68, // Oboe
        _ => 0,
    }
}

/// Remap note program fields from GM → challenge; drop unmapped.
pub fn remap_notes_to_challenge(notes: &mut Vec<crate::note::NoteAnnotation>) {
    notes.retain_mut(|n| {
        if let Some(c) = gm_to_challenge(n.program) {
            n.program = c;
            true
        } else {
            false
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piano_and_violin_map() {
        assert_eq!(gm_to_challenge(0), Some(0));
        assert_eq!(gm_to_challenge(40), Some(1));
        assert_eq!(gm_to_challenge(41), Some(2));
        assert_eq!(gm_to_challenge(42), Some(3));
        assert_eq!(gm_to_challenge(73), Some(4));
        assert_eq!(gm_to_challenge(70), Some(5));
        assert_eq!(gm_to_challenge(57), Some(6));
        assert_eq!(gm_to_challenge(68), Some(7));
    }

    #[test]
    fn drums_and_fx_dropped() {
        assert_eq!(gm_to_challenge(128), None);
        assert_eq!(gm_to_challenge(103), None); // FX
    }
}
