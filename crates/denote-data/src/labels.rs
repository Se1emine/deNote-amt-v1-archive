use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::note::NoteAnnotation;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelConfig {
    pub n_pitches: usize,
    pub midi_offset: u8,
    pub sample_rate: u32,
    pub hop_length: usize,
    pub onset_smear_frames: usize,
    pub frame_onset_boost_frames: usize,
    pub frame_onset_boost: f32,
}

impl Default for LabelConfig {
    fn default() -> Self {
        Self {
            n_pitches: 88,
            midi_offset: 21,
            sample_rate: 22050,
            hop_length: 256,
            onset_smear_frames: 1,
            frame_onset_boost_frames: 5,
            frame_onset_boost: 5.0,
        }
    }
}

impl LabelConfig {
    pub fn fps(&self) -> f64 {
        self.sample_rate as f64 / self.hop_length as f64
    }

    pub fn pitch_index(&self, midi: u8) -> Option<usize> {
        if midi < self.midi_offset {
            return None;
        }
        let idx = (midi - self.midi_offset) as usize;
        if idx < self.n_pitches {
            Some(idx)
        } else {
            None
        }
    }
}

/// Frame-level multi-task targets: shape each `[T, n_pitches]`.
#[derive(Debug, Clone)]
pub struct FrameTargets {
    pub onset: Array2<f32>,
    pub frame: Array2<f32>,
    pub offset: Array2<f32>,
    /// Per-element weights for frame BCE (onset boost).
    pub frame_weight: Array2<f32>,
}

/// Convert note events into frame-level onset / frame / offset targets.
pub fn notes_to_targets(
    notes: &[NoteAnnotation],
    n_frames: usize,
    cfg: &LabelConfig,
) -> FrameTargets {
    let p = cfg.n_pitches;
    let mut onset = Array2::<f32>::zeros((n_frames, p));
    let mut frame = Array2::<f32>::zeros((n_frames, p));
    let mut offset = Array2::<f32>::zeros((n_frames, p));
    let mut frame_weight = Array2::<f32>::ones((n_frames, p));
    let fps = cfg.fps();

    for n in notes {
        let Some(pi) = cfg.pitch_index(n.pitch_midi) else {
            continue;
        };
        if n.end_sec <= n.start_sec {
            continue;
        }
        let start_f = (n.start_sec * fps).round() as isize;
        let end_f = (n.end_sec * fps).round() as isize;
        if end_f <= start_f {
            continue;
        }

        // Onset smear
        let smear = cfg.onset_smear_frames as isize;
        for dt in -smear..=smear {
            let t = start_f + dt;
            if t >= 0 && (t as usize) < n_frames {
                let w = if dt == 0 { 1.0 } else { 0.5 };
                let cur = onset[[t as usize, pi]];
                if w > cur {
                    onset[[t as usize, pi]] = w;
                }
            }
        }

        // Frame activity [start, end)
        let t0 = start_f.max(0) as usize;
        let t1 = (end_f as usize).min(n_frames);
        for t in t0..t1 {
            frame[[t, pi]] = 1.0;
            let rel = t as isize - start_f;
            if rel >= 0 && (rel as usize) < cfg.frame_onset_boost_frames {
                frame_weight[[t, pi]] = cfg.frame_onset_boost;
            }
        }

        // Offset smear around end-1
        let off = end_f - 1;
        for dt in -smear..=smear {
            let t = off + dt;
            if t >= 0 && (t as usize) < n_frames {
                let w = if dt == 0 { 1.0 } else { 0.5 };
                let cur = offset[[t as usize, pi]];
                if w > cur {
                    offset[[t as usize, pi]] = w;
                }
            }
        }
    }

    FrameTargets {
        onset,
        frame,
        offset,
        frame_weight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::note::NoteAnnotation;

    #[test]
    fn single_note_labels() {
        let cfg = LabelConfig::default();
        let notes = [NoteAnnotation {
            start_sec: 0.5,
            end_sec: 1.0,
            pitch_midi: 60,
            velocity: 80,
            confidence: 1.0,
            program: 0,
        }];
        let n_frames = 200;
        let t = notes_to_targets(&notes, n_frames, &cfg);
        let pi = cfg.pitch_index(60).unwrap();
        let start = (0.5 * cfg.fps()).round() as usize;
        assert!(t.onset[[start, pi]] > 0.9);
        assert!(t.frame[[start, pi]] > 0.9);
    }
}
