//! Note-level metrics (mir_eval transcription style).
//!
//! Phase 3: multi-instrument onset F1 with **program + pitch + onset ±50 ms**
//! (challenge-aligned Multi Onset F1, MIROS ref 0.5998).

use std::collections::{BTreeMap, BTreeSet};

use denote_decode::NoteEvent;
use serde::{Deserialize, Serialize};

/// Reference note for legacy pitch+time evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefNote {
    pub start_sec: f64,
    pub end_sec: f64,
    pub pitch_midi: u8,
}

impl From<&NoteEvent> for RefNote {
    fn from(n: &NoteEvent) -> Self {
        Self {
            start_sec: n.start_sec,
            end_sec: n.end_sec,
            pitch_midi: n.pitch_midi,
        }
    }
}

/// Reference note for multi-instrument onset F1 (program required).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefNoteMulti {
    pub start_sec: f64,
    pub end_sec: f64,
    pub pitch_midi: u8,
    /// Challenge program index (0..K-1) or GM-mapped id used in labels.
    pub program: u8,
}

impl From<&NoteEvent> for RefNoteMulti {
    fn from(n: &NoteEvent) -> Self {
        Self {
            start_sec: n.start_sec,
            end_sec: n.end_sec,
            pitch_midi: n.pitch_midi,
            program: n.program,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricScores {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

impl MetricScores {
    pub fn from_counts(tp: usize, fp: usize, fn_: usize) -> Self {
        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Self {
            precision,
            recall,
            f1,
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub onset_f1: MetricScores,
    pub note_with_offset_f1: MetricScores,
    pub onset_tolerance_sec: f64,
    pub offset_ratio: f64,
    pub offset_min_sec: f64,
}

/// Per-instrument scores + macro/micro for multi-instrument onset F1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiInstrumentReport {
    /// Macro F1 = unweighted mean over instruments that appear in the reference.
    pub macro_f1: f64,
    pub macro_precision: f64,
    pub macro_recall: f64,
    /// Micro F1 pooled over all notes.
    pub micro: MetricScores,
    /// Per program id → scores (only instruments present in ref or est).
    pub per_instrument: BTreeMap<u8, MetricScores>,
    pub onset_tolerance_sec: f64,
    /// Challenge / paper reference for thesis tables.
    pub miros_multi_onset_f1_ref: f64,
}

impl MultiInstrumentReport {
    pub const MIROS_REF: f64 = 0.5998;
}

/// Greedy bipartite matching for note F1 (pitch + onset, optional offset).
///
/// - Onset-only: pitch exact + |onset_est - onset_ref| ≤ tolerance
/// - With offset: also |offset_est - offset_ref| ≤ max(offset_min, offset_ratio * ref_duration)
pub fn evaluate_notes(
    est: &[NoteEvent],
    reference: &[RefNote],
    onset_tolerance_sec: f64,
    with_offset: bool,
    offset_ratio: f64,
    offset_min_sec: f64,
) -> MetricScores {
    let mut ref_matched = vec![false; reference.len()];
    let mut tp = 0usize;

    let mut est_idx: Vec<usize> = (0..est.len()).collect();
    est_idx.sort_by(|&a, &b| {
        est[b]
            .confidence
            .partial_cmp(&est[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for ei in est_idx {
        let e = &est[ei];
        let mut best: Option<usize> = None;
        let mut best_err = f64::MAX;
        for (ri, r) in reference.iter().enumerate() {
            if ref_matched[ri] {
                continue;
            }
            if e.pitch_midi != r.pitch_midi {
                continue;
            }
            let onset_err = (e.start_sec - r.start_sec).abs();
            if onset_err > onset_tolerance_sec {
                continue;
            }
            if with_offset {
                let ref_dur = (r.end_sec - r.start_sec).max(1e-6);
                let tol = offset_min_sec.max(offset_ratio * ref_dur);
                let offset_err = (e.end_sec - r.end_sec).abs();
                if offset_err > tol {
                    continue;
                }
            }
            if onset_err < best_err {
                best_err = onset_err;
                best = Some(ri);
            }
        }
        if let Some(ri) = best {
            ref_matched[ri] = true;
            tp += 1;
        }
    }

    let fp = est.len().saturating_sub(tp);
    let fn_ = reference.len().saturating_sub(tp);
    MetricScores::from_counts(tp, fp, fn_)
}

pub fn evaluate_full(
    est: &[NoteEvent],
    reference: &[RefNote],
    onset_tolerance_sec: f64,
) -> EvalReport {
    let onset_f1 = evaluate_notes(est, reference, onset_tolerance_sec, false, 0.2, 0.05);
    let note_with_offset_f1 =
        evaluate_notes(est, reference, onset_tolerance_sec, true, 0.2, 0.05);
    EvalReport {
        onset_f1,
        note_with_offset_f1,
        onset_tolerance_sec,
        offset_ratio: 0.2,
        offset_min_sec: 0.05,
    }
}

/// Multi-instrument Note Onset F1: **program + pitch + onset** must all match.
///
/// Greedy match by est confidence (desc), bipartite one-to-one.
/// Macro = unweighted mean over instruments that appear in the reference.
pub fn evaluate_notes_multi_onset(
    est: &[NoteEvent],
    reference: &[RefNoteMulti],
    onset_tolerance_sec: f64,
) -> MultiInstrumentReport {
    // Micro (all notes)
    let micro = match_counts(est, reference, onset_tolerance_sec, None);

    // Instruments present in reference (macro denominator preference)
    let mut ref_progs: BTreeSet<u8> = BTreeSet::new();
    for r in reference {
        ref_progs.insert(r.program);
    }
    // Also report est-only programs as zero-recall rows if useful? Plan: macro over instruments in ref.
    let mut per_instrument: BTreeMap<u8, MetricScores> = BTreeMap::new();
    for &prog in &ref_progs {
        let scores = match_counts(est, reference, onset_tolerance_sec, Some(prog));
        per_instrument.insert(prog, scores);
    }

    let n = per_instrument.len().max(1) as f64;
    let mut sum_p = 0.0;
    let mut sum_r = 0.0;
    let mut sum_f = 0.0;
    if per_instrument.is_empty() {
        // No ref notes → undefined; report zeros
        return MultiInstrumentReport {
            macro_f1: 0.0,
            macro_precision: 0.0,
            macro_recall: 0.0,
            micro,
            per_instrument,
            onset_tolerance_sec,
            miros_multi_onset_f1_ref: MultiInstrumentReport::MIROS_REF,
        };
    }
    for s in per_instrument.values() {
        sum_p += s.precision;
        sum_r += s.recall;
        sum_f += s.f1;
    }

    MultiInstrumentReport {
        macro_f1: sum_f / n,
        macro_precision: sum_p / n,
        macro_recall: sum_r / n,
        micro,
        per_instrument,
        onset_tolerance_sec,
        miros_multi_onset_f1_ref: MultiInstrumentReport::MIROS_REF,
    }
}

fn match_counts(
    est: &[NoteEvent],
    reference: &[RefNoteMulti],
    onset_tolerance_sec: f64,
    only_program: Option<u8>,
) -> MetricScores {
    let ref_idx: Vec<usize> = reference
        .iter()
        .enumerate()
        .filter(|(_, r)| only_program.map(|p| r.program == p).unwrap_or(true))
        .map(|(i, _)| i)
        .collect();
    let est_idx_all: Vec<usize> = est
        .iter()
        .enumerate()
        .filter(|(_, e)| only_program.map(|p| e.program == p).unwrap_or(true))
        .map(|(i, _)| i)
        .collect();

    let mut ref_matched = vec![false; reference.len()];
    let mut tp = 0usize;

    let mut est_idx = est_idx_all;
    est_idx.sort_by(|&a, &b| {
        est[b]
            .confidence
            .partial_cmp(&est[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for ei in est_idx {
        let e = &est[ei];
        let mut best: Option<usize> = None;
        let mut best_err = f64::MAX;
        for &ri in &ref_idx {
            if ref_matched[ri] {
                continue;
            }
            let r = &reference[ri];
            if e.program != r.program {
                continue;
            }
            if e.pitch_midi != r.pitch_midi {
                continue;
            }
            let onset_err = (e.start_sec - r.start_sec).abs();
            if onset_err > onset_tolerance_sec {
                continue;
            }
            if onset_err < best_err {
                best_err = onset_err;
                best = Some(ri);
            }
        }
        if let Some(ri) = best {
            ref_matched[ri] = true;
            tp += 1;
        }
    }

    let n_est = est
        .iter()
        .filter(|e| only_program.map(|p| e.program == p).unwrap_or(true))
        .count();
    let n_ref = ref_idx.len();
    let fp = n_est.saturating_sub(tp);
    let fn_ = n_ref.saturating_sub(tp);
    MetricScores::from_counts(tp, fp, fn_)
}

#[cfg(test)]
mod tests {
    use super::*;
    use denote_decode::NoteEvent;

    fn ne(start: f64, pitch: u8, program: u8, conf: f32) -> NoteEvent {
        NoteEvent {
            start_sec: start,
            end_sec: start + 0.4,
            duration_sec: 0.4,
            pitch_midi: pitch,
            confidence: conf,
            velocity: 100,
            program,
        }
    }

    fn rf(start: f64, pitch: u8, program: u8) -> RefNoteMulti {
        RefNoteMulti {
            start_sec: start,
            end_sec: start + 0.4,
            pitch_midi: pitch,
            program,
        }
    }

    #[test]
    fn perfect_match_f1() {
        let notes = vec![NoteEvent {
            start_sec: 1.0,
            end_sec: 1.5,
            duration_sec: 0.5,
            pitch_midi: 60,
            confidence: 0.9,
            velocity: 100,
            program: 0,
        }];
        let refs = vec![RefNote {
            start_sec: 1.02,
            end_sec: 1.48,
            pitch_midi: 60,
        }];
        let s = evaluate_notes(&notes, &refs, 0.05, false, 0.2, 0.05);
        assert!((s.f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn multi_program_mismatch_not_tp() {
        let est = vec![ne(1.0, 60, 0, 1.0)];
        let refs = vec![rf(1.0, 60, 1)]; // same pitch/time, different program
        let r = evaluate_notes_multi_onset(&est, &refs, 0.05);
        assert_eq!(r.micro.true_positives, 0);
        assert_eq!(r.micro.false_positives, 1);
        assert_eq!(r.micro.false_negatives, 1);
        assert!((r.micro.f1).abs() < 1e-9);
    }

    #[test]
    fn multi_onset_49ms_match_51ms_miss() {
        let est_ok = vec![ne(1.0, 60, 0, 1.0)];
        let ref_ok = vec![rf(1.049, 60, 0)]; // 49 ms
        let r_ok = evaluate_notes_multi_onset(&est_ok, &ref_ok, 0.05);
        assert_eq!(r_ok.micro.true_positives, 1);
        assert!((r_ok.micro.f1 - 1.0).abs() < 1e-9);

        let est_miss = vec![ne(1.0, 60, 0, 1.0)];
        let ref_miss = vec![rf(1.051, 60, 0)]; // 51 ms
        let r_miss = evaluate_notes_multi_onset(&est_miss, &ref_miss, 0.05);
        assert_eq!(r_miss.micro.true_positives, 0);
    }

    #[test]
    fn macro_f1_hand_fixture() {
        // Inst 0: perfect 1 note → F1=1
        // Inst 1: 1 est wrong pitch → TP=0, FP=1, FN=1 → F1=0
        // Macro F1 = 0.5
        let est = vec![ne(0.0, 60, 0, 1.0), ne(0.0, 70, 1, 1.0)];
        let refs = vec![rf(0.0, 60, 0), rf(0.0, 64, 1)];
        let r = evaluate_notes_multi_onset(&est, &refs, 0.05);
        assert!((r.per_instrument.get(&0).unwrap().f1 - 1.0).abs() < 1e-9);
        assert!((r.per_instrument.get(&1).unwrap().f1).abs() < 1e-9);
        assert!((r.macro_f1 - 0.5).abs() < 1e-9);
        assert_eq!(r.micro.true_positives, 1);
        assert_eq!(r.micro.false_positives, 1);
        assert_eq!(r.micro.false_negatives, 1);
    }
}
