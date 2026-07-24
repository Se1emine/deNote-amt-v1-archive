//! Curriculum sampling by `n_instruments` (Phase 3: 1 → 2 → 3).

use crate::manifest::ManifestEntry;

/// Curriculum stages: only samples with `n_instruments <= max` (and >=1).
#[derive(Debug, Clone)]
pub struct CurriculumSchedule {
    /// Absolute step where stage ends (exclusive upper for next).
    /// Stage 0: steps [0, s1) → max_inst=1
    /// Stage 1: [s1, s2) → max_inst=2
    /// Stage 2: [s2, ∞) → max_inst=3
    pub stage1_end: u64,
    pub stage2_end: u64,
    pub max_instruments_cap: usize,
}

impl Default for CurriculumSchedule {
    fn default() -> Self {
        Self {
            stage1_end: 500,
            stage2_end: 1500,
            max_instruments_cap: 3,
        }
    }
}

impl CurriculumSchedule {
    pub fn max_instruments_at_step(&self, step: u64) -> usize {
        if step < self.stage1_end {
            1
        } else if step < self.stage2_end {
            2
        } else {
            self.max_instruments_cap.max(3)
        }
    }

    pub fn stage_id(&self, step: u64) -> u32 {
        if step < self.stage1_end {
            1
        } else if step < self.stage2_end {
            2
        } else {
            3
        }
    }
}

/// Filter manifest entries eligible under curriculum at `step`.
pub fn filter_curriculum<'a>(
    entries: &'a [ManifestEntry],
    schedule: &CurriculumSchedule,
    step: u64,
) -> Vec<&'a ManifestEntry> {
    let max_n = schedule.max_instruments_at_step(step);
    entries
        .iter()
        .filter(|e| e.n_instruments >= 1 && e.n_instruments <= max_n)
        .collect()
}

/// Mean `n_instruments` of a batch (for logging `data/n_instruments_mean`).
pub fn mean_n_instruments(entries: &[&ManifestEntry]) -> f32 {
    if entries.is_empty() {
        return 0.0;
    }
    let s: usize = entries.iter().map(|e| e.n_instruments).sum();
    s as f32 / entries.len() as f32
}

/// Piece-level train/val split: hash piece_id (stable).
pub fn split_for_piece(piece_id: &str, train_ratio: f64) -> &'static str {
    crate::manifest::split_for_id(piece_id, train_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, n: usize) -> ManifestEntry {
        ManifestEntry {
            id: id.into(),
            latent: format!("latents/{id}.lat.bin"),
            tokens: format!("tokens/{id}.tok.bin"),
            programs: (0..n as u8).collect(),
            n_instruments: n,
            split: "train".into(),
            duration_sec: 4.0,
            sample_rate: 24_000,
            d_model: 128,
            n_frames: 48,
            audio_path: None,
            notes_path: None,
            tokens_by_group: Default::default(),
            groups_present: vec![],
        }
    }

    #[test]
    fn curriculum_stages() {
        let sch = CurriculumSchedule {
            stage1_end: 10,
            stage2_end: 20,
            max_instruments_cap: 3,
        };
        assert_eq!(sch.max_instruments_at_step(0), 1);
        assert_eq!(sch.max_instruments_at_step(10), 2);
        assert_eq!(sch.max_instruments_at_step(20), 3);
        let all = vec![entry("a", 1), entry("b", 2), entry("c", 3)];
        assert_eq!(filter_curriculum(&all, &sch, 0).len(), 1);
        assert_eq!(filter_curriculum(&all, &sch, 10).len(), 2);
        assert_eq!(filter_curriculum(&all, &sch, 20).len(), 3);
    }
}
