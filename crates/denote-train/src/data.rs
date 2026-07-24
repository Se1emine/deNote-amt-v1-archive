use anyhow::{bail, Result};
use burn::tensor::{backend::Backend, Tensor};
use log::info;
use rand::seq::SliceRandom;

use crate::config::AppConfig;
use denote_data::FeatureCache;

pub struct BatchCpu {
    /// Mel: flat `[B * 1 * n_mels * T]` in NCHW order
    pub mel: Vec<f32>,
    pub onset: Vec<f32>,
    pub frame: Vec<f32>,
    pub offset: Vec<f32>,
    pub frame_weight: Vec<f32>,
    pub batch: usize,
    pub n_mels: usize,
    pub n_frames: usize,
    pub n_pitches: usize,
}

/// Load cached samples and sample random crops into a CPU batch.
pub fn sample_batch(
    cfg: &AppConfig,
    rng: &mut impl rand::Rng,
    batch_size: Option<usize>,
) -> Result<BatchCpu> {
    let cache = FeatureCache::new(&cfg.train.data_dir);
    let ids = cache.list_ids()?;
    if ids.is_empty() {
        bail!(
            "no cached samples in {} — run feature build or --smoke first",
            cfg.train.data_dir
        );
    }

    let chunk = cfg.chunk_frames();
    let batch = batch_size.unwrap_or_else(|| cfg.effective_batch_size());
    let n_mels = cfg.audio.n_mels;
    let n_pitches = cfg.model.n_pitches;

    let mut mel = Vec::with_capacity(batch * n_mels * chunk);
    let mut onset = Vec::with_capacity(batch * chunk * n_pitches);
    let mut frame = Vec::with_capacity(batch * chunk * n_pitches);
    let mut offset = Vec::with_capacity(batch * chunk * n_pitches);
    let mut frame_weight = Vec::with_capacity(batch * chunk * n_pitches);

    for _ in 0..batch {
        let id = ids.choose(rng).expect("ids non-empty");
        let sample = cache.load(id)?;
        let max_start = sample.meta.n_frames.saturating_sub(chunk);
        let start = if max_start == 0 {
            0
        } else {
            rng.gen_range(0..=max_start)
        };
        let (mel_c, tgt) = FeatureCache::crop(&sample, chunk, start);
        push_mel_nchw(&mut mel, &mel_c, chunk);
        push_pad_tp(&mut onset, &tgt.onset, chunk, n_pitches);
        push_pad_tp(&mut frame, &tgt.frame, chunk, n_pitches);
        push_pad_tp(&mut offset, &tgt.offset, chunk, n_pitches);
        push_pad_tp(&mut frame_weight, &tgt.frame_weight, chunk, n_pitches);
    }

    Ok(BatchCpu {
        mel,
        onset,
        frame,
        offset,
        frame_weight,
        batch,
        n_mels,
        n_frames: chunk,
        n_pitches,
    })
}

fn push_mel_nchw(dst: &mut Vec<f32>, mel: &ndarray::Array2<f32>, chunk: usize) {
    let n_mels = mel.nrows();
    let t = mel.ncols();
    for m in 0..n_mels {
        for ti in 0..chunk {
            if ti < t {
                dst.push(mel[[m, ti]]);
            } else {
                dst.push(0.0);
            }
        }
    }
}

fn push_pad_tp(dst: &mut Vec<f32>, arr: &ndarray::Array2<f32>, chunk: usize, n_pitches: usize) {
    let t = arr.nrows();
    for ti in 0..chunk {
        for p in 0..n_pitches {
            if ti < t {
                dst.push(arr[[ti, p]]);
            } else {
                dst.push(0.0);
            }
        }
    }
}

pub fn batch_to_tensors<B: Backend>(
    batch: &BatchCpu,
    device: &B::Device,
) -> (
    Tensor<B, 4>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
) {
    let mel = Tensor::<B, 1>::from_floats(batch.mel.as_slice(), device).reshape([
        batch.batch,
        1,
        batch.n_mels,
        batch.n_frames,
    ]);
    let onset = Tensor::<B, 1>::from_floats(batch.onset.as_slice(), device).reshape([
        batch.batch,
        batch.n_frames,
        batch.n_pitches,
    ]);
    let frame = Tensor::<B, 1>::from_floats(batch.frame.as_slice(), device).reshape([
        batch.batch,
        batch.n_frames,
        batch.n_pitches,
    ]);
    let offset = Tensor::<B, 1>::from_floats(batch.offset.as_slice(), device).reshape([
        batch.batch,
        batch.n_frames,
        batch.n_pitches,
    ]);
    let frame_w = Tensor::<B, 1>::from_floats(batch.frame_weight.as_slice(), device).reshape([
        batch.batch,
        batch.n_frames,
        batch.n_pitches,
    ]);
    (mel, onset, frame, offset, frame_w)
}

/// Create one synthetic cached sample for smoke / overfit tests.
pub fn write_synthetic_smoke_cache(cfg: &AppConfig) -> Result<()> {
    use denote_audio::log_mel_spectrogram;
    use denote_data::{notes_to_targets, NoteAnnotation, NoteList, SampleMeta, save_notes_json};

    let sr = cfg.audio.sample_rate as usize;
    let duration_sec = 4.0;
    let n_samples = (sr as f64 * duration_sec) as usize;
    let mut audio = vec![0.0f32; n_samples];

    // simple sine notes at MIDI 60 and 64
    let notes_spec = [(0.5, 1.5, 60u8), (2.0, 3.0, 64u8)];
    for &(s, e, midi) in &notes_spec {
        let freq = 440.0 * 2f32.powf((midi as f32 - 69.0) / 12.0);
        let i0 = (s * sr as f64) as usize;
        let i1 = ((e * sr as f64) as usize).min(n_samples);
        for i in i0..i1 {
            let t = i as f32 / sr as f32;
            let env = 0.3;
            audio[i] += env * (2.0 * std::f32::consts::PI * freq * t).sin();
        }
    }

    let mel_cfg = cfg.mel_config();
    let label_cfg = cfg.label_config();
    let mel = log_mel_spectrogram(&audio, &mel_cfg);
    let notes = NoteList {
        track_id: "smoke_synth".into(),
        sr: cfg.audio.sample_rate,
        notes: notes_spec
            .iter()
            .map(|&(s, e, midi)| NoteAnnotation {
                start_sec: s,
                end_sec: e,
                pitch_midi: midi,
                velocity: 80,
                confidence: 1.0,
                program: 0,
            })
            .collect(),
    };
    let targets = notes_to_targets(&notes.notes, mel.n_frames(), &label_cfg);
    let sample = denote_data::CachedSample {
        meta: SampleMeta {
            track_id: "smoke_synth".into(),
            n_frames: mel.n_frames(),
            n_mels: mel.n_mels(),
            n_pitches: label_cfg.n_pitches,
            sample_rate: mel_cfg.sample_rate,
            hop_length: mel_cfg.hop_length,
        },
        mel: mel.data,
        targets,
    };
    let cache = FeatureCache::new(&cfg.train.data_dir);
    cache.save(&sample)?;
    // Side-car JSON for debugging labels (also keeps save_notes_json in use).
    let json_path = cache.root.join("smoke_synth.notes.json");
    save_notes_json(&json_path, &notes)?;
    info!(
        "Wrote synthetic smoke sample → {} ({} frames)",
        cache.sample_path("smoke_synth").display(),
        sample.meta.n_frames
    );
    Ok(())
}
