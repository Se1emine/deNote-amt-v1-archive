//! Job L0: wav+notes → latent cache + token cache + manifest.jsonl.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use burn::{
    backend::NdArray,
    tensor::backend::Backend,
};
use log::{info, warn};
use serde::Deserialize;

use crate::device::{log_backend, resolve_device, DeviceKind};
use crate::phase1_config::CacheJobConfig;
use denote_audio::{load_mono_resampled, musicfm_log_mel, MelConfig, MelStats};
use denote_data::{
    notes_from_json, split_for_id, unique_programs, CacheManifest, LatentFeatureCache, LatentSample,
    ManifestEntry, NoteList, TokenFeatureCache,
};
use denote_encoder::{
    load_musicfm_numpy_dir, log_weight_inventory, mel_to_tensor, musicfm_weights_available,
    MusicFmEncoder,
};
#[derive(Debug, Clone)]
pub struct CacheL0Opts {
    pub config: PathBuf,
    pub manifest_out: Option<PathBuf>,
    pub device: String,
    pub input_dir: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
}

pub fn run_cache_l0(opts: CacheL0Opts) -> Result<()> {
    let kind = resolve_device(&opts.device)?;
    log_backend(kind, "Phase1 Job L0 cache");

    match kind {
        DeviceKind::Cpu => run_l0_backend::<NdArray>(opts, kind),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => run_l0_backend::<burn::backend::Cuda>(opts, kind),
        #[cfg(feature = "wgpu")]
        DeviceKind::Wgpu => run_l0_backend::<burn::backend::Wgpu>(opts, kind),
        #[cfg(not(feature = "cuda"))]
        DeviceKind::Cuda => unreachable!(),
        #[cfg(not(feature = "wgpu"))]
        DeviceKind::Wgpu => unreachable!(),
    }
}

fn run_l0_backend<B: Backend>(opts: CacheL0Opts, kind: DeviceKind) -> Result<()> {
    let cfg = CacheJobConfig::load(&opts.config)?;
    let input_dir = opts
        .input_dir
        .unwrap_or_else(|| PathBuf::from(&cfg.data.input_dir));
    let cache_dir = opts
        .cache_dir
        .unwrap_or_else(|| PathBuf::from(&cfg.data.cache_dir));
    let manifest_path = opts.manifest_out.unwrap_or_else(|| {
        cache_dir.join(&cfg.data.manifest_name)
    });

    fs::create_dir_all(cache_dir.join("latents"))?;
    fs::create_dir_all(cache_dir.join("tokens"))?;
    fs::create_dir_all(cache_dir.join("meta"))?;

    let device = B::Device::default();
    let enc_cfg = cfg.encoder_config();
    let encoder: MusicFmEncoder<B> = enc_cfg.init(&device);

    // E1.1: optional pretrained dump inventory (partial mapping; encoder stays random if no map).
    if let Some(ref wdir) = cfg.encoder.weights_dir {
        if musicfm_weights_available(wdir) {
            match load_musicfm_numpy_dir(wdir) {
                Ok((_map, report)) => {
                    log_weight_inventory(&report);
                    info!(
                        "E1.1 partial: numpy dump loaded into inventory ({} tensors). \
                         Burn scaffold does not apply full PT→module map yet; \
                         encoder uses init weights. Latent pipeline still valid.",
                        report.n_tensors
                    );
                    info!(
                        "encoder params (Burn module) ≈ count via forward dims d_model={}",
                        encoder.d_model()
                    );
                }
                Err(e) => warn!("weights dir present but failed to read: {e:#}"),
            }
        } else if cfg.encoder.allow_random_encoder {
            warn!(
                "weights_dir={} missing or incomplete — using random encoder (allow_random_encoder=true)",
                wdir
            );
        } else {
            bail!("weights_dir={} required but not available", wdir);
        }
    } else {
        info!(
            "No encoder.weights_dir — random/smoke encoder (d_model={})",
            encoder.d_model()
        );
    }

    let vocab = cfg.vocab();
    let mel_cfg = MelConfig::musicfm();
    let stats = MelStats::identity(mel_cfg.n_mels);
    let sr = cfg.audio.sample_rate;

    let clips = discover_clips(&input_dir)?;
    if clips.is_empty() {
        bail!(
            "no clips in {} (expect {{id}}.wav + {{id}}.notes.json). \
             Run: python tools/build_phase1_mini.py",
            input_dir.display()
        );
    }
    info!(
        "L0 export | clips={} cache={} backend={}",
        clips.len(),
        cache_dir.display(),
        kind.as_str()
    );

    let latent_cache = LatentFeatureCache::new(cache_dir.join("latents"));
    let token_cache = TokenFeatureCache::new(cache_dir.join("tokens"));
    let mut entries = Vec::new();
    let skip_existing = cfg.data.skip_existing;
    let mut n_skipped = 0usize;
    let mut n_failed = 0usize;
    let mut n_encoded = 0usize;

    for (i, clip) in clips.iter().enumerate() {
        let id = &clip.id;
        let lat_rel = format!("latents/{id}.lat.bin");
        let tok_rel = format!("tokens/{id}.tok.bin");
        let lat_path = cache_dir.join(&lat_rel);
        let tok_path = cache_dir.join(&tok_rel);
        let meta_path = cache_dir.join("meta").join(format!("{id}.json"));
        let piece_id = piece_id_from_clip_id(id);
        let split = split_for_id(&piece_id, cfg.data.train_ratio).to_string();

        if skip_existing && lat_path.exists() && tok_path.exists() {
            // Fast path: meta.json if present (no re-load latent body).
            if let Ok(entry) = manifest_entry_from_meta(
                id,
                &lat_rel,
                &tok_rel,
                &meta_path,
                &split,
                clip,
                sr,
            ) {
                entries.push(entry);
                n_skipped += 1;
                if n_skipped % 1000 == 0 {
                    info!("L0 skip progress {n_skipped}/{} …", clips.len());
                }
                continue;
            }
            if let Ok(sample) = latent_cache.load(id) {
                let notes = load_notes_for_clip(clip).unwrap_or_else(|_| NoteList {
                    track_id: id.clone(),
                    sr,
                    notes: vec![],
                });
                let programs = unique_programs(notes.notes.iter().map(|n| n.program));
                entries.push(ManifestEntry {
                    id: id.clone(),
                    latent: lat_rel.clone(),
                    tokens: tok_rel.clone(),
                    programs: programs.clone(),
                    n_instruments: programs.len().max(1),
                    split: split.clone(),
                    duration_sec: sample.meta.duration_sec,
                    sample_rate: sample.meta.sample_rate,
                    d_model: sample.meta.d_model,
                    n_frames: sample.meta.n_frames,
                    audio_path: Some(clip.wav_path.display().to_string()),
                    notes_path: clip.notes_path.as_ref().map(|p| p.display().to_string()),
                    tokens_by_group: Default::default(),
                    groups_present: vec![],
                });
                n_skipped += 1;
                continue;
            }
        }

        if (i + 1) % 50 == 0 || i == 0 {
            info!(
                "[{}/{}] encode {} (encoded={} skipped={} failed={})",
                i + 1,
                clips.len(),
                id,
                n_encoded,
                n_skipped,
                n_failed
            );
        }

        // Per-clip isolation: never abort whole L0 on one bad file.
        let result = (|| -> Result<ManifestEntry> {
            let audio = load_mono_resampled(&clip.wav_path, sr)
                .with_context(|| format!("load wav {}", clip.wav_path.display()))?;
            let duration_sec = audio.len() as f64 / sr as f64;
            if audio.is_empty() {
                bail!("empty audio");
            }

            let mel = musicfm_log_mel(&audio, &mel_cfg, &stats);
            let mel_t = mel_to_tensor::<B>(mel.as_slice(), mel.n_mels(), mel.n_frames(), &device);

            let latents = encoder.forward_mel(mel_t);
            let [b, t, d] = latents.dims();
            if b != 1 {
                bail!("batch dim {b} != 1");
            }
            let z: Vec<f32> = latents
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("latent to_vec: {e:?}"))?;
            let z = ensure_f32_vec(z, t * d)?;

            let sample = LatentSample::new(id, t, d, z)?.with_audio_meta(sr, duration_sec);
            latent_cache.save(&sample)?;

            let notes = load_notes_for_clip(clip)?;
            let programs = unique_programs(notes.notes.iter().map(|n| n.program));
            let tok = TokenFeatureCache::build_from_notes(
                id,
                &notes,
                &vocab,
                sr,
                duration_sec,
                Some(clip.wav_path.display().to_string()),
                None,
            );
            token_cache.save(&tok)?;

            let meta_json = serde_json::json!({
                "id": id,
                "programs": programs,
                "duration_sec": duration_sec,
                "n_frames": t,
                "d_model": d,
                "sample_rate": sr,
            });
            fs::write(&meta_path, serde_json::to_string_pretty(&meta_json)?)?;

            Ok(ManifestEntry {
                id: id.clone(),
                latent: lat_rel.clone(),
                tokens: tok_rel.clone(),
                programs: programs.clone(),
                n_instruments: programs.len().max(1),
                split: split.clone(),
                duration_sec,
                sample_rate: sr,
                d_model: d,
                n_frames: t,
                audio_path: Some(clip.wav_path.display().to_string()),
                notes_path: clip.notes_path.as_ref().map(|p| p.display().to_string()),
                tokens_by_group: Default::default(),
                groups_present: vec![],
            })
        })();

        match result {
            Ok(entry) => {
                entries.push(entry);
                n_encoded += 1;
            }
            Err(e) => {
                n_failed += 1;
                warn!("L0 skip failed clip {id}: {e:#}");
            }
        }
    }
    info!(
        "L0 encode summary | encoded={n_encoded} skipped={n_skipped} failed={n_failed} entries={}",
        entries.len()
    );
    if n_skipped > 0 {
        info!("L0 skip_existing: reused {n_skipped} clips");
    }
    if entries.is_empty() {
        bail!("L0 produced 0 manifest entries");
    }

    let manifest = CacheManifest {
        root: cache_dir.clone(),
        path: manifest_path.clone(),
    };
    manifest.write_all(&entries)?;

    let n_train = entries.iter().filter(|e| e.split == "train").count();
    let n_val = entries.iter().filter(|e| e.split == "val").count();
    info!(
        "L0 done | total={} train={} val={} manifest={}",
        entries.len(),
        n_train,
        n_val,
        manifest_path.display()
    );
    if n_train < 32 || n_val < 8 {
        warn!(
            "N_min soft miss: need ≥32 train + ≥8 val for hard E1.2; got train={n_train} val={n_val}. \
             Rebuild with: py -3 tools/build_phase1_mini.py --n 50"
        );
    }
    let _ = enc_cfg;
    Ok(())
}

struct ClipPaths {
    id: String,
    wav_path: PathBuf,
    notes_path: Option<PathBuf>,
}

fn discover_clips(input_dir: &Path) -> Result<Vec<ClipPaths>> {
    if !input_dir.is_dir() {
        bail!("input_dir is not a directory: {}", input_dir.display());
    }
    let mut clips = Vec::new();
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if let Some(stem) = name.strip_suffix(".wav") {
            let notes = input_dir.join(format!("{stem}.notes.json"));
            clips.push(ClipPaths {
                id: stem.to_string(),
                wav_path: path,
                notes_path: if notes.exists() { Some(notes) } else { None },
            });
        }
    }
    clips.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(clips)
}

fn load_notes_for_clip(clip: &ClipPaths) -> Result<NoteList> {
    if let Some(ref p) = clip.notes_path {
        notes_from_json(p)
    } else {
        // Empty notes → all-zero presence (still valid for pipeline).
        Ok(NoteList {
            track_id: clip.id.clone(),
            sr: 24_000,
            notes: vec![],
        })
    }
}

fn ensure_f32_vec(z: Vec<f32>, expected: usize) -> Result<Vec<f32>> {
    if z.len() != expected {
        bail!("latent flat len {} != T*D {}", z.len(), expected);
    }
    Ok(z)
}

#[derive(Debug, Deserialize)]
struct MetaSidecar {
    id: Option<String>,
    programs: Option<Vec<u8>>,
    duration_sec: Option<f64>,
    n_frames: Option<usize>,
    d_model: Option<usize>,
    sample_rate: Option<u32>,
}

fn manifest_entry_from_meta(
    id: &str,
    lat_rel: &str,
    tok_rel: &str,
    meta_path: &Path,
    split: &str,
    clip: &ClipPaths,
    default_sr: u32,
) -> Result<ManifestEntry> {
    let text = fs::read_to_string(meta_path)?;
    let m: MetaSidecar = serde_json::from_str(&text)?;
    let programs = m.programs.unwrap_or_default();
    let n_instruments = programs.len().max(1);
    Ok(ManifestEntry {
        id: id.to_string(),
        latent: lat_rel.to_string(),
        tokens: tok_rel.to_string(),
        programs,
        n_instruments,
        split: split.to_string(),
        duration_sec: m.duration_sec.unwrap_or(0.0),
        sample_rate: m.sample_rate.unwrap_or(default_sr),
        d_model: m.d_model.unwrap_or(1024),
        n_frames: m.n_frames.unwrap_or(0),
        audio_path: Some(clip.wav_path.display().to_string()),
        notes_path: clip.notes_path.as_ref().map(|p| p.display().to_string()),
        tokens_by_group: Default::default(),
        groups_present: vec![],
    })
}

/// Piece-level id for train/val split (Phase 3 leakage rule: no shared pieces).
/// Chunk ids look like `{piece}_c0003` → piece; otherwise full id.
fn piece_id_from_clip_id(id: &str) -> String {
    if let Some(pos) = id.rfind("_c") {
        let suffix = &id[pos + 2..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return id[..pos].to_string();
        }
    }
    id.to_string()
}

#[cfg(test)]
mod piece_split_tests {
    use super::piece_id_from_clip_id;

    #[test]
    fn strips_chunk_suffix() {
        assert_eq!(
            piece_id_from_clip_id("Track00001_mix0_c0042"),
            "Track00001_mix0"
        );
        assert_eq!(piece_id_from_clip_id("m0001_Foo_c0000"), "m0001_Foo");
        assert_eq!(piece_id_from_clip_id("clip_0001"), "clip_0001");
    }
}
