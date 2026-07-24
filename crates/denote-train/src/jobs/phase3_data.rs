//! Phase 3 corpus: FluidR3 render + BabySlakh stem-mix → flat wav/notes for L0.
//!
//! CPU-only. Streams chunks to disk (no multi-GB RAM hold).

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use log::{info, warn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::json;

use crate::phase3_config::Phase3CorpusConfig;
use denote_audio::{load_wav_mono_f32_max_sec, resample_linear};
use denote_data::{
    chunk_audio_notes, default_fluidsynth_bin, default_sf2, discover_babyslakh,
    generate_stem_mix_chunks, render_midi_fluidr3, split_for_piece, write_chunks_flat, ChunkConfig,
    FluidRenderOpts, StemMixConfig,
};

#[derive(Debug, Clone)]
pub struct Phase3DataOpts {
    pub config: PathBuf,
}

pub fn run_phase3_data(opts: Phase3DataOpts) -> Result<()> {
    let cfg = Phase3CorpusConfig::load(&opts.config)?;
    let repo = PathBuf::from(&cfg.data.repo_root);
    let out_dir = PathBuf::from(&cfg.data.out_dir);
    fs::create_dir_all(&out_dir)?;
    if let Some(parent) = Path::new(&cfg.data.render_index).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut rng = StdRng::seed_from_u64(cfg.data.seed);
    let mut index_lines: Vec<String> = Vec::new();
    let mut n_chunks_total = 0usize;
    let mut n_by_inst = [0usize; 4];
    let mut n_fluid = 0usize;
    let mut n_stem = 0usize;

    let chunk_cfg = ChunkConfig {
        chunk_sec: cfg.chunk.chunk_sec,
        hop_sec: cfg.chunk.hop_sec,
        sample_rate: cfg.render.sample_rate,
        min_notes: cfg.chunk.min_notes,
        max_instruments: cfg.chunk.max_instruments,
        remap_gm: true,
    };

    // --- FluidR3 MIDI renders ---
    if cfg.render.enabled {
        if let Some(ref midi_dir) = cfg.data.midi_dir {
            let midi_root = PathBuf::from(midi_dir);
            if midi_root.is_dir() {
                let midis = collect_midis(&midi_root, cfg.data.max_midi, &mut rng)?;
                info!(
                    "FluidR3 | midis={} → stream chunks into {}",
                    midis.len(),
                    out_dir.display()
                );
                let fs_bin = cfg
                    .render
                    .fluidsynth
                    .as_ref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| default_fluidsynth_bin(&repo));
                let sf2 = cfg
                    .render
                    .sf2
                    .as_ref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| default_sf2(&repo));
                let ropts = FluidRenderOpts {
                    fluidsynth: fs_bin,
                    sf2,
                    sample_rate: cfg.render.sample_rate,
                    remap_challenge: true,
                    drop_unmapped: true,
                };
                let render_dir = out_dir.join("_renders");
                fs::create_dir_all(&render_dir)?;

                // Resume: skip MIDI whose first chunk already on disk
                let max_piece_sec = 120.0_f64;
                for (i, midi) in midis.iter().enumerate() {
                    let id = stable_midi_id(midi, i);
                    let probe = out_dir.join(format!("{id}_c0000.wav"));
                    if probe.exists() {
                        // Count existing chunks for this piece
                        let existing = count_piece_chunks(&out_dir, &id);
                        n_chunks_total += existing;
                        n_fluid += existing;
                        info!("skip existing piece {id} ({existing} chunks)");
                        continue;
                    }
                    let wav = render_dir.join(format!("{id}.wav"));
                    match render_midi_fluidr3(midi, &wav, &ropts) {
                        Ok(mut notes) => {
                            if notes.notes.is_empty() {
                                warn!("no challenge notes after remap: {}", midi.display());
                                let _ = fs::remove_file(&wav);
                                let _ = fs::remove_file(wav.with_extension("notes.json"));
                                continue;
                            }
                            // Stream only first max_piece_sec (never load multi-hour WAVs).
                            let (samples, sr) =
                                match load_wav_mono_f32_max_sec(&wav, max_piece_sec) {
                                    Ok(x) => x,
                                    Err(e) => {
                                        warn!("load render {}: {e:#}", wav.display());
                                        let _ = fs::remove_file(&wav);
                                        let _ = fs::remove_file(wav.with_extension("notes.json"));
                                        continue;
                                    }
                                };
                            let audio = if sr != cfg.render.sample_rate {
                                resample_linear(&samples, sr, cfg.render.sample_rate)
                            } else {
                                samples
                            };
                            notes.notes.retain(|n| n.start_sec < max_piece_sec);
                            for n in &mut notes.notes {
                                if n.end_sec > max_piece_sec {
                                    n.end_sec = max_piece_sec;
                                }
                            }
                            let piece = id.clone();
                            let chunks = chunk_audio_notes(&piece, &audio, &notes, &chunk_cfg);
                            let nc = chunks.len();
                            for c in &chunks {
                                let k = c.n_instruments.min(3);
                                n_by_inst[k] += 1;
                                let _ = split_for_piece(&c.piece_id, cfg.data.train_ratio);
                            }
                            match write_chunks_flat(&chunks, &out_dir) {
                                Ok(written) => {
                                    n_chunks_total += written;
                                    n_fluid += written;
                                }
                                Err(e) => {
                                    warn!("write chunks for {piece}: {e:#}");
                                }
                            }
                            index_lines.push(
                                json!({
                                    "kind": "fluidr3",
                                    "piece_id": piece,
                                    "midi": midi.display().to_string(),
                                    "n_chunks": nc,
                                    "n_notes": notes.notes.len(),
                                })
                                .to_string(),
                            );
                            // Drop full-length render (chunks on disk).
                            let _ = fs::remove_file(&wav);
                            let _ = fs::remove_file(wav.with_extension("notes.json"));
                            if (i + 1) % 10 == 0 || i + 1 == midis.len() {
                                info!(
                                    "FluidR3 progress {}/{} | chunks_so_far={}",
                                    i + 1,
                                    midis.len(),
                                    n_chunks_total
                                );
                            }
                        }
                        Err(e) => warn!("render fail {}: {e:#}", midi.display()),
                    }
                }
                // optional empty _renders
                let _ = fs::remove_dir_all(&render_dir);
            } else {
                warn!("midi_dir not found: {}", midi_root.display());
            }
        } else {
            info!("render.enabled but data.midi_dir unset — skip FluidR3");
        }
    }

    // --- BabySlakh stem mixes (stream per-track) ---
    if cfg.stem.enabled {
        if let Some(ref bs) = cfg.data.babyslakh_dir {
            let bs_root = PathBuf::from(bs);
            if bs_root.is_dir() {
                let tracks = discover_babyslakh(&bs_root)?;
                let sm_cfg = StemMixConfig {
                    sample_rate: cfg.render.sample_rate,
                    max_instruments: cfg.stem.max_instruments,
                    min_instruments: cfg.stem.min_instruments,
                    mixes_per_track: cfg.stem.mixes_per_track,
                    chunk: ChunkConfig {
                        chunk_sec: cfg.chunk.chunk_sec,
                        hop_sec: cfg.chunk.hop_sec,
                        sample_rate: cfg.render.sample_rate,
                        min_notes: cfg.chunk.min_notes,
                        max_instruments: cfg.chunk.max_instruments,
                        remap_gm: false,
                    },
                };
                // Process one track at a time to bound RAM
                for (ti, track) in tracks.iter().enumerate() {
                    let mixed = generate_stem_mix_chunks(
                        std::slice::from_ref(track),
                        &sm_cfg,
                        &mut rng,
                    )?;
                    for c in &mixed {
                        let k = c.n_instruments.min(3);
                        n_by_inst[k] += 1;
                        index_lines.push(
                            json!({
                                "kind": "stem_mix",
                                "piece_id": c.piece_id,
                                "id": c.id,
                                "n_instruments": c.n_instruments,
                                "programs": c.programs,
                            })
                            .to_string(),
                        );
                    }
                    let w = write_chunks_flat(&mixed, &out_dir)?;
                    n_chunks_total += w;
                    n_stem += w;
                    info!(
                        "stem track {}/{} {} → +{} chunks (total={})",
                        ti + 1,
                        tracks.len(),
                        track.track_id,
                        w,
                        n_chunks_total
                    );
                }
            } else {
                warn!("babyslakh_dir not found: {}", bs_root.display());
            }
        } else {
            info!("stem.enabled but data.babyslakh_dir unset — skip stem mix");
        }
    }

    if n_chunks_total == 0 {
        bail!(
            "phase3-data produced 0 chunks. Check midi_dir / babyslakh_dir / FluidSynth / SF2.\n\
             Config: {}",
            opts.config.display()
        );
    }

    let index_path = PathBuf::from(&cfg.data.render_index);
    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&index_path)
            .with_context(|| format!("write {}", index_path.display()))?;
        for line in &index_lines {
            writeln!(f, "{line}")?;
        }
        writeln!(
            f,
            "{}",
            json!({
                "kind": "summary",
                "n_chunks": n_chunks_total,
                "n_fluid_chunks": n_fluid,
                "n_stem_chunks": n_stem,
                "n_inst_1": n_by_inst[1],
                "n_inst_2": n_by_inst[2],
                "n_inst_3": n_by_inst[3],
                "out_dir": out_dir.display().to_string(),
                "max_midi": cfg.data.max_midi,
                "mixes_per_track": cfg.stem.mixes_per_track,
            })
        )?;
    }

    info!(
        "phase3-data done | chunks={} fluid={} stem={} | n_inst 1/2/3 = {}/{}/{} | out={} index={}",
        n_chunks_total,
        n_fluid,
        n_stem,
        n_by_inst[1],
        n_by_inst[2],
        n_by_inst[3],
        out_dir.display(),
        index_path.display()
    );
    if n_chunks_total < 500 {
        warn!(
            "below hard min 500 chunks ({n_chunks_total}) — increase max_midi / mixes_per_track"
        );
    } else if n_chunks_total < 5000 {
        info!(
            "above hard min; soft target 5k–50k: got {n_chunks_total} (disk/time limited)"
        );
    } else {
        info!("soft-target chunk volume reached: {n_chunks_total}");
    }
    info!(
        "Next: cargo run -p denote-train --release --features cuda -- --config configs/phase3_cache.toml --job cache-l0 --device cuda"
    );
    Ok(())
}

fn collect_midis(root: &Path, max_n: usize, rng: &mut StdRng) -> Result<Vec<PathBuf>> {
    let mut all = Vec::new();
    collect_midis_rec(root, &mut all)?;
    all.sort();
    if all.len() > max_n {
        all.shuffle(rng);
        all.truncate(max_n);
        all.sort();
    }
    Ok(all)
}

fn collect_midis_rec(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_midis_rec(&path, out)?;
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("mid") || e.eq_ignore_ascii_case("midi"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
    Ok(())
}

fn count_piece_chunks(out_dir: &Path, piece_id: &str) -> usize {
    let prefix = format!("{piece_id}_c");
    let mut n = 0usize;
    if let Ok(rd) = fs::read_dir(out_dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && name.ends_with(".wav") {
                n += 1;
            }
        }
    }
    n
}

fn stable_midi_id(midi: &Path, idx: usize) -> String {
    let stem = midi
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("midi");
    let safe: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(40)
        .collect();
    format!("m{idx:04}_{safe}")
}
