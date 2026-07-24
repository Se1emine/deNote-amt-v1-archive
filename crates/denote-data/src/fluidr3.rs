//! FluidR3 MIDI → WAV via FluidSynth CLI (CPU, no Python).
//!
//! Expects `fluidsynth` on PATH or bundled at `data/sf2/fluidsynth-*/bin/fluidsynth.exe`.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use log::info;

use crate::note::NoteList;
use crate::piano_roll::load_midi_notes;
use crate::programs::{gm_to_challenge, remap_notes_to_challenge};
use denote_audio::{load_wav_mono_f32_max_sec, resample_linear, write_wav_mono_normalized};

/// Resolve bundled FluidSynth on Windows if present.
pub fn default_fluidsynth_bin(repo_root: &Path) -> PathBuf {
    let bundled = repo_root.join(
        "data/sf2/fluidsynth-v2.5.6-win10-x64-cpp11/bin/fluidsynth.exe",
    );
    if bundled.exists() {
        return bundled;
    }
    PathBuf::from("fluidsynth")
}

pub fn default_sf2(repo_root: &Path) -> PathBuf {
    repo_root.join("data/sf2/FluidR3_GM.sf2")
}

#[derive(Debug, Clone)]
pub struct FluidRenderOpts {
    pub fluidsynth: PathBuf,
    pub sf2: PathBuf,
    pub sample_rate: u32,
    /// Remap GM programs to challenge K=8 in notes JSON.
    pub remap_challenge: bool,
    /// Drop notes with unmapped programs when remapping.
    pub drop_unmapped: bool,
}

impl Default for FluidRenderOpts {
    fn default() -> Self {
        Self {
            fluidsynth: PathBuf::from("fluidsynth"),
            sf2: PathBuf::from("data/sf2/FluidR3_GM.sf2"),
            sample_rate: 24_000,
            remap_challenge: true,
            drop_unmapped: true,
        }
    }
}

/// Render MIDI → mono WAV at `sample_rate` + notes JSON (sidecar).
///
/// `out_wav` is the target path; notes written to same stem + `.notes.json`.
pub fn render_midi_fluidr3(
    midi: &Path,
    out_wav: &Path,
    opts: &FluidRenderOpts,
) -> Result<NoteList> {
    if !midi.exists() {
        bail!("midi not found: {}", midi.display());
    }
    if !opts.sf2.exists() {
        bail!(
            "SoundFont not found: {} (place FluidR3_GM.sf2 under data/sf2/)",
            opts.sf2.display()
        );
    }
    if let Some(parent) = out_wav.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // FluidSynth writes its own WAV (often stereo float/int). Use temp path then mono convert.
    let tmp = out_wav.with_extension("fs_tmp.wav");
    let mut child = Command::new(&opts.fluidsynth)
        .args([
            "-ni",
            "-F",
            tmp.to_str().unwrap_or("tmp.wav"),
            "-r",
            &opts.sample_rate.to_string(),
            // quieter
            "-g",
            "0.5",
        ])
        .arg(&opts.sf2)
        .arg(midi)
        .spawn()
        .with_context(|| {
            format!(
                "spawn fluidsynth ({}) — install or use bundled path",
                opts.fluidsynth.display()
            )
        })?;
    // Long MIDIs can hang/OOM; hard timeout for free/1070 path.
    let timeout = std::time::Duration::from_secs(90);
    let start = std::time::Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = std::fs::remove_file(&tmp);
                    bail!(
                        "fluidsynth timeout (>{}s) midi={}",
                        timeout.as_secs(),
                        midi.display()
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                bail!("fluidsynth wait: {e}");
            }
        }
    };
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        bail!(
            "fluidsynth failed (status={status}) midi={}",
            midi.display()
        );
    }

    // Never load multi-hour temps into RAM — first 120 s only.
    let max_sec = 120.0;
    let (samples, sr) = load_wav_mono_f32_max_sec(&tmp, max_sec)
        .with_context(|| format!("load {}", tmp.display()))?;
    let _ = std::fs::remove_file(&tmp);
    let mono = if sr != opts.sample_rate {
        resample_linear(&samples, sr, opts.sample_rate)
    } else {
        samples
    };
    write_wav_mono_normalized(out_wav, &mono, opts.sample_rate, 0.9)?;

    let mut notes = load_midi_notes(midi)?;
    notes.sr = opts.sample_rate;
    notes.track_id = out_wav
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("clip")
        .to_string();

    if opts.remap_challenge {
        if opts.drop_unmapped {
            remap_notes_to_challenge(&mut notes.notes);
        } else {
            for n in &mut notes.notes {
                if let Some(c) = gm_to_challenge(n.program) {
                    n.program = c;
                }
            }
        }
    }

    let notes_path = out_wav.with_extension("notes.json");
    let json = serde_json::to_string_pretty(&notes)?;
    std::fs::write(&notes_path, json)?;
    info!(
        "render ok | midi={} wav={} notes={} n={}",
        midi.display(),
        out_wav.display(),
        notes_path.display(),
        notes.notes.len()
    );
    Ok(notes)
}

/// Batch-render a list of MIDI files into `out_dir/{id}.wav` + `.notes.json`.
pub fn render_midi_batch(
    midis: &[(String, PathBuf)],
    out_dir: &Path,
    opts: &FluidRenderOpts,
) -> Result<Vec<(String, PathBuf, PathBuf)>> {
    std::fs::create_dir_all(out_dir)?;
    let mut done = Vec::new();
    for (id, midi) in midis {
        let wav = out_dir.join(format!("{id}.wav"));
        let notes_path = out_dir.join(format!("{id}.notes.json"));
        if wav.exists() && notes_path.exists() {
            info!("skip existing {id}");
            done.push((id.clone(), wav, notes_path));
            continue;
        }
        match render_midi_fluidr3(midi, &wav, opts) {
            Ok(_) => done.push((id.clone(), wav, notes_path)),
            Err(e) => {
                log::warn!("skip {} ({e:#})", midi.display());
            }
        }
    }
    Ok(done)
}
