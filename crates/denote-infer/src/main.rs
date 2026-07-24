//! Inference: modular multi-decoder (Phase 2) or legacy CRNN.
//!
//! Modular (primary Phase 2):
//! ```text
//! cargo run -p denote-infer --release -- `
//!   --job-a runs/jobA_smoke/checkpoints/last `
//!   --decoders runs/jobB_g0/checkpoints/last/decoder_g0,runs/jobB_g1/checkpoints/last/decoder_g1 `
//!   --groups configs/groups_challenge8.toml `
//!   --latent data/cache/phase1_mini/latents/clip_0002.lat.bin `
//!   --presence-threshold 0.5 `
//!   --out out/notes.json
//! ```
//!
//! Legacy CRNN:
//! ```text
//! cargo run -p denote-infer --release -- --config configs/v1_synth.toml \
//!   --checkpoint runs/synth/best.mpk --audio path/to.wav --out notes.json
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use burn::{
    backend::NdArray,
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{activation::sigmoid, Int, Tensor},
};
use clap::Parser;
use log::info;
use serde::Deserialize;

use denote_audio::{
    load_mono_f32, log_mel_spectrogram, resample_linear, resample_sinc, MelConfig,
};
use denote_decode::{
    ar_decode_loop, decode_notes, detokenize_merge, group_active_from_presence, DecodeConfig,
    MergeConfig, NoteEvent, ProgramNoteEvent,
};
use denote_model::{
    AdapterConfig, CrnnConfig, DenoteCrnn, DenoteModular, DenoteModularConfig, GroupsConfig,
    PresenceHeadConfig, TokenDecoderConfig,
};
use denote_tokens::{decode_tokens, Vocab, VocabConfig};

type B = NdArray;

#[derive(Parser, Debug)]
#[command(name = "denote-infer")]
struct Args {
    /// Legacy CRNN config TOML
    #[arg(long)]
    config: Option<PathBuf>,

    /// Legacy CRNN checkpoint stem
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Audio file (legacy CRNN or modular with encoder — not required if --latent)
    #[arg(long)]
    audio: Option<PathBuf>,

    #[arg(long, default_value = "notes.json")]
    out: PathBuf,

    #[arg(long)]
    midi: Option<PathBuf>,

    // --- Phase 2 modular ---
    /// Job A modular dir (`adapter.mpk` + `presence.mpk`)
    #[arg(long)]
    job_a: Option<PathBuf>,

    /// Comma-separated decoder stems: `.../decoder_g0,.../decoder_g1` (or dir with decoder_g{N}.mpk)
    #[arg(long)]
    decoders: Option<String>,

    /// Groups TOML
    #[arg(long, default_value = "configs/groups_challenge8.toml")]
    groups: PathBuf,

    /// Latent bin path (`*.lat.bin`) for modular path without encoder
    #[arg(long)]
    latent: Option<PathBuf>,

    #[arg(long, default_value_t = 0.5)]
    presence_threshold: f32,

    /// Decode all groups regardless of presence
    #[arg(long, default_value_t = false)]
    force_all: bool,

    #[arg(long, default_value_t = 256)]
    max_decode_len: usize,

    #[arg(long, default_value_t = 128)]
    d_model: usize,

    #[arg(long, default_value_t = 8)]
    presence_k: usize,

    #[arg(long, default_value_t = 2)]
    decoder_layers: usize,

    #[arg(long, default_value_t = 4)]
    decoder_heads: usize,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    if args.job_a.is_some() || args.decoders.is_some() || args.latent.is_some() {
        run_modular_infer(&args)?;
    } else {
        run_legacy_crnn(&args)?;
    }
    Ok(())
}

fn run_modular_infer(args: &Args) -> Result<()> {
    let job_a = args
        .job_a
        .as_ref()
        .context("--job-a required for modular infer")?;
    let decoders_arg = args
        .decoders
        .as_ref()
        .context("--decoders required for modular infer")?;
    let latent_path = args
        .latent
        .as_ref()
        .context("--latent required for modular infer (encoder path not in Phase 2 hard scope)")?;

    let groups = if args.groups.exists() {
        GroupsConfig::load(&args.groups).map_err(|e| anyhow::anyhow!(e))?
    } else {
        info!("groups file missing, using challenge8 defaults");
        GroupsConfig::challenge8()
    };

    let vocab = Vocab::new(VocabConfig::default());
    let vocab_size = vocab.size as usize;
    let d_model = args.d_model;
    let k = args.presence_k;
    let n_groups = groups.n_groups().max(1);
    let device = Default::default();

    let model_cfg = DenoteModularConfig {
        adapter: AdapterConfig::new(d_model, k)
            .with_n_pitches_aux(128)
            .with_dropout(0.0),
        presence: PresenceHeadConfig::new(d_model, k).with_hidden(d_model),
        decoder: TokenDecoderConfig::new(
            vocab_size,
            d_model,
            args.decoder_layers.max(1),
            args.decoder_heads.max(1),
            d_model * 2,
        )
        .with_dropout(0.0),
        n_groups,
    };
    let mut model: DenoteModular<B> = model_cfg.init(&device);
    model
        .load_job_a(job_a, &device)
        .map_err(|e| anyhow::anyhow!(e))?;

    let decoder_stems = parse_decoder_list(decoders_arg, n_groups)?;
    for (g, stem) in decoder_stems.iter().enumerate() {
        if let Some(s) = stem {
            if path_mpk_exists(s) {
                info!("Load decoder_g{g} from {}", s.display());
                model
                    .load_decoder_g(g, s, &device)
                    .map_err(|e| anyhow::anyhow!(e))?;
            } else {
                info!("decoder_g{g} missing at {} — skip group", s.display());
            }
        }
    }

    // Load latent [T, D] row-major → [1, T, D]
    let (z_data, n_frames, d) = load_latent_bin(latent_path, d_model)?;
    if d != d_model {
        info!("latent d_model={d} vs config {d_model}; using latent dim");
    }
    let z = Tensor::<B, 1>::from_floats(z_data.as_slice(), &device).reshape([1, n_frames, d]);

    // Presence: zero multi-hot probe (neutral), then gate by max p in group
    let multi_zero = Tensor::<B, 1>::from_floats(vec![0.0f32; k].as_slice(), &device).reshape([1, k]);
    let presence_logits = model.presence_from_latents(z.clone(), multi_zero);
    let presence_probs = sigmoid(presence_logits)
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();
    info!(
        "presence probs (K={}): {:?}",
        k,
        presence_probs
            .iter()
            .map(|p| format!("{p:.2}"))
            .collect::<Vec<_>>()
    );

    let merge_cfg = MergeConfig::default();
    let mut group_notes: Vec<(u32, Vec<denote_tokens::TokenNote>)> = Vec::new();
    let mut active_groups = Vec::new();

    for gdef in &groups.groups {
        let g = gdef.id as usize;
        if g >= model.n_groups() {
            continue;
        }
        let stem_ok = decoder_stems
            .get(g)
            .and_then(|s| s.as_ref())
            .map(|s| path_mpk_exists(s))
            .unwrap_or(false);
        if !stem_ok {
            info!(
                "group {} ({}) — no decoder checkpoint loaded, skip",
                gdef.id, gdef.name
            );
            continue;
        }

        // force_all only bypasses presence gate, not missing weights
        let active = args.force_all
            || group_active_from_presence(&presence_probs, &gdef.programs, args.presence_threshold);
        if !active {
            info!("group {} ({}) inactive — skip", gdef.id, gdef.name);
            continue;
        }
        active_groups.push(gdef.id);
        info!("group {} ({}) AR decode…", gdef.id, gdef.name);

        // Condition on programs of this group with high presence (or all group programs)
        let mut mh = vec![0.0f32; k];
        for &p in &gdef.programs {
            let i = p as usize;
            if i < k && (args.force_all || presence_probs.get(i).copied().unwrap_or(0.0) > args.presence_threshold) {
                mh[i] = 1.0;
            }
        }
        // If none flagged, use primary program of group
        if mh.iter().all(|&x| x == 0.0) {
            if let Some(p) = gdef.programs.first() {
                let i = *p as usize;
                if i < k {
                    mh[i] = 1.0;
                }
            }
        }
        let multi =
            Tensor::<B, 1>::from_floats(mh.as_slice(), &device).reshape([1, k]);
        let memory = model.adapter_memory(z.clone(), multi, true);

        let primary = gdef.programs.first().copied();
        let decoder = model.decoder(g);
        let tokens = ar_decode_loop(&vocab, args.max_decode_len, primary, |prefix| {
            score_prefix_logits(decoder, &memory, prefix, vocab_size)
        });
        let notes = decode_tokens(&tokens, &vocab);
        info!("  group {} → {} notes ({} tokens)", gdef.id, notes.len(), tokens.len());
        group_notes.push((gdef.id, notes));
    }

    let merged = detokenize_merge(&group_notes, &merge_cfg);
    let events: Vec<NoteEvent> = merged.iter().map(ProgramNoteEvent::to_note_event).collect();
    info!(
        "Merged {} notes from groups {:?}",
        events.len(),
        active_groups
    );

    let payload = serde_json::json!({
        "model": "denote-modular-v1",
        "latent": latent_path.display().to_string(),
        "job_a": job_a.display().to_string(),
        "presence_threshold": args.presence_threshold,
        "force_all": args.force_all,
        "active_groups": active_groups,
        "presence_probs": presence_probs,
        "notes": events,
    });
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&args.out, serde_json::to_string_pretty(&payload)?)?;
    info!("Wrote {}", args.out.display());

    if let Some(ref midi_path) = args.midi {
        write_midi(&events, midi_path)?;
        info!("Wrote {}", midi_path.display());
    }
    Ok(())
}

fn score_prefix_logits(
    decoder: &denote_model::TokenDecoder<B>,
    memory: &Tensor<B, 3>,
    prefix: &[u32],
    vocab_size: usize,
) -> Vec<f32> {
    let device = memory.device();
    let s = prefix.len().max(1);
    let tokens = Tensor::<B, 1, Int>::from_ints(prefix, &device).reshape([1, s]);
    let logits = decoder.forward(tokens, memory.clone()); // [1, S, V]
    let data = logits.into_data().to_vec::<f32>().unwrap_or_default();
    let v = vocab_size;
    if data.len() < v {
        return vec![0.0; v];
    }
    // last position logits
    let start = (s - 1) * v;
    data[start..start + v].to_vec()
}

fn parse_decoder_list(arg: &str, n_groups: usize) -> Result<Vec<Option<PathBuf>>> {
    let mut out = vec![None; n_groups];
    for part in arg.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let p = PathBuf::from(part);
        // If directory: look for decoder_g{N}
        if p.is_dir() {
            for g in 0..n_groups {
                let stem = p.join(format!("decoder_g{g}"));
                if path_mpk_exists(&stem) {
                    out[g] = Some(stem);
                } else {
                    let alt = p.join("decoder");
                    if g == 0 && path_mpk_exists(&alt) {
                        out[0] = Some(alt);
                    }
                }
            }
            continue;
        }
        // Stem path: try to parse group from filename
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if let Some(rest) = name.strip_prefix("decoder_g") {
            if let Ok(g) = rest.parse::<usize>() {
                if g < n_groups {
                    out[g] = Some(p.clone());
                }
                continue;
            }
        }
        if name == "decoder" || name.starts_with("decoder") {
            out[0] = Some(p);
        } else {
            // parent is last/ and name is decoder_gN.mpk stripped
            out[0] = Some(p);
        }
    }
    Ok(out)
}

fn load_latent_bin(path: &Path, expected_d: usize) -> Result<(Vec<f32>, usize, usize)> {
    // Prefer LatentFeatureCache format via denote_data if available
    // Layout: bincode LatentSample or raw? Check latent_cache.
    use denote_data::LatentFeatureCache;
    if let Some(parent) = path.parent() {
        let id = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .trim_end_matches(".lat.bin")
            .to_string();
        let cache = LatentFeatureCache::new(parent);
        if let Ok(sample) = cache.load(&id) {
            let t = sample.meta.n_frames;
            let d = sample.meta.d_model;
            return Ok((sample.z, t, d));
        }
    }
    // Fallback: raw f32 LE assuming square-ish reshape with expected_d
    let bytes = fs::read(path).with_context(|| format!("read latent {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("latent file size not multiple of 4");
    }
    let mut z = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        z.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let d = expected_d.max(1);
    if z.len() % d != 0 {
        bail!(
            "latent len {} not divisible by d_model {}",
            z.len(),
            d
        );
    }
    let t = z.len() / d;
    Ok((z, t, d))
}

fn path_mpk_exists(stem: &Path) -> bool {
    stem.with_extension("mpk").exists()
        || PathBuf::from(format!("{}.mpk", stem.display())).exists()
}

// --- Legacy CRNN path ---

#[derive(Debug, Deserialize)]
struct InferToml {
    audio: AudioSec,
    model: ModelSec,
    decode: DecodeSec,
}

#[derive(Debug, Deserialize)]
struct AudioSec {
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
    mel_eps: f32,
}

#[derive(Debug, Deserialize)]
struct ModelSec {
    n_pitches: usize,
    midi_offset: u8,
    feat_dim: usize,
    onset_hidden: usize,
    frame_hidden: usize,
    offset_hidden: usize,
    dropout: f64,
}

#[derive(Debug, Deserialize)]
struct DecodeSec {
    onset_threshold: f32,
    frame_threshold: f32,
    offset_threshold: f32,
    min_note_ms: f64,
    merge_gap_ms: f64,
}

fn run_legacy_crnn(args: &Args) -> Result<()> {
    let config = args
        .config
        .as_ref()
        .context("legacy path requires --config (or use modular --job-a/--decoders/--latent)")?;
    let checkpoint = args
        .checkpoint
        .as_ref()
        .context("legacy path requires --checkpoint")?;
    let audio = args
        .audio
        .as_ref()
        .context("legacy path requires --audio")?;

    let text = fs::read_to_string(config)?;
    let cfg: InferToml = toml::from_str(&text)?;

    let device = Default::default();
    let model_cfg = CrnnConfig::new()
        .with_n_mels(cfg.audio.n_mels)
        .with_n_pitches(cfg.model.n_pitches)
        .with_feat_dim(cfg.model.feat_dim)
        .with_onset_hidden(cfg.model.onset_hidden)
        .with_frame_hidden(cfg.model.frame_hidden)
        .with_offset_hidden(cfg.model.offset_hidden)
        .with_dropout(cfg.model.dropout);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let model: DenoteCrnn<B> = model_cfg
        .init(&device)
        .load_file(
            checkpoint.to_string_lossy().as_ref(),
            &recorder,
            &device,
        )
        .with_context(|| format!("load checkpoint {}", checkpoint.display()))?;

    let (raw, sr) = load_mono_f32(audio)?;
    let audio_pcm = if sr == cfg.audio.sample_rate {
        raw
    } else {
        match resample_sinc(&raw, sr, cfg.audio.sample_rate) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("sinc resample failed ({e}), falling back to linear");
                resample_linear(&raw, sr, cfg.audio.sample_rate)
            }
        }
    };

    let mel_cfg = MelConfig {
        sample_rate: cfg.audio.sample_rate,
        n_fft: cfg.audio.n_fft,
        hop_length: cfg.audio.hop_length,
        n_mels: cfg.audio.n_mels,
        fmin: cfg.audio.fmin,
        fmax: cfg.audio.fmax,
        mel_eps: cfg.audio.mel_eps,
    };
    let mel = log_mel_spectrogram(&audio_pcm, &mel_cfg);
    let n_frames = mel.n_frames();
    let n_mels = mel.n_mels();

    let mel_data: Vec<f32> = mel.as_slice().to_vec();
    let input = Tensor::<B, 1>::from_floats(mel_data.as_slice(), &device)
        .reshape([1, 1, n_mels, n_frames]);

    let out = model.forward(input);
    let onset = sigmoid(out.onset).into_data().to_vec::<f32>().unwrap();
    let frame = sigmoid(out.frame).into_data().to_vec::<f32>().unwrap();
    let offset = sigmoid(out.offset).into_data().to_vec::<f32>().unwrap();

    let dec = DecodeConfig {
        sample_rate: cfg.audio.sample_rate,
        hop_length: cfg.audio.hop_length,
        midi_offset: cfg.model.midi_offset,
        onset_threshold: cfg.decode.onset_threshold,
        frame_threshold: cfg.decode.frame_threshold,
        offset_threshold: cfg.decode.offset_threshold,
        min_note_ms: cfg.decode.min_note_ms,
        merge_gap_ms: cfg.decode.merge_gap_ms,
    };

    let notes = decode_notes(
        &onset,
        &frame,
        Some(&offset),
        n_frames,
        cfg.model.n_pitches,
        &dec,
    );
    info!("Decoded {} notes from {}", notes.len(), audio.display());

    let payload = serde_json::json!({
        "model": "denote-crnn-v1",
        "sr": cfg.audio.sample_rate,
        "hop": cfg.audio.hop_length,
        "audio": audio.display().to_string(),
        "notes": notes,
    });
    fs::write(&args.out, serde_json::to_string_pretty(&payload)?)?;
    info!("Wrote {}", args.out.display());

    if let Some(ref midi_path) = args.midi {
        write_midi(&notes, midi_path)?;
        info!("Wrote {}", midi_path.display());
    }
    Ok(())
}

fn write_midi(notes: &[NoteEvent], path: &PathBuf) -> Result<()> {
    use midly::{
        num::{u15, u24, u28, u4, u7},
        Format, Header, MetaMessage, MidiMessage, Smf, Track, TrackEvent, TrackEventKind, Timing,
    };

    let ticks_per_beat = 480u16;
    let us_per_quarter = 500_000u32;
    let spb = us_per_quarter as f64 / 1_000_000.0;
    let sec_to_tick = |s: f64| -> u32 { ((s / spb) * ticks_per_beat as f64).round() as u32 };

    #[derive(Clone, Copy)]
    enum Ev {
        On { tick: u32, pitch: u8, vel: u8 },
        Off { tick: u32, pitch: u8 },
    }
    let mut events = Vec::new();
    for n in notes {
        events.push(Ev::On {
            tick: sec_to_tick(n.start_sec),
            pitch: n.pitch_midi,
            vel: n.velocity.max(1),
        });
        events.push(Ev::Off {
            tick: sec_to_tick(n.end_sec),
            pitch: n.pitch_midi,
        });
    }
    events.sort_by_key(|e| match *e {
        Ev::On { tick, .. } => (tick, 1u8),
        Ev::Off { tick, .. } => (tick, 0u8),
    });

    let mut track = Track::new();
    track.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::from(us_per_quarter))),
    });

    let mut last = 0u32;
    for e in events {
        let (tick, kind) = match e {
            Ev::On { tick, pitch, vel } => (
                tick,
                TrackEventKind::Midi {
                    channel: u4::from(0),
                    message: MidiMessage::NoteOn {
                        key: u7::from(pitch),
                        vel: u7::from(vel.min(127)),
                    },
                },
            ),
            Ev::Off { tick, pitch } => (
                tick,
                TrackEventKind::Midi {
                    channel: u4::from(0),
                    message: MidiMessage::NoteOff {
                        key: u7::from(pitch),
                        vel: u7::from(0),
                    },
                },
            ),
        };
        let delta = tick.saturating_sub(last);
        last = tick;
        track.push(TrackEvent {
            delta: u28::from(delta),
            kind,
        });
    }
    track.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    let smf = Smf {
        header: Header {
            format: Format::SingleTrack,
            timing: Timing::Metrical(u15::from(ticks_per_beat)),
        },
        tracks: vec![track],
    };
    let mut buf = Vec::new();
    smf.write(&mut buf).map_err(|e| anyhow::anyhow!("{e}"))?;
    fs::write(path, buf)?;
    Ok(())
}
