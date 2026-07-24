//! Job E — multi-instrument Note Onset F1 (Phase 3).
//!
//! ```text
//! cargo run -p denote-eval --release -- `
//!   --manifest data/cache/phase3/manifest.jsonl `
//!   --job-a runs/jobA_phase3/checkpoints/last `
//!   --decoders runs/jobB_g0_phase3/checkpoints/last/decoder_g0,runs/jobB_g1_phase3/checkpoints/last/decoder_g1 `
//!   --onset-tol 0.05 `
//!   --out runs/eval_phase3
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use burn::{
    backend::NdArray,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{activation::sigmoid, Int, Tensor},
};
use clap::Parser;
use log::info;
use serde::Serialize;

use denote_data::{load_notes_for_entry, CacheManifest, LatentFeatureCache, NoteList};
use denote_decode::{
    ar_decode_loop, detokenize_merge, group_active_from_presence, MergeConfig, NoteEvent,
    ProgramNoteEvent,
};
use denote_eval::{evaluate_notes_multi_onset, MultiInstrumentReport, RefNoteMulti};
use denote_model::{
    AdapterConfig, DenoteModular, DenoteModularConfig, GroupsConfig, PresenceHeadConfig,
    TokenDecoderConfig,
};
use denote_tokens::{decode_tokens, Vocab, VocabConfig};

type B = NdArray;

#[derive(Parser, Debug)]
#[command(name = "denote-eval", about = "Job E: multi-onset F1 vs MIROS protocol")]
struct Args {
    /// Manifest (val split used by default)
    #[arg(long)]
    manifest: PathBuf,

    /// Job A modular dir
    #[arg(long)]
    job_a: PathBuf,

    /// Comma-separated decoder stems
    #[arg(long)]
    decoders: String,

    #[arg(long, default_value = "configs/groups_challenge8.toml")]
    groups: PathBuf,

    #[arg(long, default_value_t = 0.05)]
    onset_tol: f64,

    #[arg(long, default_value = "runs/eval_phase3")]
    out: PathBuf,

    #[arg(long, default_value = "val")]
    split: String,

    #[arg(long, default_value_t = 0.5)]
    presence_threshold: f32,

    #[arg(long, default_value_t = false)]
    force_all: bool,

    /// Disable FSM constraints in AR (ablation A4)
    #[arg(long, default_value_t = false)]
    no_fsm: bool,

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

    /// Max val clips (0 = all)
    #[arg(long, default_value_t = 0)]
    max_clips: usize,
}

#[derive(Serialize)]
struct JobEReport {
    protocol: String,
    onset_tolerance_sec: f64,
    miros_multi_onset_f1_ref: f64,
    n_clips: usize,
    split: String,
    force_all: bool,
    no_fsm: bool,
    /// Macro F1 averaged over clips (each clip macro, then mean)
    mean_macro_f1: f64,
    mean_micro_f1: f64,
    /// Pooled multi-instrument report over all notes
    pooled: MultiInstrumentReport,
    gap_to_miros: f64,
    job_a: String,
    decoders: String,
    manifest: String,
}

#[derive(Serialize)]
struct PerFileLine {
    id: String,
    macro_f1: f64,
    micro_f1: f64,
    n_est: usize,
    n_ref: usize,
    n_instruments: usize,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    run_job_e(&args)
}

fn run_job_e(args: &Args) -> Result<()> {
    fs::create_dir_all(&args.out)?;
    let manifest = CacheManifest::from_path(&args.manifest);
    let cache_root = manifest.root.clone();
    let mut entries = manifest.list_by_split(&args.split)?;
    if entries.is_empty() {
        // fallback: all entries
        entries = manifest.load()?;
        info!(
            "split '{}' empty — using all {} entries",
            args.split,
            entries.len()
        );
    }
    if args.max_clips > 0 && entries.len() > args.max_clips {
        entries.truncate(args.max_clips);
    }
    if entries.is_empty() {
        bail!("no entries in {}", args.manifest.display());
    }

    let groups = if args.groups.exists() {
        GroupsConfig::load(&args.groups).map_err(|e| anyhow::anyhow!(e))?
    } else {
        GroupsConfig::challenge8()
    };

    let vocab = Vocab::new(VocabConfig::default());
    let vocab_size = vocab.size as usize;
    let k = args.presence_k;
    let n_groups = groups.n_groups().max(1);
    let device = Default::default();
    let d_model = args.d_model;

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
        .load_job_a(&args.job_a, &device)
        .map_err(|e| anyhow::anyhow!(e))?;

    let decoder_stems = parse_decoder_list(&args.decoders, n_groups)?;
    for (g, stem) in decoder_stems.iter().enumerate() {
        if let Some(s) = stem {
            if path_mpk_exists(s) {
                info!("Load decoder_g{g} from {}", s.display());
                model
                    .load_decoder_g(g, s, &device)
                    .map_err(|e| anyhow::anyhow!(e))?;
            }
        }
    }

    let latent_cache = LatentFeatureCache::new(cache_root.join("latents"));
    let merge_cfg = MergeConfig::default();

    let mut all_est: Vec<NoteEvent> = Vec::new();
    let mut all_ref: Vec<RefNoteMulti> = Vec::new();
    let mut per_file: Vec<PerFileLine> = Vec::new();
    let mut sum_macro = 0.0f64;
    let mut sum_micro = 0.0f64;

    for (i, e) in entries.iter().enumerate() {
        info!("[{}/{}] eval {}", i + 1, entries.len(), e.id);
        let sample = latent_cache
            .load(&e.id)
            .with_context(|| format!("latent {}", e.id))?;
        let t = sample.meta.n_frames;
        let d = sample.meta.d_model;
        let z = Tensor::<B, 1>::from_floats(sample.z.as_slice(), &device).reshape([1, t, d]);

        let est = decode_clip(
            &model,
            &z,
            &groups,
            &vocab,
            vocab_size,
            k,
            &decoder_stems,
            args.presence_threshold,
            args.force_all,
            args.max_decode_len,
            &merge_cfg,
            args.no_fsm,
        )?;

        let notes = load_notes_for_entry(e.notes_path.as_deref(), e.audio_path.as_deref(), &e.id)?;
        let refs = notes_to_ref(&notes);
        let report = evaluate_notes_multi_onset(&est, &refs, args.onset_tol);
        sum_macro += report.macro_f1;
        sum_micro += report.micro.f1;
        per_file.push(PerFileLine {
            id: e.id.clone(),
            macro_f1: report.macro_f1,
            micro_f1: report.micro.f1,
            n_est: est.len(),
            n_ref: refs.len(),
            n_instruments: e.n_instruments,
        });
        all_est.extend(est);
        all_ref.extend(refs);
    }

    let n = entries.len().max(1) as f64;
    let pooled = evaluate_notes_multi_onset(&all_est, &all_ref, args.onset_tol);
    let mean_macro = sum_macro / n;
    let mean_micro = sum_micro / n;
    let report = JobEReport {
        protocol: "multi_instrument_onset_f1_program_pitch_onset".into(),
        onset_tolerance_sec: args.onset_tol,
        miros_multi_onset_f1_ref: MultiInstrumentReport::MIROS_REF,
        n_clips: entries.len(),
        split: args.split.clone(),
        force_all: args.force_all,
        no_fsm: args.no_fsm,
        mean_macro_f1: mean_macro,
        mean_micro_f1: mean_micro,
        gap_to_miros: MultiInstrumentReport::MIROS_REF - pooled.macro_f1,
        pooled,
        job_a: args.job_a.display().to_string(),
        decoders: args.decoders.clone(),
        manifest: args.manifest.display().to_string(),
    };

    let report_path = args.out.join("report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;

    let per_path = args.out.join("per_file.jsonl");
    {
        let mut body = String::new();
        for line in &per_file {
            body.push_str(&serde_json::to_string(line)?);
            body.push('\n');
        }
        fs::write(&per_path, body)?;
    }

    // ckpt paths snapshot
    fs::write(
        args.out.join("ckpt_paths.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "job_a": args.job_a.display().to_string(),
            "decoders": args.decoders,
            "groups": args.groups.display().to_string(),
            "manifest": args.manifest.display().to_string(),
        }))?,
    )?;

    info!(
        "Job E done | mean_macro_f1={:.4} pooled_macro_f1={:.4} MIROS_ref={:.4} gap={:.4} → {}",
        mean_macro,
        report.pooled.macro_f1,
        MultiInstrumentReport::MIROS_REF,
        report.gap_to_miros,
        report_path.display()
    );
    let _ = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    Ok(())
}

fn notes_to_ref(notes: &NoteList) -> Vec<RefNoteMulti> {
    notes
        .notes
        .iter()
        .map(|n| RefNoteMulti {
            start_sec: n.start_sec,
            end_sec: n.end_sec,
            pitch_midi: n.pitch_midi,
            program: n.program,
        })
        .collect()
}

fn decode_clip(
    model: &DenoteModular<B>,
    z: &Tensor<B, 3>,
    groups: &GroupsConfig,
    vocab: &Vocab,
    vocab_size: usize,
    k: usize,
    decoder_stems: &[Option<PathBuf>],
    presence_threshold: f32,
    force_all: bool,
    max_decode_len: usize,
    merge_cfg: &MergeConfig,
    no_fsm: bool,
) -> Result<Vec<NoteEvent>> {
    let device = z.device();
    let multi_zero =
        Tensor::<B, 1>::from_floats(vec![0.0f32; k].as_slice(), &device).reshape([1, k]);
    let presence_logits = model.presence_from_latents(z.clone(), multi_zero);
    let presence_probs = sigmoid(presence_logits)
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();

    let mut group_notes: Vec<(u32, Vec<denote_tokens::TokenNote>)> = Vec::new();

    for gdef in &groups.groups {
        let g = gdef.id as usize;
        if g >= model.n_groups() {
            continue;
        }
        let stem_ok = decoder_stems
            .get(g)
            .and_then(|s| s.as_ref())
            .map(|s| path_mpk_exists(s.as_path()))
            .unwrap_or(false);
        if !stem_ok {
            continue;
        }
        let active =
            force_all || group_active_from_presence(&presence_probs, &gdef.programs, presence_threshold);
        if !active {
            continue;
        }

        let mut mh = vec![0.0f32; k];
        for &p in &gdef.programs {
            let i = p as usize;
            if i < k
                && (force_all
                    || presence_probs.get(i).copied().unwrap_or(0.0) > presence_threshold)
            {
                mh[i] = 1.0;
            }
        }
        if mh.iter().all(|&x| x == 0.0) {
            if let Some(p) = gdef.programs.first() {
                let i = *p as usize;
                if i < k {
                    mh[i] = 1.0;
                }
            }
        }
        let multi = Tensor::<B, 1>::from_floats(mh.as_slice(), &device).reshape([1, k]);
        let memory = model.adapter_memory(z.clone(), multi, true);
        let primary = if no_fsm {
            None
        } else {
            gdef.programs.first().copied()
        };
        let decoder = model.decoder(g);
        let tokens = ar_decode_loop(vocab, max_decode_len, primary, |prefix| {
            score_prefix_logits(decoder, &memory, prefix, vocab_size)
        });
        let notes = decode_tokens(&tokens, vocab);
        group_notes.push((gdef.id, notes));
    }

    let merged = detokenize_merge(&group_notes, merge_cfg);
    Ok(merged.iter().map(ProgramNoteEvent::to_note_event).collect())
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
    let logits = decoder.forward(tokens, memory.clone());
    let data = logits.into_data().to_vec::<f32>().unwrap_or_default();
    let v = vocab_size;
    if data.len() < v {
        return vec![0.0; v];
    }
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
        if p.is_dir() {
            for g in 0..n_groups {
                let stem = p.join(format!("decoder_g{g}"));
                if path_mpk_exists(&stem) {
                    out[g] = Some(stem);
                }
            }
            continue;
        }
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if let Some(rest) = name.strip_prefix("decoder_g") {
            if let Ok(g) = rest.parse::<usize>() {
                if g < n_groups {
                    out[g] = Some(p);
                }
            }
        } else {
            out[0] = Some(p);
        }
    }
    Ok(out)
}

fn path_mpk_exists(stem: &Path) -> bool {
    stem.with_extension("mpk").exists()
        || PathBuf::from(format!("{}.mpk", stem.display())).exists()
        || stem.exists()
}
