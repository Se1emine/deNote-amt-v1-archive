//! Training binary for Denote-MIROS (seq2seq) and legacy CRNN.
//!
//! ```text
//! cargo run -p denote-train --release -- --config configs/miros_v1_smoke.toml --smoke
//! cargo run -p denote-train --release -- --config configs/phase0_mini.toml --mini-steps 100
//! cargo run -p denote-train --release -- --config configs/free_1070_cache.toml --job cache-l0
//! cargo run -p denote-train --release -- --config configs/free_1070_jobA.toml --job job-a
//! cargo run -p denote-train --release -- --config configs/free_1070_jobB_g0.toml --job job-b
//! cargo run -p denote-train --release -- --config configs/v1_synth.toml --smoke
//! ```

mod config;
mod data;
mod device;
mod jobs;
mod metrics_log;
mod mini_train;
mod miros_smoke;
mod phase1_config;
mod phase2_config;
mod phase3_config;
mod train_loop;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use log::info;

use crate::config::AppConfig;
use crate::device::{log_backend, resolve_device};
use crate::jobs::{
    run_cache_l0, run_job_a, run_job_b, run_phase3_data, CacheL0Opts, JobAOpts, JobBOpts,
    Phase3DataOpts,
};
use crate::mini_train::{run_mini_train, MiniTrainOpts};

#[derive(Parser, Debug)]
#[command(
    name = "denote-train",
    about = "Train Denote-MIROS (primary) or legacy CRNN"
)]
struct Args {
    /// Path to TOML config (`configs/phase0_mini.toml`, `miros_v1_smoke.toml`, or legacy `v1_*.toml`)
    #[arg(long, default_value = "configs/miros_v1_smoke.toml")]
    config: PathBuf,

    /// Job: `cache-l0` | `job-a` | `job-b` | `phase3-data` (Phase 1–3)
    #[arg(long)]
    job: Option<String>,

    /// Only run a synthetic smoke step (no real dataset required)
    #[arg(long)]
    smoke: bool,

    /// Phase 0 mini-train: N teacher-forcing CE steps on synthetic data
    #[arg(long)]
    mini_steps: Option<u64>,

    /// Experiment run directory (default: runs/phase0_mini_YYYYmmdd_HHMMSS)
    #[arg(long)]
    run_dir: Option<PathBuf>,

    /// Resume from checkpoint stem (e.g. runs/.../checkpoints/last)
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Job L0: override manifest output path
    #[arg(long)]
    manifest_out: Option<PathBuf>,

    /// Log scalars every N steps (omit to use config)
    #[arg(long)]
    log_every: Option<u64>,

    /// Save checkpoint every N steps (omit to use config; always saves `last` at end)
    #[arg(long)]
    ckpt_every: Option<u64>,

    /// Override number of epochs (legacy CRNN)
    #[arg(long)]
    epochs: Option<usize>,

    /// Override batch size (legacy CRNN / Job A)
    #[arg(long)]
    batch_size: Option<usize>,

    /// Override max steps (Job A / mini)
    #[arg(long)]
    max_steps: Option<u64>,

    /// Device: auto | cpu | cuda | wgpu
    #[arg(long, default_value = "auto")]
    device: String,

    /// Job B: override group id (0..G-1)
    #[arg(long)]
    group: Option<u32>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    // Phase 1 jobs (cache L0 / Job A)
    if let Some(ref job) = args.job {
        match job.as_str() {
            "cache-l0" | "l0" | "cache" => {
                info!(
                    "Phase1 Job L0 | config={} device={}",
                    args.config.display(),
                    args.device
                );
                run_cache_l0(CacheL0Opts {
                    config: args.config,
                    manifest_out: args.manifest_out,
                    device: args.device,
                    input_dir: None,
                    cache_dir: None,
                })?;
                return Ok(());
            }
            "job-a" | "jobA" | "a" => {
                info!(
                    "Phase1 Job A | config={} device={}",
                    args.config.display(),
                    args.device
                );
                run_job_a(JobAOpts {
                    config: args.config,
                    run_dir: args.run_dir,
                    resume: args.resume,
                    device: args.device,
                    max_steps: args.max_steps,
                    log_every: args.log_every.map(|n| n.max(1)),
                    ckpt_every: args.ckpt_every,
                    batch_size: args.batch_size,
                })?;
                return Ok(());
            }
            "job-b" | "jobB" | "b" => {
                info!(
                    "Phase2 Job B | config={} device={} group={:?}",
                    args.config.display(),
                    args.device,
                    args.group
                );
                run_job_b(JobBOpts {
                    config: args.config,
                    run_dir: args.run_dir,
                    resume: args.resume,
                    device: args.device,
                    max_steps: args.max_steps,
                    log_every: args.log_every.map(|n| n.max(1)),
                    ckpt_every: args.ckpt_every,
                    batch_size: args.batch_size,
                    group_id: args.group,
                })?;
                return Ok(());
            }
            "phase3-data" | "phase3_data" | "corpus" => {
                info!(
                    "Phase3 corpus | config={} (CPU FluidR3 + stem-mix)",
                    args.config.display()
                );
                run_phase3_data(Phase3DataOpts {
                    config: args.config,
                })?;
                return Ok(());
            }
            other => {
                anyhow::bail!(
                    "unknown --job {other:?}; expected cache-l0 | job-a | job-b | phase3-data"
                );
            }
        }
    }

    // Phase 0 mini-train (synthetic multi-step job)
    if let Some(n) = args.mini_steps {
        if n == 0 {
            anyhow::bail!("--mini-steps must be >= 1");
        }
        let lr = load_lr_hint(&args.config).unwrap_or(3e-4);
        info!(
            "Denote-MIROS mini-train | config={} mini_steps={} device={}",
            args.config.display(),
            n,
            args.device
        );
        run_mini_train(MiniTrainOpts {
            config: args.config,
            mini_steps: n,
            run_dir: args.run_dir,
            resume: args.resume,
            device: args.device,
            log_every: args.log_every.unwrap_or(1).max(1),
            ckpt_every: args.ckpt_every.unwrap_or(50),
            lr,
        })?;
        return Ok(());
    }

    // Primary path: Denote-MIROS seq2seq smoke (config name contains "miros")
    if args.smoke && miros_smoke::is_miros_config(&args.config) {
        let kind = resolve_device(&args.device)?;
        log_backend(kind, "Denote-MIROS smoke");
        info!("Denote-MIROS smoke | config={}", args.config.display());
        miros_smoke::run_miros_smoke(kind)?;
        info!("MIROS Smoke OK");
        return Ok(());
    }

    let mut cfg = AppConfig::load(&args.config)
        .with_context(|| format!("load config {}", args.config.display()))?;
    if let Some(e) = args.epochs {
        cfg.train.epochs = e;
    }
    if let Some(b) = args.batch_size {
        cfg.train.batch_size = b;
    }

    info!(
        "Denote-CRNN (legacy) train | chunk={}s batch={} epochs={} device={}",
        cfg.train.chunk_sec, cfg.train.batch_size, cfg.train.epochs, args.device
    );

    if args.smoke {
        info!("Running legacy CRNN smoke (synthetic tensors, CPU/ndarray)");
        train_loop::run_smoke(&cfg)?;
        info!("Smoke OK");
        return Ok(());
    }

    train_loop::run_training(&cfg, &args.device)?;
    Ok(())
}

/// Read `[train].lr` if present (MIROS/phase0 TOML); default 3e-4.
fn load_lr_hint(path: &std::path::Path) -> Option<f64> {
    let text = std::fs::read_to_string(path).ok()?;
    let val: toml::Value = toml::from_str(&text).ok()?;
    val.get("train")?.get("lr")?.as_float()
}
