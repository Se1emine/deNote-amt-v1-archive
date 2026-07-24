//! Run layout + JSONL metrics logger (source of truth on disk).
//!
//! TensorBoard: offline via `tools/jsonl_to_tb.py` (no Rust TB crate required).

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use log::info;
use serde_json::{json, Map, Value};

/// Append-only scalar metrics logger for one experiment run directory.
pub struct RunLogger {
    #[allow(dead_code)]
    run_dir: PathBuf,
    writer: BufWriter<File>,
}

impl RunLogger {
    /// Open (or create) `run_dir/metrics.jsonl` for append.
    pub fn new(run_dir: &Path) -> Result<Self> {
        fs::create_dir_all(run_dir)
            .with_context(|| format!("create run_dir {}", run_dir.display()))?;
        let path = run_dir.join("metrics.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        Ok(Self {
            run_dir: run_dir.to_path_buf(),
            writer: BufWriter::new(file),
        })
    }

    /// Log one training/eval step. Keys in `kv` become JSON fields (e.g. loss, lr).
    pub fn log_scalars(&mut self, step: u64, split: &str, kv: &[(&str, f32)]) -> Result<()> {
        let mut map = Map::new();
        map.insert("ts".into(), Value::String(utc_rfc3339()));
        map.insert("step".into(), json!(step));
        map.insert("split".into(), Value::String(split.to_string()));
        for (k, v) in kv {
            map.insert((*k).to_string(), json!(v));
        }
        let line = Value::Object(map);
        writeln!(self.writer, "{line}").context("write metrics.jsonl")?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("flush metrics.jsonl")?;
        Ok(())
    }
}

/// Create standard Phase 0 run layout under `run_dir`.
pub fn prepare_run_dir(run_dir: &Path) -> Result<()> {
    fs::create_dir_all(run_dir.join("checkpoints"))
        .with_context(|| format!("create checkpoints {}", run_dir.display()))?;
    fs::create_dir_all(run_dir.join("tb"))
        .with_context(|| format!("create tb {}", run_dir.display()))?;
    Ok(())
}

/// Copy config next to the run for reproducibility.
pub fn copy_config(config_src: &Path, run_dir: &Path) -> Result<()> {
    let dest = run_dir.join("config.toml.copy");
    fs::copy(config_src, &dest)
        .with_context(|| format!("copy config {} → {}", config_src.display(), dest.display()))?;
    Ok(())
}

/// Write lightweight hparams snapshot.
pub fn write_hparams(run_dir: &Path, hparams: &Value) -> Result<()> {
    let path = run_dir.join("hparams.json");
    let text = serde_json::to_string_pretty(hparams).context("serialize hparams")?;
    fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Default run dir: `runs/{name}_{YYYYmmdd_HHMMSS}`.
pub fn default_run_dir(name: &str) -> PathBuf {
    let stamp = local_stamp();
    PathBuf::from("runs").join(format!("{name}_{stamp}"))
}

/// Train state persisted beside checkpoints (step counter; optim optional later).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainState {
    pub step: u64,
    pub lr: f32,
    pub backend: String,
    /// Optim state is not restored in Phase 0 — model weights + step only.
    pub optim_resume: bool,
}

impl TrainState {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read state {}", path.display()))?;
        let s: Self = serde_json::from_str(&text).context("parse state.json")?;
        Ok(s)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self).context("serialize state")?;
        // Atomic-ish: write temp then rename.
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &text).with_context(|| format!("write {}", tmp.display()))?;
        fs::rename(&tmp, path).with_context(|| format!("rename → {}", path.display()))?;
        Ok(())
    }
}

/// Infer run dir from `--resume path/to/checkpoints/last` → parent of `checkpoints`.
pub fn run_dir_from_resume(resume: &Path) -> Option<PathBuf> {
    // .../checkpoints/last  or  .../checkpoints/last.mpk
    let p = resume;
    let parent = p.parent()?; // checkpoints
    if parent.file_name()?.to_str()? == "checkpoints" {
        return parent.parent().map(|x| x.to_path_buf());
    }
    // resume points at run_dir/checkpoints
    if p.file_name()?.to_str()? == "checkpoints" {
        return p.parent().map(|x| x.to_path_buf());
    }
    None
}

pub fn log_run_banner(run_dir: &Path, backend: &str) {
    info!(
        "RunLogger | run_dir={} backend={} metrics=metrics.jsonl",
        run_dir.display(),
        backend
    );
}

fn utc_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = civil_utc(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn local_stamp() -> String {
    // Wall-clock stamp for folder names; UTC is fine and portable.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = civil_utc(secs);
    format!("{y:04}{mo:02}{d:02}_{h:02}{mi:02}{s:02}")
}

/// Days since Unix epoch → (year, month, day, hour, min, sec) UTC.
fn civil_utc(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    let tod = (secs % 86_400) as u32;
    let h = tod / 3600;
    let mi = (tod % 3600) / 60;
    let s = tod % 60;
    let z = (secs / 86_400) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = (y + if m <= 2 { 1 } else { 0 }) as i32;
    (y, m, d, h, mi, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn jsonl_roundtrip_fields() {
        let dir = std::env::temp_dir().join(format!(
            "denote_runlog_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&dir);
        prepare_run_dir(&dir).unwrap();
        let mut log = RunLogger::new(&dir).unwrap();
        log.log_scalars(1, "train", &[("loss", 1.5), ("lr", 3e-4), ("step_time_ms", 10.0)])
            .unwrap();
        log.flush().unwrap();
        let text = fs::read_to_string(dir.join("metrics.jsonl")).unwrap();
        assert!(text.contains("\"step\":1"));
        assert!(text.contains("\"loss\":1.5"));
        assert!(text.contains("\"step_time_ms\":10.0"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_dir_from_resume_path() {
        let p = PathBuf::from("runs/phase0_resume_test/checkpoints/last");
        assert_eq!(
            run_dir_from_resume(&p).as_deref(),
            Some(Path::new("runs/phase0_resume_test"))
        );
    }
}
