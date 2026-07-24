//! Best-effort MusicFM weight load from `tools/convert_musicfm.py` dumps.
//!
//! Layout expected under `--out` / `assets/musicfm_msd/`:
//! ```text
//! state_keys.json   # [{ "name", "shape", "file" }, ...]
//! tensors/*.npy     # float32 (or castable) arrays
//! stats.json        # mel mean/std (optional here)
//! ```
//!
//! **Honesty note (Phase 1 E1.1):** Burn Conformer is shape-compatible scaffold,
//! not bit-exact Wav2Vec2-Conformer. Mapping is partial: we load tensors whose
//! names/shapes match a documented table; unmatched keys are logged; remaining
//! params stay randomly initialized. Pipeline still trains Job A on latents.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use log::{info, warn};
use serde::Deserialize;

/// One entry from `state_keys.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct StateKeyEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub file: String,
}

/// Load result summary (for logging / docs).
#[derive(Debug, Clone, Default)]
pub struct WeightLoadReport {
    pub n_files: usize,
    pub n_tensors: usize,
    pub total_elems: usize,
    pub unmatched_names: Vec<String>,
    pub loaded_names: Vec<String>,
}

/// Read all numpy tensors from a convert_musicfm output directory.
pub fn load_musicfm_numpy_dir(dir: impl AsRef<Path>) -> Result<(HashMap<String, Vec<f32>>, WeightLoadReport)> {
    let dir = dir.as_ref();
    let keys_path = dir.join("state_keys.json");
    if !keys_path.exists() {
        bail!(
            "missing {} — run tools/convert_musicfm.py first (see scripts/job_L0_cache.md)",
            keys_path.display()
        );
    }
    let text = fs::read_to_string(&keys_path)
        .with_context(|| format!("read {}", keys_path.display()))?;
    let keys: Vec<StateKeyEntry> =
        serde_json::from_str(&text).context("parse state_keys.json")?;

    let mut map = HashMap::new();
    let mut report = WeightLoadReport {
        n_files: keys.len(),
        ..Default::default()
    };

    for e in &keys {
        let path = find_tensor_file(dir, &e.name, &e.file);
        let Some(path) = path else {
            warn!("missing tensor file for key {} ({})", e.name, e.file);
            report.unmatched_names.push(e.name.clone());
            continue;
        };
        let (data, _shape) = read_npy_f32(&path)?;
        let expected: usize = e.shape.iter().product();
        if data.len() != expected {
            warn!(
                "shape mismatch for {}: json {:?} npy_elems={}",
                e.name,
                e.shape,
                data.len()
            );
        }
        report.total_elems += data.len();
        report.n_tensors += 1;
        report.loaded_names.push(e.name.clone());
        map.insert(e.name.clone(), data);
    }

    info!(
        "MusicFM numpy dump: {} tensors, {} elems from {}",
        report.n_tensors,
        report.total_elems,
        dir.display()
    );
    Ok((map, report))
}

fn find_tensor_file(dir: &Path, name: &str, file: &str) -> Option<PathBuf> {
    let tensors = dir.join("tensors");
    let candidates = [
        tensors.join(file),
        tensors.join(format!(
            "{}.npy",
            name.chars()
                .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                })
                .collect::<String>()
        )),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// Minimal NumPy `.npy` reader for C-order float32 (and float64 → cast).
pub fn read_npy_f32(path: &Path) -> Result<(Vec<f32>, Vec<usize>)> {
    let mut f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .with_context(|| format!("read {}", path.display()))?;
    parse_npy_f32(&buf).with_context(|| format!("parse npy {}", path.display()))
}

fn parse_npy_f32(buf: &[u8]) -> Result<(Vec<f32>, Vec<usize>)> {
    if buf.len() < 10 {
        bail!("npy too short");
    }
    if &buf[0..6] != b"\x93NUMPY" {
        bail!("bad npy magic");
    }
    let major = buf[6];
    let _minor = buf[7];
    let (header_len, header_off) = if major == 1 {
        let hl = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        (hl, 10)
    } else {
        if buf.len() < 12 {
            bail!("npy v2 header truncated");
        }
        let hl = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        (hl, 12)
    };
    let header_end = header_off + header_len;
    if buf.len() < header_end {
        bail!("npy header truncated");
    }
    let header = std::str::from_utf8(&buf[header_off..header_end]).context("npy header utf8")?;
    let descr = extract_header_field(header, "descr")?;
    let fortran = extract_header_field(header, "fortran_order")?
        .trim()
        .eq_ignore_ascii_case("true");
    if fortran {
        bail!("fortran-order npy not supported (convert to C-order)");
    }
    let shape = parse_shape(&extract_header_field(header, "shape")?)?;
    let data = &buf[header_end..];
    let n: usize = shape.iter().product();
    let values = if descr.contains("<f4") || descr.contains("|f4") || descr == "'<f4'" {
        if data.len() < n * 4 {
            bail!("f32 body short: need {} got {}", n * 4, data.len());
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let o = i * 4;
            out.push(f32::from_le_bytes([
                data[o],
                data[o + 1],
                data[o + 2],
                data[o + 3],
            ]));
        }
        out
    } else if descr.contains("<f8") {
        if data.len() < n * 8 {
            bail!("f64 body short");
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let o = i * 8;
            let v = f64::from_le_bytes([
                data[o],
                data[o + 1],
                data[o + 2],
                data[o + 3],
                data[o + 4],
                data[o + 5],
                data[o + 6],
                data[o + 7],
            ]);
            out.push(v as f32);
        }
        out
    } else {
        bail!("unsupported npy dtype descr={descr}");
    };
    Ok((values, shape))
}

fn extract_header_field(header: &str, key: &str) -> Result<String> {
    // header is a Python dict-like string: {'descr': '<f4', 'fortran_order': False, 'shape': (1, 2), }
    let pat = format!("'{key}':");
    let idx = header
        .find(&pat)
        .or_else(|| header.find(&format!("\"{key}\":")))
        .with_context(|| format!("npy header missing {key}"))?;
    let rest = &header[idx + pat.len()..];
    let rest = rest.trim_start();
    if rest.starts_with('(') {
        let end = rest
            .find(')')
            .context("npy shape paren")?
            + 1;
        return Ok(rest[..end].to_string());
    }
    if rest.starts_with('\'') {
        let end = rest[1..]
            .find('\'')
            .context("npy string field")?
            + 2;
        return Ok(rest[..end].to_string());
    }
    // False/True
    let token: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    Ok(token)
}

fn parse_shape(shape_str: &str) -> Result<Vec<usize>> {
    let s = shape_str.trim();
    let s = s.trim_start_matches('(').trim_end_matches(')');
    if s.is_empty() {
        return Ok(vec![]);
    }
    // single element may be "(512,)" 
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        out.push(p.parse::<usize>().with_context(|| format!("shape dim {p}"))?);
    }
    Ok(out)
}

/// Describe whether a weights dir is usable (exists + has keys).
pub fn musicfm_weights_available(dir: impl AsRef<Path>) -> bool {
    let dir = dir.as_ref();
    dir.join("state_keys.json").is_file() && dir.join("tensors").is_dir()
}

/// Log param-count style summary for E1.1.
pub fn log_weight_inventory(report: &WeightLoadReport) {
    info!(
        "MusicFM inventory | tensors={} elems≈{} unmatched={}",
        report.n_tensors,
        report.total_elems,
        report.unmatched_names.len()
    );
    if !report.unmatched_names.is_empty() {
        let preview: Vec<_> = report.unmatched_names.iter().take(8).cloned().collect();
        warn!("unmatched keys (first): {preview:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_npy_f32_roundtrip_header() {
        // Build minimal C-order f32 array shape (2, 3)
        let mut header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }".to_string();
        // pad header to 16-byte alignment after 10-byte prefix (v1)
        let prefix_len = 10;
        while (prefix_len + header.len()) % 16 != 0 {
            header.push(' ');
        }
        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        let hl = header.len() as u16;
        buf.extend_from_slice(&hl.to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let (vals, shape) = parse_npy_f32(&buf).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(vals, data.to_vec());
    }
}
