//! WAV writers (PCM int16 mono).

use std::path::Path;

use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};

/// Write mono f32 samples as 16-bit PCM WAV.
pub fn write_wav_mono_i16(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer =
        WavWriter::create(path, spec).with_context(|| format!("create wav {}", path.display()))?;
    for &x in samples {
        let v = (x.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Peak-normalize to `peak` (e.g. 0.9) then write mono int16 WAV.
pub fn write_wav_mono_normalized(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
    peak: f32,
) -> Result<()> {
    let max = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max > 1e-8 { peak / max } else { 1.0 };
    let scaled: Vec<f32> = samples.iter().map(|x| x * scale).collect();
    write_wav_mono_i16(path, &scaled, sample_rate)
}
