use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use hound::WavReader;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::resample::resample_linear;

/// Load only the first `max_sec` seconds of a mono f32 WAV (avoids multi-GB loads).
pub fn load_wav_mono_f32_max_sec(
    path: impl AsRef<Path>,
    max_sec: f64,
) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let mut reader = WavReader::open(path)
        .with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;
    let max_frames = ((max_sec.max(0.1) * sr as f64).ceil() as usize).max(1);
    let max_interleaved = max_frames.saturating_mul(channels.max(1));

    let mut interleaved: Vec<f32> = Vec::with_capacity(max_interleaved.min(1 << 22));
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for (i, s) in reader.samples::<f32>().enumerate() {
                if i >= max_interleaved {
                    break;
                }
                interleaved.push(s.map_err(|e| anyhow!(e))?);
            }
        }
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            for (i, s) in reader.samples::<i32>().enumerate() {
                if i >= max_interleaved {
                    break;
                }
                let v = s.map_err(|e| anyhow!(e))? as f32 / max;
                interleaved.push(v);
            }
        }
    }
    Ok((downmix_mono(&interleaved, channels), sr))
}

/// Load a mono f32 WAV (fast path via hound). Returns (samples, sample_rate).
pub fn load_wav_mono_f32(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let mut reader = WavReader::open(path)
        .with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| anyhow!(e)))
            .collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max).map_err(|e| anyhow!(e)))
                .collect()
        }
    };
    let interleaved = samples?;
    Ok((downmix_mono(&interleaved, channels), sr))
}

/// Load audio (wav/flac/mp3/ogg) as mono f32 via Symphonia.
pub fn load_mono_f32(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
    {
        return load_wav_mono_f32(path);
    }

    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("probe {}", path.display()))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no audio track in {}", path.display()))?
        .clone();

    let sr = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("missing sample rate in {}", path.display()))?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1) as usize;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("create decoder")?;

    let mut interleaved = Vec::new();
    let track_id = track.id;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::ResetRequired) => continue,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(anyhow!("packet read: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => append_buffer(&mut interleaved, &buf),
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow!("decode: {e}")),
        }
    }

    Ok((downmix_mono(&interleaved, channels), sr))
}

/// Load mono and resample to target sample rate if needed.
pub fn load_mono_resampled(path: impl AsRef<Path>, target_sr: u32) -> Result<Vec<f32>> {
    let (samples, sr) = load_mono_f32(path)?;
    if sr == target_sr {
        return Ok(samples);
    }
    Ok(resample_linear(&samples, sr, target_sr))
}

fn downmix_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let n = interleaved.len() / channels;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = 0.0f32;
        for c in 0..channels {
            s += interleaved[i * channels + c];
        }
        out.push(s / channels as f32);
    }
    out
}

fn append_buffer(dst: &mut Vec<f32>, buf: &AudioBufferRef<'_>) {
    match buf {
        AudioBufferRef::F32(b) => {
            let ch = b.spec().channels.count();
            let frames = b.frames();
            for i in 0..frames {
                for c in 0..ch {
                    dst.push(b.chan(c)[i]);
                }
            }
        }
        AudioBufferRef::U8(b) => convert_int_buf(dst, b, u8::MAX as f32, true),
        AudioBufferRef::U16(b) => convert_int_buf(dst, b, u16::MAX as f32, true),
        AudioBufferRef::U24(b) => {
            let ch = b.spec().channels.count();
            let frames = b.frames();
            let scale = 1.0 / ((1u32 << 23) as f32);
            for i in 0..frames {
                for c in 0..ch {
                    let v = b.chan(c)[i].inner() as i32;
                    dst.push((v as f32) * scale);
                }
            }
        }
        AudioBufferRef::U32(b) => convert_int_buf(dst, b, u32::MAX as f32, true),
        AudioBufferRef::S8(b) => convert_int_buf(dst, b, i8::MAX as f32, false),
        AudioBufferRef::S16(b) => convert_int_buf(dst, b, i16::MAX as f32, false),
        AudioBufferRef::S24(b) => {
            let ch = b.spec().channels.count();
            let frames = b.frames();
            let scale = 1.0 / ((1u32 << 23) as f32);
            for i in 0..frames {
                for c in 0..ch {
                    let v = b.chan(c)[i].inner();
                    dst.push((v as f32) * scale);
                }
            }
        }
        AudioBufferRef::S32(b) => convert_int_buf(dst, b, i32::MAX as f32, false),
        AudioBufferRef::F64(b) => {
            let ch = b.spec().channels.count();
            let frames = b.frames();
            for i in 0..frames {
                for c in 0..ch {
                    dst.push(b.chan(c)[i] as f32);
                }
            }
        }
    }
}

fn convert_int_buf<T>(
    dst: &mut Vec<f32>,
    b: &symphonia::core::audio::AudioBuffer<T>,
    max: f32,
    unsigned: bool,
) where
    T: Copy + Into<f64> + symphonia::core::sample::Sample,
{
    let ch = b.spec().channels.count();
    let frames = b.frames();
    let offset = if unsigned { max * 0.5 } else { 0.0 };
    let scale = if unsigned { max * 0.5 } else { max };
    for i in 0..frames {
        for c in 0..ch {
            let v: f64 = b.chan(c)[i].into();
            dst.push(((v as f32) - offset) / scale);
        }
    }
}
