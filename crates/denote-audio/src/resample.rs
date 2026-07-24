/// Simple linear resampler (good enough for training features; use rubato for product path later).
pub fn resample_linear(input: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || input.is_empty() {
        return input.to_vec();
    }
    if src_sr == 2 * dst_sr {
        let out_len = input.len() / 2;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            out.push((input[i * 2] + input[i * 2 + 1]) * 0.5);
        }
        return out;
    }

    let ratio = src_sr as f64 / dst_sr as f64;
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len() - 1);
        let frac = (src_idx - idx0 as f64) as f32;
        out.push(input[idx0] * (1.0 - frac) + input[idx1] * frac);
    }
    out
}

/// High-quality resample via rubato (sinc). Prefer for inference / final metrics.
pub fn resample_sinc(input: &[f32], src_sr: u32, dst_sr: u32) -> Result<Vec<f32>, String> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    if src_sr == dst_sr || input.is_empty() {
        return Ok(input.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = dst_sr as f64 / src_sr as f64;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, input.len().min(4096), 1)
        .map_err(|e| format!("rubato init: {e}"))?;

    let mut out = Vec::with_capacity(((input.len() as f64) * ratio) as usize + 16);
    let chunk = resampler.input_frames_next();
    let mut pos = 0usize;
    while pos + chunk <= input.len() {
        let waves = vec![&input[pos..pos + chunk]];
        let result = resampler
            .process(&waves, None)
            .map_err(|e| format!("rubato process: {e}"))?;
        out.extend_from_slice(&result[0]);
        pos += chunk;
    }
    // flush remaining with zeros pad
    if pos < input.len() {
        let mut pad = vec![0.0f32; chunk];
        let rem = input.len() - pos;
        pad[..rem].copy_from_slice(&input[pos..]);
        let waves = vec![&pad[..]];
        if let Ok(result) = resampler.process(&waves, None) {
            // approximate keep ratio for remainder
            let keep = ((rem as f64) * ratio).ceil() as usize;
            let take = keep.min(result[0].len());
            out.extend_from_slice(&result[0][..take]);
        }
    }
    Ok(out)
}
