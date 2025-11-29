use num_complex::Complex32;
use once_cell::sync::Lazy;
use rustfft::{num_traits::Zero, Fft, FftPlanner};
use std::sync::Arc;

struct FftCache {
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    hann_window: Vec<f32>,
}

static FFT_CACHE_4096: Lazy<FftCache> = Lazy::new(|| {
    let mut planner = FftPlanner::new();
    FftCache {
        fft_forward: planner.plan_fft_forward(4096),
        fft_inverse: planner.plan_fft_inverse(4096),
        hann_window: compute_hann(4096),
    }
});

fn compute_hann(n_fft: usize) -> Vec<f32> {
    if n_fft <= 1 {
        return vec![1.0];
    }
    let denom = (n_fft - 1) as f32;
    (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (i as f32) / denom).cos())
        .collect()
}

pub fn to_planar_stereo(interleaved: &[f32], channels: u16) -> Vec<[f32; 2]> {
    if channels == 1 {
        interleaved.iter().map(|&x| [x, x]).collect()
    } else {
        let mut out = Vec::with_capacity(interleaved.len() / 2);
        let mut i = 0;
        while i + 1 < interleaved.len() {
            out.push([interleaved[i], interleaved[i + 1]]);
            i += 2;
        }
        out
    }
}

/// Compute complex-as-channels spectrogram for stereo with center padding.
/// Returns (buffer, F=2048, Frames=336) for T=343_980, n_fft=4096, hop=1024.
/// Layout is [1, 4, F, Frames] flattened => channels order: L.re, L.im, R.re, R.im.
pub fn stft_cac_stereo_centered(
    left: &[f32],
    right: &[f32],
    n_fft: usize,
    hop: usize,
) -> (Vec<f32>, usize, usize) {
    assert_eq!(left.len(), right.len());
    assert_eq!(n_fft, 4096, "Only n_fft=4096 is supported (cached)");
    
    let t = left.len();
    let pad = n_fft / 2;


    let padded_len = pad + t + pad;
    let mut l_sig = vec![0.0f32; padded_len];
    let mut r_sig = vec![0.0f32; padded_len];
    
    l_sig[pad..pad + t].copy_from_slice(left);
    r_sig[pad..pad + t].copy_from_slice(right);

    let frames = 1 + (t / hop);
    let f_bins = n_fft / 2;

    let cache = &*FFT_CACHE_4096;
    let fft = &cache.fft_forward;
    let window = &cache.hann_window;

    let mut out = vec![0.0f32; 4 * f_bins * frames];

    let mut buf_l = vec![Complex32::zero(); n_fft];
    let mut buf_r = vec![Complex32::zero(); n_fft];

    for fr in 0..frames {
        let start = fr * hop;
        let li = &l_sig[start..start + n_fft];
        let ri = &r_sig[start..start + n_fft];

        for i in 0..n_fft {
            let w = window[i];
            buf_l[i] = Complex32::new(li[i] * w, 0.0);
            buf_r[i] = Complex32::new(ri[i] * w, 0.0);
        }

        fft.process(&mut buf_l);
        fft.process(&mut buf_r);

        for fi in 0..f_bins {
            let base_fr = fi * frames + fr;
            out[0 * f_bins * frames + base_fr] = buf_l[fi].re;
            out[1 * f_bins * frames + base_fr] = buf_l[fi].im;
            out[2 * f_bins * frames + base_fr] = buf_r[fi].re;
            out[3 * f_bins * frames + base_fr] = buf_r[fi].im;
        }
    }

    (out, f_bins, frames)
}

/// Inverse STFT for complex-as-channels stereo spectrogram
/// Input: complex-as-channels [L.re, L.im, R.re, R.im] with shape [4, F, Frames]
/// Returns: (left, right) stereo waveform of length target_length
pub fn istft_cac_stereo(
    spec_cac: &[f32],
    f_bins: usize,
    frames: usize,
    n_fft: usize,
    hop: usize,
    target_length: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(n_fft, 4096, "Only n_fft=4096 is supported (cached)");
    
    // Use cached IFFT and window
    let cache = &*FFT_CACHE_4096;
    let ifft = &cache.fft_inverse;
    let window = &cache.hann_window;

    let pad = n_fft / 2;
    let padded_length = target_length + 2 * pad;

    // Output buffers
    let mut left_out = vec![0.0f32; padded_length];
    let mut right_out = vec![0.0f32; padded_length];
    let mut window_sum = vec![0.0f32; padded_length];

    // Scratch buffers
    let mut buf_l = vec![Complex32::zero(); n_fft];
    let mut buf_r = vec![Complex32::zero(); n_fft];
    
    let scale = 1.0 / (n_fft as f32);

    for fr in 0..frames {
        // Clear buffers
        buf_l.fill(Complex32::zero());
        buf_r.fill(Complex32::zero());

        // Fill positive frequencies [0..f_bins]
        for fi in 0..f_bins {
            let base_fr = fi * frames + fr;
            buf_l[fi] = Complex32::new(
                spec_cac[0 * f_bins * frames + base_fr],
                spec_cac[1 * f_bins * frames + base_fr],
            );
            buf_r[fi] = Complex32::new(
                spec_cac[2 * f_bins * frames + base_fr],
                spec_cac[3 * f_bins * frames + base_fr],
            );
        }

        // Fill negative frequencies (complex conjugate mirror)
        for fi in 1..f_bins {
            let neg_fi = n_fft - fi;
            buf_l[neg_fi] = buf_l[fi].conj();
            buf_r[neg_fi] = buf_r[fi].conj();
        }

        // Ensure DC and Nyquist are real
        buf_l[0].im = 0.0;
        buf_r[0].im = 0.0;
        buf_l[n_fft / 2].im = 0.0;
        buf_r[n_fft / 2].im = 0.0;

        // Apply IFFT
        ifft.process(&mut buf_l);
        ifft.process(&mut buf_r);

        // Overlap-add with window
        let start = fr * hop;
        for i in 0..n_fft {
            let pos = start + i;
            if pos < padded_length {
                let w = window[i];
                left_out[pos] += buf_l[i].re * w * scale;
                right_out[pos] += buf_r[i].re * w * scale;
                window_sum[pos] += w * w;
            }
        }
    }

    // Normalize by window sum
    for i in 0..padded_length {
        let sum = window_sum[i];
        if sum > 1e-10 {
            left_out[i] /= sum;
            right_out[i] /= sum;
        }
    }

    // Remove padding
    let start = pad.min(left_out.len());
    let end = (pad + target_length).min(left_out.len());

    let left_final = if end > start {
        left_out[start..end].to_vec()
    } else {
        vec![0.0; target_length]
    };

    let right_final = if end > start {
        right_out[start..end].to_vec()
    } else {
        vec![0.0; target_length]
    };

    (left_final, right_final)
}

/// Parallel iSTFT for multiple sources - processes all 4 stems in parallel
pub fn istft_cac_stereo_parallel(
    sources_data: &[&[f32]],  // Slice of 4 source spectrograms
    f_bins: usize,
    frames: usize,
    n_fft: usize,
    hop: usize,
    target_length: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    use rayon::prelude::*;
    
    sources_data
        .par_iter()
        .map(|spec_cac| {
            istft_cac_stereo(spec_cac, f_bins, frames, n_fft, hop, target_length)
        })
        .collect()
}
