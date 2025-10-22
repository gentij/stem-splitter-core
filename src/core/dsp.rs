use num_complex::Complex32;
use rustfft::{num_traits::Zero, FftPlanner};

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

/// Hann window function
fn hann(n_fft: usize) -> Vec<f32> {
    if n_fft <= 1 {
        return vec![1.0];
    }
    let denom = (n_fft - 1) as f32;
    (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (i as f32) / denom).cos())
        .collect()
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
    let t = left.len();
    // Demucs export expects center=True: pad n_fft/2 both sides
    let pad = n_fft / 2;
    let lpad = vec![0.0f32; pad];
    let rpad = vec![0.0f32; pad];

    let mut l_sig = Vec::with_capacity(pad + t + pad);
    let mut r_sig = Vec::with_capacity(pad + t + pad);
    l_sig.extend_from_slice(&lpad);
    l_sig.extend_from_slice(left);
    l_sig.extend_from_slice(&lpad);
    r_sig.extend_from_slice(&rpad);
    r_sig.extend_from_slice(right);
    r_sig.extend_from_slice(&rpad);

    // Frames = 1 + floor(T / hop) when center=True and T divisible-ish
    let frames = 1 + (t / hop);
    let window = hann(n_fft);

    // FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    // We keep only F = n_fft/2 bins (drop Nyquist so F=2048 for 4096)
    let f_bins = n_fft / 2;

    // Layout target: [1, 4, F, Frames]
    let mut out = vec![0.0f32; 4 * f_bins * frames];

    // Scratch buffers
    let mut buf_l = vec![Complex32::zero(); n_fft];
    let mut buf_r = vec![Complex32::zero(); n_fft];

    for fr in 0..frames {
        let start = fr * hop;
        // slice from padded signals
        let li = &l_sig[start..start + n_fft];
        let ri = &r_sig[start..start + n_fft];

        // window + pack into complex
        for i in 0..n_fft {
            let w = window[i];
            buf_l[i].re = li[i] * w;
            buf_l[i].im = 0.0;
            buf_r[i].re = ri[i] * w;
            buf_r[i].im = 0.0;
        }

        fft.process(&mut buf_l);
        fft.process(&mut buf_r);

        // write channels [L.re, L.im, R.re, R.im] over [F,Frames]
        for fi in 0..f_bins {
            let base_fr = fi * frames + fr; // [F,Frames] index

            // L.re
            out[0 * f_bins * frames + base_fr] = buf_l[fi].re;
            // L.im
            out[1 * f_bins * frames + base_fr] = buf_l[fi].im;
            // R.re
            out[2 * f_bins * frames + base_fr] = buf_r[fi].re;
            // R.im
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
    let window = hann(n_fft);
    
    // Prepare IFFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    
    // Padded length (matching forward STFT)
    let pad = n_fft / 2;
    let padded_length = target_length + 2 * pad;
    
    // Output buffers (padded)
    let mut left_out = vec![0.0f32; padded_length];
    let mut right_out = vec![0.0f32; padded_length];
    let mut window_sum = vec![0.0f32; padded_length];
    
    // Scratch buffers for IFFT
    let mut buf_l = vec![Complex32::zero(); n_fft];
    let mut buf_r = vec![Complex32::zero(); n_fft];
    
    for fr in 0..frames {
        // Clear buffers
        buf_l.fill(Complex32::zero());
        buf_r.fill(Complex32::zero());
        
        // Reconstruct full spectrum from half spectrum
        // Fill positive frequencies [0..f_bins]
        for fi in 0..f_bins {
            let base_fr = fi * frames + fr;
            buf_l[fi] = Complex32::new(
                spec_cac[0 * f_bins * frames + base_fr],  // L.re
                spec_cac[1 * f_bins * frames + base_fr],  // L.im
            );
            buf_r[fi] = Complex32::new(
                spec_cac[2 * f_bins * frames + base_fr],  // R.re
                spec_cac[3 * f_bins * frames + base_fr],  // R.im
            );
        }
        
        // Fill negative frequencies (complex conjugate mirror)
        // Skip DC (fi=0) and only mirror [1..f_bins-1]
        for fi in 1..f_bins {
            let neg_fi = n_fft - fi;
            buf_l[neg_fi] = buf_l[fi].conj();
            buf_r[neg_fi] = buf_r[fi].conj();
        }
        
        // Ensure DC and Nyquist are real
        buf_l[0].im = 0.0;
        buf_r[0].im = 0.0;
        if n_fft % 2 == 0 && f_bins < n_fft {
            buf_l[n_fft / 2].im = 0.0;
            buf_r[n_fft / 2].im = 0.0;
        }
        
        // Apply IFFT
        ifft.process(&mut buf_l);
        ifft.process(&mut buf_r);
        
        // Overlap-add with window (no extra scaling - already in IFFT)
        let start = fr * hop;
        for i in 0..n_fft {
            let pos = start + i;
            if pos < padded_length {
                let w = window[i];
                // IFFT returns normalized values, apply window for overlap-add
                left_out[pos] += buf_l[i].re * w / (n_fft as f32);
                right_out[pos] += buf_r[i].re * w / (n_fft as f32);
                window_sum[pos] += w * w;
            }
        }
    }
    
    // Normalize by window sum to account for overlap
    for i in 0..padded_length {
        let sum = window_sum[i];
        if sum > 1e-10 {
            left_out[i] /= sum;
            right_out[i] /= sum;
        }
    }
    
    // Remove padding and ensure we don't go out of bounds
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
