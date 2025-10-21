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

pub fn hann2(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let w = (std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).sin();
            w * w
        })
        .collect()
}

/// Classic Hann (not squared), length n_fft.
pub fn hann(n_fft: usize) -> Vec<f32> {
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
