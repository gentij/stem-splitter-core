use approx::assert_abs_diff_eq;
use stem_splitter_core::core::dsp::{istft_cac_stereo, stft_cac_stereo_centered, to_planar_stereo};

#[test]
fn to_planar_stereo_mono_duplicates_channel() {
    let mono = vec![0.1, -0.2, 0.3, -0.4];
    let planar = to_planar_stereo(&mono, 1);
    assert_eq!(planar.len(), mono.len());
    for i in 0..mono.len() {
        assert_abs_diff_eq!(planar[i][0], mono[i], epsilon = 1e-7);
        assert_abs_diff_eq!(planar[i][1], mono[i], epsilon = 1e-7);
    }
}

#[test]
fn to_planar_stereo_interleaved_ok() {
    let stereo_inter = vec![0.1, 0.2, -0.3, -0.4, 1.0, 0.5, 0.0, -1.0];
    let planar = to_planar_stereo(&stereo_inter, 2);
    assert_eq!(planar.len(), stereo_inter.len() / 2);
    for (i, frame) in planar.iter().enumerate() {
        assert_abs_diff_eq!(frame[0], stereo_inter[2 * i], epsilon = 1e-7);
        assert_abs_diff_eq!(frame[1], stereo_inter[2 * i + 1], epsilon = 1e-7);
    }
}

#[test]
fn stft_istft_roundtrip() {
    use approx::assert_abs_diff_eq;

    let n_fft = 1024usize;
    let hop = 256usize;
    let t = 4096usize;

    let mut left = vec![0.0f32; t];
    let mut right = vec![0.0f32; t];
    left[100] = 1.0;
    right[200] = -1.0;
    for i in 0..t {
        left[i] += (i as f32 * 0.01).cos() * 0.1;
        right[i] += (i as f32 * 0.02).sin() * 0.1;
    }

    let (spec, f_bins, frames) = stft_cac_stereo_centered(&left, &right, n_fft, hop);
    let (l2, r2) = istft_cac_stereo(&spec, f_bins, frames, n_fft, hop, t);
    assert_eq!(l2.len(), t);
    assert_eq!(r2.len(), t);

    let margin = n_fft;
    for i in margin..(t - margin) {
        assert_abs_diff_eq!(l2[i], left[i], epsilon = 1e-3);
        assert_abs_diff_eq!(r2[i], right[i], epsilon = 1e-3);
    }
}

#[test]
fn stft_demucs_dims_reference() {
    let n_fft = 4096usize;
    let hop = 1024usize;
    let t = 343_980usize;
    let left = vec![0.0f32; t];
    let right = vec![0.0f32; t];
    let (_spec, f_bins, frames) = stft_cac_stereo_centered(&left, &right, n_fft, hop);
    assert_eq!(f_bins, 2048);
    assert_eq!(frames, 1 + (t / hop));
}
