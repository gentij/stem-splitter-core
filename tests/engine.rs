#[cfg(not(feature = "engine-mock"))]
#[test]
fn run_window_demucs_rejects_len_mismatch() {
    use stem_splitter_core::core::engine::run_window_demucs;
    let left = vec![0.0f32; 1000];
    let right = vec![0.0f32; 999];
    let _ = run_window_demucs(&left, &right).unwrap_err();
}

#[cfg(not(feature = "engine-mock"))]
#[test]
fn run_window_demucs_rejects_wrong_t() {
    use stem_splitter_core::core::engine::run_window_demucs;
    // any T ≠ DEMUCS_T should error in real engine
    let left = vec![0.0f32; 1024];
    let right = vec![0.0f32; 1024];
    let _ = run_window_demucs(&left, &right).unwrap_err();
}

#[cfg(feature = "engine-mock")]
#[test]
fn engine_mock_accepts_any_t_and_returns_identity_stems() {
    use stem_splitter_core::core::engine::run_window_demucs;
    let t = 1024;
    let left = vec![1.0f32; t];
    let right = vec![0.5f32; t];
    let out = run_window_demucs(&left, &right).unwrap();
    assert_eq!(out.shape(), &[4, 2, t]);

    for s in 0..4 {
        assert_eq!(out[(s, 0, 0)], 1.0);
        assert_eq!(out[(s, 1, 0)], 0.5);
    }
}
