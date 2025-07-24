use std::path::Path;
use stem_splitter_core::read_wav_mono;

#[test]
fn test_read_wav_file() {
    let path = "assets/test.wav";
    assert!(Path::new(path).exists(), "Missing test file: {}", path);

    let samples = read_wav_mono(path).expect("Failed to read audio");
    assert!(!samples.is_empty(), "Read 0 samples");

    println!(
        "Read {} samples. First sample: {:?}",
        samples.len(),
        samples[0]
    );
}
