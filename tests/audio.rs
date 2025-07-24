use std::fs;
use std::path::Path;
use stem_splitter_core::{read_audio, write_audio};

#[test]
fn test_read_and_write_audio() {
    let input_path = "assets/test.wav";
    let output_path = "assets/test_output.wav";

    assert!(
        Path::new(input_path).exists(),
        "Test input WAV file missing"
    );

    let audio = read_audio(input_path).expect("Failed to read audio file");
    assert!(!audio.samples.is_empty(), "No samples read");
    assert!(audio.sample_rate > 0, "Invalid sample rate");

    println!(
        "Read {} samples @ {}Hz from {}",
        audio.samples.len(),
        audio.sample_rate,
        input_path
    );

    write_audio(output_path, &audio).expect("Failed to write audio file");

    assert!(
        Path::new(output_path).exists(),
        "Output file was not written"
    );

    let reread = read_audio(output_path).expect("Failed to re-read written file");
    assert_eq!(reread.sample_rate, audio.sample_rate);
    assert_eq!(reread.channels, audio.channels);
    assert_eq!(reread.samples.len(), audio.samples.len());

    fs::remove_file(output_path).expect("Failed to delete test output");
}
