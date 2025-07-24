// tests/pipeline.rs

use std::path::Path;
use stem_splitter_core::split_file;

#[test]
fn test_split_file_pipeline_mock() {
    let input_path = "assets/test.wav";

    assert!(
        Path::new(input_path).exists(),
        "Test audio file is missing: {}",
        input_path
    );

    let model_name = "mock-demucs";

    let result = split_file(input_path, model_name).expect("Pipeline split_file failed");

    let total_len =
        result.vocals.len() + result.drums.len() + result.bass.len() + result.other.len();

    assert!(total_len > 0, "All stems are empty");

    println!(
        "✅ Split complete: vocals={}, drums={}, bass={}, other={}",
        result.vocals.len(),
        result.drums.len(),
        result.bass.len(),
        result.other.len()
    );
}
