use std::path::Path;
use stem_splitter_core::{split_file, SplitConfig};

#[test]
fn test_split_file_pipeline_and_stem_output() {
    let input_path = "assets/test.wav";
    assert!(Path::new(input_path).exists(), "Test audio file is missing");

    let config = SplitConfig::default().output_dir("output");

    let result = split_file(input_path, config).expect("Pipeline failed");

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

    let stem_base = "output/test";
    for name in ["vocals", "drums", "bass", "other"] {
        let path = format!("{}_{}.wav", stem_base, name);
        assert!(
            Path::new(&path).exists(),
            "Stem file was not written: {}",
            path
        );
    }

    println!("✅ All stem files written successfully");
}
