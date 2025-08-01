use crate::{
    audio::{read_audio, write_audio},
    model::{PythonModel, StemModel},
    types::{AudioData, SplitConfig, StemResult},
};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn split_file(path: &str, config: SplitConfig) -> Result<StemResult> {
    std::fs::metadata(path).with_context(|| format!("File does not exist: {}", path))?;

    let model = PythonModel::new();

    let audio = read_audio(path)?;
    let tmp_output_dir = PathBuf::from("./tmp");
    let result = model
        .separate(&audio.samples, audio.channels, &tmp_output_dir)
        .with_context(|| format!("Failed to separate stems for file: {}", path))?;

    let file_stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let base_path = PathBuf::from(&config.output_dir).join(file_stem);
    save_stems(&result, base_path.to_str().unwrap(), audio.sample_rate)?;

    Ok(result)
}

fn save_stems(result: &StemResult, base_path: &str, sample_rate: u32) -> Result<()> {
    let make_path = |stem: &str| format!("{base_path}_{stem}.wav");

    let stems = [
        ("vocals", &result.vocals),
        ("drums", &result.drums),
        ("bass", &result.bass),
        ("other", &result.other),
    ];

    for (name, samples) in stems {
        write_audio(
            &make_path(name),
            &AudioData {
                samples: samples.clone(),
                sample_rate,
                channels: 1,
            },
        )?;
    }

    Ok(())
}
