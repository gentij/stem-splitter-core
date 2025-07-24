use crate::{
    audio::{downmix_to_mono, read_audio, write_audio},
    model::{OnnxModel, StemModel},
    types::{AudioData, StemResult},
};
use anyhow::{Context, Result};
use std::path::Path;

pub fn split_file(path: &str, model_name: &str) -> Result<StemResult> {
    std::fs::metadata(path).with_context(|| format!("File does not exist: {}", path))?;

    let model = OnnxModel::new(model_name);

    let audio = read_audio(path)?;
    let mono = downmix_to_mono(&audio.samples, audio.channels);

    let result = model
        .separate(&mono)
        .with_context(|| format!("Failed to separate stems for file: {}", path))?;

    let file_stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let base_path = format!("{}_output", file_stem);
    save_stems(&result, &base_path, audio.sample_rate)?;

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
