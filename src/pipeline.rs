use crate::{
    audio::{downmix_to_mono, read_audio, write_audio},
    model::{OnnxModel, StemModel},
    types::{AudioData, StemResult},
};
use anyhow::{Context, Result};

pub fn split_file(path: &str, model_name: &str) -> Result<StemResult> {
    std::fs::metadata(path).with_context(|| format!("File does not exist: {}", path))?;

    let model = OnnxModel::new(model_name);

    let audio = read_audio(path)?;
    let mono = downmix_to_mono(&audio.samples, audio.channels);

    model.separate(&mono)
}

fn save_stems(result: &StemResult, base_path: &str, sample_rate: u32) -> Result<()> {
    let make_path = |stem: &str| format!("{base_path}_{stem}.wav");

    write_audio(
        &make_path("vocals"),
        &AudioData {
            samples: result.vocals.clone(),
            sample_rate,
            channels: 1,
        },
    )?;

    // Repeat for other stems...

    Ok(())
}
