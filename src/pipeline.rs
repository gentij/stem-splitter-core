use crate::{audio::read_audio, model::StemModel, types::StemResult};
use anyhow::Result;

pub fn split_file(path: &str, model: &impl StemModel) -> Result<StemResult> {
    let audio = read_audio(path)?;
    model.separate(&audio.samples)
}
