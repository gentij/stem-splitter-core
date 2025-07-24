use crate::{audio::read_wav_mono, model::StemModel, types::StemResult};
use anyhow::Result;

pub fn split_file(path: &str, model: &impl StemModel) -> Result<StemResult> {
    let audio = read_wav_mono(path)?;
    model.separate(&audio)
}
