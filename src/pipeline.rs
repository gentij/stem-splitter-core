use crate::utils::tmp_dir;
use crate::{
    audio::read_audio,
    model::{PythonModel, StemModel},
    types::{SplitConfig, StemResult},
};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn split_file(path: &str, config: SplitConfig) -> Result<StemResult> {
    std::fs::metadata(path).with_context(|| format!("File does not exist: {}", path))?;

    let model = PythonModel::new();

    let audio = read_audio(path)?;
    let tmp_output_dir = tmp_dir();
    let result = model
        .separate(&audio.samples, audio.channels, &tmp_output_dir)
        .with_context(|| format!("Failed to separate stems for file: {}", path))?;

    let file_stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let base_path = PathBuf::from(&config.output_dir).join(file_stem);
    save_stems(base_path.to_str().unwrap())?;

    std::fs::remove_dir_all(&tmp_output_dir).with_context(|| {
        format!(
            "Failed to remove tmp directory: {}",
            tmp_output_dir.display()
        )
    })?;

    Ok(result)
}

fn save_stems(base_path: &str) -> Result<()> {
    let stem_names = ["vocals", "drums", "bass", "other"];

    // Ensure parent directory of base_path exists
    if let Some(parent) = Path::new(base_path).parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    for name in stem_names {
        let src = Path::new("tmp").join(format!("{name}.wav")); // or your actual tmp_output_dir
        let dst = format!("{base_path}_{name}.wav");

        std::fs::copy(&src, &dst)
            .with_context(|| format!("Failed to copy {name} from tmp to output"))?;
    }

    Ok(())
}
