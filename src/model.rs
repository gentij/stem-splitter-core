use crate::audio::{read_audio, write_audio};
use crate::types::{AudioData, StemResult};
use crate::utils::script_path;
use anyhow::{Context, Result};
use std::{fs, path::Path, process::Command};

pub trait StemModel {
    fn separate(&self, input: &[f32], channels: u16, output_dir: &Path) -> Result<StemResult>;
}

pub struct PythonModel;

impl PythonModel {
    pub fn new() -> Self {
        Self
    }
}

impl StemModel for PythonModel {
    fn separate(&self, input: &[f32], channels: u16, output_dir: &Path) -> Result<StemResult> {
        fs::create_dir_all(output_dir)?;

        let input_wav = output_dir.join("input.wav");

        // Only duplicate samples if mono
        let (final_samples, final_channels): (Vec<f32>, u16) = if channels == 1 {
            (
                input
                    .iter()
                    .flat_map(|s| std::iter::repeat(*s).take(2))
                    .collect(),
                2,
            )
        } else {
            (input.to_vec(), channels)
        };

        let input_audio = AudioData {
            samples: final_samples,
            sample_rate: 44100,
            channels: final_channels,
        };

        println!("🔍 Wrote input WAV to: {}", input_wav.display());
        println!("📂 Expecting output in: {}", output_dir.display());

        write_audio(input_wav.to_str().unwrap(), &input_audio)?;

        // Path to Python script
        let script = script_path();

        // Run Python stem separation
        let status = Command::new("python3")
            .arg(script)
            .arg("--input")
            .arg(&input_wav)
            .arg("--output")
            .arg(&output_dir)
            .status()
            .context("Failed to run Python stem splitter script")?;

        if !status.success() {
            return Err(anyhow::anyhow!("Python stem splitter script failed"));
        }

        // Read stems
        let vocals = read_audio(output_dir.join("vocals.wav"))?.samples;
        let drums = read_audio(output_dir.join("drums.wav"))?.samples;
        let bass = read_audio(output_dir.join("bass.wav"))?.samples;
        let other = read_audio(output_dir.join("other.wav"))?.samples;

        Ok(StemResult {
            vocals,
            drums,
            bass,
            other,
        })
    }
}
