use anyhow::Result;
use hound::WavReader;

pub fn read_wav_mono(path: &str) -> Result<Vec<f32>> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("Failed to open WAV file: {}", path))?;

    let spec = reader.spec();

    if spec.sample_format != SampleFormat::Int {
        anyhow::bail!("Only PCM (integer) format is supported.");
    }

    let norm_factor = match spec.bits_per_sample {
        16 => i16::MAX as f32,
        24 => (1 << 23) as f32,
        32 => i32::MAX as f32,
        _ => anyhow::bail!("Unsupported bit depth: {} bits", spec.bits_per_sample),
    };

    let num_channels = spec.channels;

    let samples = match spec.bits_per_sample {
        16 => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / norm_factor))
            .collect::<Result<Vec<_>, _>>()?,
        24 | 32 => {
            anyhow::bail!("24- and 32-bit WAV files are not currently supported")
        }
        _ => anyhow::bail!("Unsupported bit depth"),
    };

    if num_channels == 1 {
        Ok(samples)
    } else if num_channels == 2 {
        let mono_samples = samples
            .chunk(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect();
        Ok(samples)
    } else {
        anyhow::bail!("Unsupported number of channels: {}", num_channels);
    }
}
