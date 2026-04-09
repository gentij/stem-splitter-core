use std::{fs::File, io::BufWriter, path::Path};

use anyhow::{Context, Result};
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};
use symphonia::default::{get_codecs, get_probe};

use crate::types::AudioData;

pub type WavWriter = hound::WavWriter<BufWriter<File>>;

pub fn read_audio<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let path: &Path = path.as_ref();

    let file: File =
        File::open(path).with_context(|| format!("Failed to open audio file: {:?}", path))?;

    let mss: MediaSourceStream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint: Hint = Hint::new();

    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format.default_track().context("No default track found")?;

    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate: u32 = 0;
    let mut channels: u16 = 0;

    while let Ok(packet) = format.next_packet() {
        let decoded = decoder.decode(&packet)?;
        sample_rate = decoded.spec().rate;
        channels = decoded.spec().channels.count() as u16;

        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buffer.copy_interleaved_ref(decoded);

        samples.extend_from_slice(buffer.samples());
    }

    if std::env::var("DEBUG_STEMS").is_ok() {
        eprintln!(
            "🎧 Read audio: sample_rate={} Hz, channels={}, samples={} ({:.2} seconds)",
            sample_rate,
            channels,
            samples.len(),
            samples.len() as f64 / (sample_rate as f64 * channels as f64)
        );
    }

    Ok(AudioData {
        samples,
        sample_rate,
        channels,
    })
}

pub fn write_audio(path: &str, audio: &AudioData) -> Result<()> {
    let mut writer = create_wav_writer(path, audio.sample_rate, audio.channels)?;
    for sample in &audio.samples {
        writer.write_sample(sample_to_i16(*sample))?;
    }

    writer.finalize()?;
    Ok(())
}

pub fn create_wav_writer<P: AsRef<Path>>(
    path: P,
    sample_rate: u32,
    channels: u16,
) -> Result<WavWriter> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    Ok(hound::WavWriter::create(path, spec)?)
}

pub fn sample_to_i16(sample: f32) -> i16 {
    (sample * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16
}
