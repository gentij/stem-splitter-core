use crate::{
    core::{
        audio::{create_wav_writer, read_audio, sample_to_i16, WavWriter},
        engine,
    },
    error::Result,
    io::progress::{emit_split_progress, SplitProgress},
    model::model_manager::ensure_model,
    types::{SplitOptions, SplitResult},
};

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

struct StemOutput {
    stem_idx: usize,
    stem_name: String,
    writer: WavWriter,
}

fn audio_frame_count(samples: &[f32], channels: u16) -> usize {
    let channels = usize::from(channels.max(1));
    samples.len() / channels
}

fn fill_stereo_window(
    samples: &[f32],
    channels: u16,
    start_frame: usize,
    left_raw: &mut [f32],
    right_raw: &mut [f32],
) {
    let channels = usize::from(channels.max(1));

    for i in 0..left_raw.len() {
        let frame = start_frame + i;
        let base = frame * channels;
        if base >= samples.len() {
            left_raw[i] = 0.0;
            right_raw[i] = 0.0;
            continue;
        }

        let left = samples[base];
        let right = if channels == 1 {
            left
        } else {
            samples.get(base + 1).copied().unwrap_or(left)
        };

        left_raw[i] = left;
        right_raw[i] = right;
    }
}

fn build_output_paths(input_path: &str, output_dir: &str) -> (String, String, String, String) {
    let file_stem = Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let base = PathBuf::from(output_dir).join(file_stem);

    (
        format!("{}_vocals.wav", base.to_string_lossy()),
        format!("{}_drums.wav", base.to_string_lossy()),
        format!("{}_bass.wav", base.to_string_lossy()),
        format!("{}_other.wav", base.to_string_lossy()),
    )
}

fn build_stem_outputs(
    names: &[String],
    stems_count: usize,
    sample_rate: u32,
    vocals_out: String,
    drums_out: String,
    bass_out: String,
    other_out: String,
) -> Result<Vec<StemOutput>> {
    let mut name_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        name_idx.insert(name.to_lowercase(), i);
    }

    let get_idx = |key: &str, fallback: usize| -> usize {
        name_idx
            .get(key)
            .copied()
            .unwrap_or(fallback.min(stems_count.saturating_sub(1)))
    };

    Ok(vec![
        StemOutput {
            stem_idx: get_idx("vocals", 0),
            stem_name: "vocals".to_string(),
            writer: create_wav_writer(&vocals_out, sample_rate, 2)?,
        },
        StemOutput {
            stem_idx: get_idx("drums", 1),
            stem_name: "drums".to_string(),
            writer: create_wav_writer(&drums_out, sample_rate, 2)?,
        },
        StemOutput {
            stem_idx: get_idx("bass", 2),
            stem_name: "bass".to_string(),
            writer: create_wav_writer(&bass_out, sample_rate, 2)?,
        },
        StemOutput {
            stem_idx: get_idx("other", 3),
            stem_name: "other".to_string(),
            writer: create_wav_writer(&other_out, sample_rate, 2)?,
        },
    ])
}

pub fn split_file(input_path: &str, opts: SplitOptions) -> Result<SplitResult> {
    emit_split_progress(SplitProgress::Stage("resolve_model"));
    let handle = ensure_model(&opts.model_name, opts.manifest_url_override.as_deref())?;

    emit_split_progress(SplitProgress::Stage("engine_preload"));
    engine::preload(&handle)?;

    let mf = engine::manifest();

    if mf.sample_rate != 44100 {
        return Err(anyhow::anyhow!("Currently expecting 44.1k model").into());
    }

    emit_split_progress(SplitProgress::Stage("read_audio"));
    let audio = read_audio(input_path)?;
    let n = audio_frame_count(&audio.samples, audio.channels);

    if n == 0 {
        return Err(anyhow::anyhow!("Empty audio").into());
    }

    let win = mf.window;
    let hop = mf.hop;

    if !(win > 0 && hop > 0 && hop <= win) {
        return Err(anyhow::anyhow!("Bad win/hop in manifest").into());
    }

    if std::env::var("DEBUG_STEMS").is_ok() {
        eprintln!(
            "Window settings: win={}, hop={}, overlap={}",
            win,
            hop,
            win - hop
        );
    }

    let names = if mf.stems.is_empty() {
        vec![
            "vocals".into(),
            "drums".into(),
            "bass".into(),
            "other".into(),
        ]
    } else {
        mf.stems.clone()
    };

    let (vocals_out, drums_out, bass_out, other_out) =
        build_output_paths(input_path, &opts.output_dir);

    let mut left_raw = vec![0f32; win];
    let mut right_raw = vec![0f32; win];
    let mut stem_outputs: Vec<StemOutput> = Vec::new();

    let mut pos = 0usize;
    let mut chunk_done = 0usize;
    let total_chunks = if n <= hop { 1 } else { (n - 1) / hop + 1 };
    let mut first_chunk = true;

    emit_split_progress(SplitProgress::Stage("infer"));
    while pos < n {
        fill_stereo_window(
            &audio.samples,
            audio.channels,
            pos,
            &mut left_raw,
            &mut right_raw,
        );

        let out = engine::run_window_demucs(&left_raw, &right_raw)?;
        let (stems_count, _, t_out) = (out.shape()[0], out.shape()[1], out.shape()[2]);

        if first_chunk {
            stem_outputs = build_stem_outputs(
                &names,
                stems_count,
                mf.sample_rate,
                vocals_out.clone(),
                drums_out.clone(),
                bass_out.clone(),
                other_out.clone(),
            )?;
            first_chunk = false;
        }

        let copy_len = hop.min(t_out).min(n - pos);
        for stem_output in &mut stem_outputs {
            for i in 0..copy_len {
                stem_output
                    .writer
                    .write_sample(sample_to_i16(out[(stem_output.stem_idx, 0, i)]))
                    .map_err(anyhow::Error::from)?;
                stem_output
                    .writer
                    .write_sample(sample_to_i16(out[(stem_output.stem_idx, 1, i)]))
                    .map_err(anyhow::Error::from)?;
            }
        }

        chunk_done += 1;
        emit_split_progress(SplitProgress::Chunks {
            done: chunk_done,
            total: total_chunks,
            percent: chunk_done as f32 / total_chunks as f32 * 100.0,
        });

        if pos + hop >= n {
            break;
        }
        pos += hop;
    }

    emit_split_progress(SplitProgress::Stage("write_stems"));
    for (idx, stem_output) in stem_outputs.into_iter().enumerate() {
        emit_split_progress(SplitProgress::Writing {
            stem: stem_output.stem_name,
            done: idx + 1,
            total: 4,
            percent: (idx + 1) as f32 / 4.0 * 100.0,
        });
        stem_output.writer.finalize().map_err(anyhow::Error::from)?;
    }

    emit_split_progress(SplitProgress::Stage("finalize"));
    emit_split_progress(SplitProgress::Finished);

    Ok(SplitResult {
        vocals_path: vocals_out,
        drums_path: drums_out,
        bass_path: bass_out,
        other_path: other_out,
    })
}
