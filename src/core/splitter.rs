use crate::{
    core::{
        audio::{read_audio, write_audio},
        dsp::{hann2, to_planar_stereo},
        engine,
    },
    error::Result,
    model::model_manager::ensure_model,
    types::{AudioData, SplitOptions, SplitResult},
};

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use tempfile::tempdir;

pub fn split_file(input_path: &str, opts: SplitOptions) -> Result<SplitResult> {
    let handle = ensure_model(&opts.model_name, opts.manifest_url_override.as_deref())?;
    engine::preload(&handle)?;

    let mf = engine::manifest();

    if mf.sample_rate != 44100 {
        return Err(anyhow::anyhow!("Currently expecting 44.1k model").into());
    }

    let audio = read_audio(input_path)?;
    let stereo = to_planar_stereo(&audio.samples, audio.channels);
    let n = stereo.len();

    if n == 0 {
        return Err(anyhow::anyhow!("Empty audio").into());
    }

    let win = mf.window;
    let hop = mf.hop;

    if !(win > 0 && hop > 0 && hop <= win) {
        return Err(anyhow::anyhow!("Bad win/hop in manifest").into());
    }

    let w = hann2(win);

    let stems_names = mf.stems.clone();
    let mut stems_count = stems_names.len().max(1);
    let mut acc: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut wsum = vec![0f32; n + win];

    let tmp = tempdir()?;
    let tmp_dir = tmp.path().to_path_buf();

    let mut left_raw = vec![0f32; win];
    let mut right_raw = vec![0f32; win];
    let mut first_pass: Option<(usize, usize)> = None;

    let mut pos = 0usize;

    while pos < n {
        // 1) Build raw chunk for the model (no Hann here)
        for i in 0..win {
            let idx = pos + i;
            let (l, r) = if idx < n {
                (stereo[idx][0], stereo[idx][1])
            } else {
                (0.0, 0.0)
            };
            left_raw[i] = l;
            right_raw[i] = r;
        }

        // 2) Inference: [S,2,T] with Demucs runner
        let out = engine::run_window_demucs(&left_raw, &right_raw)?;
        let (s_count, _, t_out) = (out.shape()[0], out.shape()[1], out.shape()[2]);

        if first_pass.is_none() {
            first_pass = Some((s_count, t_out));
            stems_count = s_count;
            acc = vec![vec![[0f32; 2]; n + win]; stems_count];
        }

        // 3) Overlap-add with Hann^2 weights (your existing scheme)
        for st in 0..stems_count {
            for i in 0..win {
                let dst = pos + i;
                if dst >= acc[st].len() {
                    break;
                }
                let ww = w[i];
                if i < t_out {
                    acc[st][dst][0] += out[(st, 0, i)] * ww;
                    acc[st][dst][1] += out[(st, 1, i)] * ww;
                }
            }
        }
        for i in 0..win {
            let dst = pos + i;
            if dst >= wsum.len() {
                break;
            }
            let ww = w[i];
            wsum[dst] += ww * ww;
        }

        if pos + hop >= n {
            break;
        }
        pos += hop;
    }

    for i in 0..n {
        let d = wsum[i].max(1e-8);
        for st in 0..stems_count {
            acc[st][i][0] /= d;
            acc[st][i][1] /= d;
        }
    }

    let names = if stems_names.is_empty() {
        vec![
            "vocals".into(),
            "drums".into(),
            "bass".into(),
            "other".into(),
        ]
    } else {
        stems_names
    };
    let mut name_idx: HashMap<String, usize> = HashMap::new();
    for (i, n) in names.iter().enumerate() {
        name_idx.insert(n.to_lowercase(), i);
    }

    fs::create_dir_all(&opts.output_dir)?;

    let stem_to_wav = |st: usize, base: &str| -> Result<String> {
        let mut inter = Vec::<f32>::with_capacity(n * 2);
        for i in 0..n {
            inter.push(acc[st][i][0]);
            inter.push(acc[st][i][1]);
        }
        let data = AudioData {
            samples: inter,
            sample_rate: mf.sample_rate,
            channels: 2,
        };
        let p = tmp_dir.join(format!("{base}.wav"));
        write_audio(p.to_str().unwrap(), &data)?;
        Ok(p.to_string_lossy().into())
    };

    let get_idx = |key: &str, fallback: usize| -> usize {
        name_idx
            .get(key)
            .copied()
            .unwrap_or(fallback.min(stems_count.saturating_sub(1)))
    };

    let v_path = stem_to_wav(get_idx("vocals", 0), "vocals")?;
    let d_path = stem_to_wav(get_idx("drums", 1), "drums")?;
    let b_path = stem_to_wav(get_idx("bass", 2), "bass")?;
    let o_path = stem_to_wav(get_idx("other", 3), "other")?;

    let file_stem = Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let base = PathBuf::from(&opts.output_dir).join(file_stem);

    let vocals_out = copy_to(&v_path, &format!("{}_vocals.wav", base.to_string_lossy()))?;
    let drums_out = copy_to(&d_path, &format!("{}_drums.wav", base.to_string_lossy()))?;
    let bass_out = copy_to(&b_path, &format!("{}_bass.wav", base.to_string_lossy()))?;
    let other_out = copy_to(&o_path, &format!("{}_other.wav", base.to_string_lossy()))?;

    Ok(SplitResult {
        vocals_path: vocals_out,
        drums_path: drums_out,
        bass_path: bass_out,
        other_path: other_out,
    })
}

fn copy_to(src: &str, dst: &str) -> Result<String> {
    fs::copy(src, dst)?;
    Ok(dst.to_string())
}
