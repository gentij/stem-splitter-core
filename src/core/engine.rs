#![cfg_attr(feature = "engine-mock", allow(dead_code, unused_imports))]

use crate::{
    core::dsp::{istft_cac_stereo, stft_cac_stereo_centered},
    error::{Result, StemError},
    model::model_manager::ModelHandle,
    types::ModelManifest,
};

use anyhow::anyhow;
use ndarray::Array3;
use once_cell::sync::OnceCell;
use ort::{
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
    value::{Tensor, Value},
};
use std::sync::Mutex;

static SESSION: OnceCell<Mutex<Session>> = OnceCell::new();
static MANIFEST: OnceCell<ModelManifest> = OnceCell::new();
static ORT_INIT: OnceCell<()> = OnceCell::new();

const DEMUCS_T: usize = 343_980;
const DEMUCS_F: usize = 2048;
const DEMUCS_FRAMES: usize = 336;
const DEMUCS_NFFT: usize = 4096;
const DEMUCS_HOP: usize = 1024;

#[cfg(not(feature = "engine-mock"))]
pub fn preload(h: &ModelHandle) -> Result<()> {
    ORT_INIT.get_or_try_init::<_, StemError>(|| {
        ort::init().commit().map_err(StemError::from)?;
        Ok(())
    })?;

    // Use more threads for better performance
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let session = SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_threads)?
        .with_inter_threads(num_threads)?
        .with_parallel_execution(true)?
        .commit_from_file(&h.local_path)?;

    SESSION.set(Mutex::new(session)).ok();
    MANIFEST.set(h.manifest.clone()).ok();
    Ok(())
}

#[cfg(not(feature = "engine-mock"))]
pub fn manifest() -> &'static ModelManifest {
    MANIFEST
        .get()
        .expect("engine::preload() must be called once before using the engine")
}

#[cfg(not(feature = "engine-mock"))]
pub fn run_window_demucs(left: &[f32], right: &[f32]) -> Result<Array3<f32>> {
    if left.len() != right.len() {
        return Err(anyhow!("L/R length mismatch").into());
    }
    let t = left.len();
    if t != DEMUCS_T {
        return Err(anyhow!("Bad window length {} (expected {})", t, DEMUCS_T).into());
    }

    // Build time branch [1,2,T], planar
    let mut planar = Vec::with_capacity(2 * t);
    planar.extend_from_slice(left);
    planar.extend_from_slice(right);
    let time_value: Value = Tensor::from_array((vec![1, 2, t], planar))?.into_dyn();

    // Build spec branch [1,4,F,Frames] with center padding, Hann, 4096/1024
    let (spec_cac, f_bins, frames) = stft_cac_stereo_centered(left, right, DEMUCS_NFFT, DEMUCS_HOP);
    if f_bins != DEMUCS_F || frames != DEMUCS_FRAMES {
        return Err(anyhow!(
            "Spec dims mismatch: got F={},Frames={}, expected F={},Frames={}",
            f_bins,
            frames,
            DEMUCS_F,
            DEMUCS_FRAMES
        )
        .into());
    }
    let spec_value: Value = Tensor::from_array((vec![1, 4, f_bins, frames], spec_cac))?.into_dyn();

    let mut session = SESSION
        .get()
        .expect("engine::preload first")
        .lock()
        .expect("session poisoned");

    // Get input names
    let in_time = session
        .inputs
        .iter()
        .find(|i| i.name == "input")
        .map(|i| i.name.clone())
        .ok_or_else(|| anyhow!("Model missing input 'input'"))?;

    let in_spec = session
        .inputs
        .iter()
        .find(|i| i.name == "x")
        .map(|i| i.name.clone())
        .ok_or_else(|| anyhow!("Model missing input 'x'"))?;

    // Run inference
    let outputs = session.run(vec![(in_time, time_value), (in_spec, spec_value)])?;

    // Extract both outputs from the model
    // "output": frequency domain [1, sources, 4, F, Frames]
    // "add_67": time domain [1, sources, 2, T]
    let mut output_freq: Option<Value> = None;
    let mut output_time: Option<Value> = None;

    for (name, val) in outputs.into_iter() {
        if name == "output" {
            output_freq = Some(val);
        } else if name == "add_67" {
            output_time = Some(val);
        }
    }

    let out_freq =
        output_freq.ok_or_else(|| anyhow!("Model did not return 'output' (freq domain)"))?;
    let out_time =
        output_time.ok_or_else(|| anyhow!("Model did not return 'add_67' (time domain)"))?;

    // Extract time domain output [1, 4, 2, T] -> [4, 2, T]
    let (shape_time, data_time) = out_time.try_extract_tensor::<f32>()?;
    let num_sources = shape_time[1] as usize;

    // Extract frequency domain output [1, sources, 4, F, Frames]
    let (shape_freq, data_freq) = out_freq.try_extract_tensor::<f32>()?;

    // Validate shapes
    if shape_freq[0] != 1
        || shape_freq[1] != num_sources as i64
        || shape_freq[2] != 4
        || shape_freq[3] != f_bins as i64
        || shape_freq[4] != frames as i64
    {
        return Err(anyhow!(
            "Unexpected freq output shape: {:?}, expected [1, {}, 4, {}, {}]",
            shape_freq,
            num_sources,
            f_bins,
            frames
        )
        .into());
    }

    // Combine frequency and time domain outputs
    // According to demucs.onnx: final = time_domain + istft(frequency_domain)
    let mut result = Vec::with_capacity(num_sources * 2 * t);

    for src in 0..num_sources {
        // Extract frequency domain for this source [4, F, Frames]
        let src_freq_offset = src * 4 * f_bins * frames;
        let src_freq_data = &data_freq[src_freq_offset..src_freq_offset + 4 * f_bins * frames];

        // Apply iSTFT to convert frequency domain to time domain
        let (left_freq, right_freq) =
            istft_cac_stereo(src_freq_data, f_bins, frames, DEMUCS_NFFT, DEMUCS_HOP, t);

        // Extract time domain for this source [2, T]
        let src_time_offset = src * 2 * t;
        let left_time = &data_time[src_time_offset..src_time_offset + t];
        let right_time = &data_time[src_time_offset + t..src_time_offset + 2 * t];

        // Combine: output = time_domain + frequency_domain (after iSTFT)
        for i in 0..t {
            result.push(left_time[i] + left_freq[i]);
        }
        for i in 0..t {
            result.push(right_time[i] + right_freq[i]);
        }
    }

    let out = ndarray::Array3::from_shape_vec((num_sources, 2, t), result)?;
    Ok(out)
}

#[cfg(feature = "engine-mock")]
mod _engine_mock {
    use super::*;
    use once_cell::sync::OnceCell;
    static MANIFEST: OnceCell<ModelManifest> = OnceCell::new();

    pub fn preload(h: &ModelHandle) -> Result<()> {
        MANIFEST.set(h.manifest.clone()).ok();
        Ok(())
    }

    pub fn manifest() -> &'static ModelManifest {
        MANIFEST.get().expect("preload first (mock)")
    }

    pub fn run_window_demucs(left: &[f32], right: &[f32]) -> Result<Array3<f32>> {
        let t = left.len().min(right.len());
        let sources = 4usize;
        let mut out = vec![0.0f32; sources * 2 * t];
        for s in 0..sources {
            for i in 0..t {
                // “identity” stems: copy input
                out[s * 2 * t + i] = left[i]; // L
                out[s * 2 * t + t + i] = right[i]; // R
            }
        }
        Ok(ndarray::Array3::from_shape_vec((sources, 2, t), out)?)
    }
}

#[cfg(feature = "engine-mock")]
pub use _engine_mock::{manifest, preload, run_window_demucs};
