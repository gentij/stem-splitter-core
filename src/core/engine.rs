#![cfg_attr(feature = "engine-mock", allow(dead_code, unused_imports))]

use crate::{
    core::dsp::{istft_cac_stereo_parallel, stft_cac_stereo_centered},
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
// 1. IMPORT THE TRAIT REQUIRED FOR GPU REGISTRATION
use ort::ep::ExecutionProvider;

static SESSION: OnceCell<Mutex<Session>> = OnceCell::new();
static MANIFEST: OnceCell<ModelManifest> = OnceCell::new();
static ORT_INIT: OnceCell<()> = OnceCell::new();

const DEMUCS_T: usize = 343_980;
const DEMUCS_F: usize = 2048;
const DEMUCS_FRAMES: usize = 336;
const DEMUCS_NFFT: usize = 4096;
const DEMUCS_HOP: usize = 1024;

// 2. UNIFIED SESSION BUILDER
#[cfg(not(feature = "engine-mock"))]
fn commit_session(model_path: &std::path::Path, num_threads: usize) -> Result<Session> {
    let mut builder = SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_threads)?
        .with_inter_threads(num_threads)?
        .with_parallel_execution(true)?;

    #[allow(unused_assignments)]
    let mut attempted_gpu = false;

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    {
        attempted_gpu = true;
        println!("🚀 Attempting to initialize CUDA Execution Provider...");
        let cuda = ort::ep::CUDAExecutionProvider::default();
        if let Err(e) = cuda.register(&mut builder) {
            eprintln!("❌ CUDA Registration Failed: {}. \nMake sure CUDA Toolkit and cuDNN are correctly installed.", e);
        } else {
            println!("✅ CUDA Execution Provider registered successfully.");
        }
    }

    #[cfg(all(feature = "coreml", target_os = "macos"))]
    if std::env::var("ENABLE_COREML").is_ok() {
        attempted_gpu = true;
        println!("🚀 Attempting to initialize CoreML Execution Provider...");
        let coreml = ort::ep::CoreMLExecutionProvider::default();
        if let Err(e) = coreml.register(&mut builder) {
            eprintln!("❌ CoreML Registration Failed: {}", e);
        } else {
            println!("✅ CoreML Execution Provider registered successfully.");
        }
    }

    #[cfg(feature = "onednn")]
    {
        println!("🚀 Attempting to initialize oneDNN...");
        let onednn = ort::ep::OneDNNExecutionProvider::default();
        let _ = onednn.register(&mut builder);
    }

    if !attempted_gpu {
        println!("⚠️ No GPU features matched your OS/Target. Defaulting to CPU.");
    }

    println!("⏳ Committing ONNX session (this may take a moment)...");
    match builder.commit_from_file(model_path) {
        Ok(session) => {
            println!("✅ Session committed successfully.");
            Ok(session)
        },
        Err(e) => {
            eprintln!("💥 Failed to commit session: {}", e);
            Err(e.into())
        }
    }
}


// 3. CLEAN PRELOAD FUNCTION
#[cfg(not(feature = "engine-mock"))]
pub fn preload(h: &ModelHandle) -> Result<()> {
    ORT_INIT.get_or_try_init::<_, StemError>(|| {
        let _ = ort::init().with_name("stem-splitter").commit();
        Ok(())
    })?;

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    if std::env::var("STEMMER_FORCE_CPU").is_ok() {
        println!("ℹ️ STEMMER_FORCE_CPU is set: using CPU only");
    }

    let session = commit_session(h.local_path.as_path(), num_threads)?;

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

    // In ort 2.0.0-rc.11, we bypass private fields and safely assign 
    // inputs directly via the macro.
    let outputs = session.run(ort::inputs![
        "input" => time_value,
        "x" => spec_value
    ])?;
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

    // Debug: Check if model outputs are non-zero
    if std::env::var("DEBUG_STEMS").is_ok() {
        let time_max = data_time.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let freq_max = data_freq.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "Model output stats: time_max={:.6}, freq_max={:.6}",
            time_max, freq_max
        );
        if time_max < 1e-10 && freq_max < 1e-10 {
            eprintln!("WARNING: Model outputs are all zeros! This indicates a problem with the execution provider.");
        }
    }

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

    let source_specs: Vec<&[f32]> = (0..num_sources)
        .map(|src| {
            let src_freq_offset = src * 4 * f_bins * frames;
            &data_freq[src_freq_offset..src_freq_offset + 4 * f_bins * frames]
        })
        .collect();

    let istft_results =
        istft_cac_stereo_parallel(&source_specs, f_bins, frames, DEMUCS_NFFT, DEMUCS_HOP, t);

    // Debug: Check iSTFT results
    if std::env::var("DEBUG_STEMS").is_ok() {
        for (src_idx, (left, right)) in istft_results.iter().enumerate() {
            let left_max = left.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let right_max = right.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "iSTFT result [source {}]: left_max={:.6}, right_max={:.6}",
                src_idx, left_max, right_max
            );
        }
    }

    let mut result = Vec::with_capacity(num_sources * 2 * t);

    for (src, (left_freq, right_freq)) in istft_results.into_iter().enumerate() {
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
