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
    execution_providers::ExecutionProviderDispatch,
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
    value::{Tensor, Value},
};
use std::sync::Mutex;

// CUDA: Linux and Windows only
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use ort::execution_providers::CUDAExecutionProvider;
// CoreML: macOS only (Apple Silicon)
#[cfg(all(feature = "coreml", target_os = "macos"))]
use ort::execution_providers::CoreMLExecutionProvider;
// DirectML: Windows only
#[cfg(all(feature = "directml", target_os = "windows"))]
use ort::execution_providers::DirectMLExecutionProvider;
// oneDNN: All platforms
#[cfg(feature = "onednn")]
use ort::execution_providers::OneDNNExecutionProvider;

static SESSION: OnceCell<Mutex<Session>> = OnceCell::new();
static MANIFEST: OnceCell<ModelManifest> = OnceCell::new();
static ORT_INIT: OnceCell<()> = OnceCell::new();

const DEMUCS_T: usize = 343_980;
const DEMUCS_F: usize = 2048;
const DEMUCS_FRAMES: usize = 336;
const DEMUCS_NFFT: usize = 4096;
const DEMUCS_HOP: usize = 1024;

#[allow(unused_mut)]
fn get_execution_providers() -> Vec<ExecutionProviderDispatch> {
    let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    {
        providers.push(CUDAExecutionProvider::default().build());
    }

    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        // CoreML can sometimes produce silent/zero outputs on certain models
        // Only enable if ENABLE_COREML env var is set
        if std::env::var("ENABLE_COREML").is_ok() {
            if std::env::var("DEBUG_STEMS").is_ok() {
                eprintln!("ℹ️  CoreML enabled via ENABLE_COREML environment variable");
            }
            providers.push(CoreMLExecutionProvider::default().build());
        } else if std::env::var("DEBUG_STEMS").is_ok() {
            eprintln!("ℹ️  CoreML disabled by default (set ENABLE_COREML=1 to enable)");
        }
    }

    #[cfg(all(feature = "directml", target_os = "windows"))]
    {
        // DirectML can fail on some models/drivers (init errors). Keep it opt-in.
        if std::env::var("ENABLE_DIRECTML").is_ok() {
            if std::env::var("DEBUG_STEMS").is_ok() {
                eprintln!("ℹ️  DirectML enabled via ENABLE_DIRECTML environment variable");
            }
            providers.push(DirectMLExecutionProvider::default().build());
        } else if std::env::var("DEBUG_STEMS").is_ok() {
            eprintln!("ℹ️  DirectML disabled by default (set ENABLE_DIRECTML=1 to enable)");
        }
    }

    #[cfg(feature = "onednn")]
    {
        // oneDNN can improve performance on Intel CPUs
        providers.push(OneDNNExecutionProvider::default().build());
    }

    providers
}

#[cfg(not(feature = "engine-mock"))]
fn commit_cpu_session(model_path: &std::path::Path, num_threads: usize) -> Result<Session> {
    Ok(SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_threads)?
        .with_inter_threads(num_threads)?
        .with_parallel_execution(true)?
        .commit_from_file(model_path)?)
}

#[cfg(not(feature = "engine-mock"))]
fn commit_session_sequential_eps(
    model_path: &std::path::Path,
    num_threads: usize,
    providers: Vec<ExecutionProviderDispatch>,
) -> Result<Session> {
    if providers.is_empty() {
        if std::env::var("DEBUG_STEMS").is_ok() {
            eprintln!("ℹ️  Using CPU ({} threads) - no GPU features enabled", num_threads);
        }
        return commit_cpu_session(model_path, num_threads);
    }

    if std::env::var("DEBUG_STEMS").is_ok() {
        eprintln!(
            "ℹ️  Trying execution providers sequentially ({} candidates) with CPU fallback",
            providers.len()
        );
    }

    for (idx, ep) in providers.into_iter().enumerate() {
        let builder_res = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?
            .with_inter_threads(num_threads)?
            .with_execution_providers(vec![ep]);

        let builder = match builder_res {
            Ok(b) => b,
            Err(e) => {
                if std::env::var("DEBUG_STEMS").is_ok() {
                    eprintln!("⚠️  EP builder failed (attempt #{}): {}", idx + 1, e);
                }
                continue;
            }
        };

        match builder.commit_from_file(model_path) {
            Ok(sess) => {
                if std::env::var("DEBUG_STEMS").is_ok() {
                    eprintln!("✅ Execution provider selected (attempt #{}).", idx + 1);
                }
                return Ok(sess);
            }
            Err(e) => {
                if std::env::var("DEBUG_STEMS").is_ok() {
                    eprintln!("⚠️  EP commit failed (attempt #{}): {}", idx + 1, e);
                }
                continue;
            }
        }
    }

    if std::env::var("DEBUG_STEMS").is_ok() {
        eprintln!("⚠️  All EPs failed; falling back to CPU ({} threads)", num_threads);
    }
    commit_cpu_session(model_path, num_threads)
}


#[cfg(not(feature = "engine-mock"))]
pub fn preload(h: &ModelHandle) -> Result<()> {
    ORT_INIT.get_or_try_init::<_, StemError>(|| {
        ort::init().commit().map_err(StemError::from)?;
        Ok(())
    })?;

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Debug / escape hatch: force CPU
    if std::env::var("STEMMER_FORCE_CPU").is_ok() {
        if std::env::var("DEBUG_STEMS").is_ok() {
            eprintln!("ℹ️  STEMMER_FORCE_CPU is set: using CPU only");
        }
        let session = commit_cpu_session(h.local_path.as_path(), num_threads)?;
        SESSION.set(Mutex::new(session)).ok();
        MANIFEST.set(h.manifest.clone()).ok();
        return Ok(());
    }

    // Build provider list (may be empty)
    let providers = get_execution_providers();

    // Optional: print provider list names (for logs)
    #[allow(unused_mut)]
    let mut provider_names: Vec<&str> = Vec::new();
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    provider_names.push("CUDA");
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    provider_names.push("CoreML");
    #[cfg(all(feature = "directml", target_os = "windows"))]
    provider_names.push("DirectML (opt-in)");
    #[cfg(feature = "onednn")]
    provider_names.push("oneDNN");

    if std::env::var("DEBUG_STEMS").is_ok() {
        eprintln!("ℹ️  Configured EP candidates: {:?}", provider_names);
    }

    let session = commit_session_sequential_eps(h.local_path.as_path(), num_threads, providers)?;

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
