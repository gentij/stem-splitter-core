#![cfg_attr(feature = "engine-mock", allow(dead_code, unused_imports))]

use crate::{
    core::{
        dsp::{istft_cac_stereo_parallel, stft_cac_stereo_centered},
        ep,
    },
    error::{Result, StemError},
    io::ep_cache,
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
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};

static SESSION: OnceCell<Mutex<Session>> = OnceCell::new();
static MANIFEST: OnceCell<ModelManifest> = OnceCell::new();
static ORT_INIT: OnceCell<()> = OnceCell::new();
#[cfg(not(feature = "engine-mock"))]
static ENGINE_CONTEXT: OnceCell<EngineContext> = OnceCell::new();
#[cfg(not(feature = "engine-mock"))]
static RUNTIME_EP_FALLBACK_USED: AtomicBool = AtomicBool::new(false);

const DEMUCS_T: usize = 343_980;
const DEMUCS_F: usize = 2048;
const DEMUCS_FRAMES: usize = 336;
const DEMUCS_NFFT: usize = 4096;
const DEMUCS_HOP: usize = 1024;

#[cfg(not(feature = "engine-mock"))]
struct EngineContext {
    model_path: PathBuf,
    num_threads: usize,
    selected_kind: ep::EpKind,
}

#[cfg(not(feature = "engine-mock"))]
struct DemucsRawOutput {
    num_sources: usize,
    data_time: Vec<f32>,
    data_freq: Vec<f32>,
    time_max: f32,
    freq_max: f32,
}

#[cfg(not(feature = "engine-mock"))]
#[derive(Clone, Copy)]
struct OrtThreading {
    intra_threads: usize,
    inter_threads: usize,
    parallel_execution: bool,
}

#[cfg(not(feature = "engine-mock"))]
fn parse_env_usize(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    let parsed = raw.parse::<usize>().ok()?;
    if parsed == 0 {
        None
    } else {
        Some(parsed)
    }
}

#[cfg(not(feature = "engine-mock"))]
fn parse_env_bool(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(not(feature = "engine-mock"))]
fn apply_thread_overrides(mut cfg: OrtThreading) -> OrtThreading {
    if let Some(intra) = parse_env_usize("STEMMER_ORT_INTRA_THREADS") {
        cfg.intra_threads = intra;
    }
    if let Some(inter) = parse_env_usize("STEMMER_ORT_INTER_THREADS") {
        cfg.inter_threads = inter;
    }
    if let Some(parallel) = parse_env_bool("STEMMER_ORT_PARALLEL") {
        cfg.parallel_execution = parallel;
    }
    cfg
}

#[cfg(not(feature = "engine-mock"))]
fn cpu_threading(num_threads: usize) -> OrtThreading {
    let base = OrtThreading {
        intra_threads: num_threads.max(1),
        inter_threads: 1,
        parallel_execution: false,
    };
    apply_thread_overrides(base)
}

#[cfg(not(feature = "engine-mock"))]
fn ep_threading(kind: ep::EpKind, num_threads: usize) -> OrtThreading {
    let base = match kind {
        ep::EpKind::Cuda | ep::EpKind::CoreML | ep::EpKind::DirectML => OrtThreading {
            intra_threads: num_threads.clamp(1, 4),
            inter_threads: 1,
            parallel_execution: false,
        },
        ep::EpKind::OneDNN | ep::EpKind::Cpu => OrtThreading {
            intra_threads: num_threads.max(1),
            inter_threads: 1,
            parallel_execution: false,
        },
    };
    apply_thread_overrides(base)
}

#[cfg(not(feature = "engine-mock"))]
fn commit_cpu_session(model_path: &std::path::Path, num_threads: usize) -> Result<Session> {
    let threading = cpu_threading(num_threads);

    if std::env::var("DEBUG_STEMS").is_ok() {
        eprintln!(
            "ℹ️  ORT CPU threading: intra={}, inter={}, parallel={}",
            threading.intra_threads, threading.inter_threads, threading.parallel_execution
        );
    }

    Ok(SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threading.intra_threads)?
        .with_inter_threads(threading.inter_threads)?
        .with_parallel_execution(threading.parallel_execution)?
        .commit_from_file(model_path)?)
}

#[cfg(not(feature = "engine-mock"))]
fn commit_ep_session(
    model_path: &std::path::Path,
    num_threads: usize,
    kind: ep::EpKind,
    provider: ort::execution_providers::ExecutionProviderDispatch,
) -> Result<Session> {
    let threading = ep_threading(kind, num_threads);

    if std::env::var("DEBUG_STEMS").is_ok() {
        eprintln!(
            "ℹ️  ORT EP threading: intra={}, inter={}, parallel={}",
            threading.intra_threads, threading.inter_threads, threading.parallel_execution
        );
    }

    let builder = SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threading.intra_threads)?
        .with_inter_threads(threading.inter_threads)?
        .with_parallel_execution(threading.parallel_execution)?
        .with_execution_providers(vec![provider])?;

    Ok(builder.commit_from_file(model_path)?)
}

#[cfg(not(feature = "engine-mock"))]
fn run_demucs_raw_from_inputs(
    session: &mut Session,
    t: usize,
    f_bins: usize,
    frames: usize,
    time_branch: Vec<f32>,
    spec_branch: Vec<f32>,
) -> Result<DemucsRawOutput> {
    let time_value: Value = Tensor::from_array((vec![1, 2, t], time_branch))?.into_dyn();
    let spec_value: Value =
        Tensor::from_array((vec![1, 4, f_bins, frames], spec_branch))?.into_dyn();

    let in_time = session
        .inputs()
        .iter()
        .find(|i| i.name() == "input")
        .map(|i| i.name().to_owned())
        .ok_or_else(|| anyhow!("Model missing input 'input'"))?;

    let in_spec = session
        .inputs()
        .iter()
        .find(|i| i.name() == "x")
        .map(|i| i.name().to_owned())
        .ok_or_else(|| anyhow!("Model missing input 'x'"))?;

    let outputs = session.run(vec![(in_time, time_value), (in_spec, spec_value)])?;

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

    let (shape_time, data_time) = out_time.try_extract_tensor::<f32>()?;
    if shape_time.len() != 4
        || shape_time[0] != 1
        || shape_time[2] != 2
        || shape_time[3] != t as i64
    {
        return Err(anyhow!(
            "Unexpected time output shape: {:?}, expected [1, sources, 2, {}]",
            shape_time,
            t
        )
        .into());
    }
    let num_sources = shape_time[1] as usize;

    let (shape_freq, data_freq) = out_freq.try_extract_tensor::<f32>()?;
    if shape_freq.len() != 5
        || shape_freq[0] != 1
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

    let time_max = data_time.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let freq_max = data_freq.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    Ok(DemucsRawOutput {
        num_sources,
        data_time: data_time.to_vec(),
        data_freq: data_freq.to_vec(),
        time_max,
        freq_max,
    })
}

#[cfg(not(feature = "engine-mock"))]
fn run_demucs_raw_with_session(
    session: &mut Session,
    left: &[f32],
    right: &[f32],
) -> Result<DemucsRawOutput> {
    if left.len() != right.len() {
        return Err(anyhow!("L/R length mismatch").into());
    }
    let t = left.len();
    if t != DEMUCS_T {
        return Err(anyhow!("Bad window length {} (expected {})", t, DEMUCS_T).into());
    }

    let mut time_branch = Vec::with_capacity(2 * t);
    time_branch.extend_from_slice(left);
    time_branch.extend_from_slice(right);

    let (spec_branch, f_bins, frames) =
        stft_cac_stereo_centered(left, right, DEMUCS_NFFT, DEMUCS_HOP);
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

    run_demucs_raw_from_inputs(session, t, f_bins, frames, time_branch, spec_branch)
}

#[cfg(not(feature = "engine-mock"))]
pub fn preload(h: &ModelHandle) -> Result<()> {
    ORT_INIT.get_or_try_init::<_, StemError>(|| {
        let _ = ort::init().commit();
        Ok(())
    })?;

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let selected = ep::create_best_session(
        h.local_path.as_path(),
        num_threads,
        commit_cpu_session,
        commit_ep_session,
        |_| Ok(()),
    )?;

    ENGINE_CONTEXT
        .set(EngineContext {
            model_path: h.local_path.clone(),
            num_threads,
            selected_kind: selected.kind,
        })
        .ok();
    RUNTIME_EP_FALLBACK_USED.store(false, Ordering::Relaxed);

    SESSION.set(Mutex::new(selected.session)).ok();
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
const NEAR_SILENT_ERROR_PREFIX: &str = "near-silent execution output";

#[cfg(not(feature = "engine-mock"))]
enum RuntimeFallbackDecision {
    RetryOnCpu,
    ForcedProviderError,
    PropagateOriginal,
}

#[cfg(not(feature = "engine-mock"))]
fn output_is_near_silent(time_max: f32, freq_max: f32) -> bool {
    time_max < 1e-6 && freq_max < 1e-3
}

#[cfg(not(feature = "engine-mock"))]
fn input_is_near_silent(left: &[f32], right: &[f32]) -> bool {
    let left_max = left.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let right_max = right.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    left_max.max(right_max) < 1e-4
}

#[cfg(not(feature = "engine-mock"))]
fn is_forced_non_cpu_ep() -> bool {
    let Ok(value) = std::env::var("STEMMER_EP_FORCE") else {
        return false;
    };

    let v = value.trim().to_ascii_lowercase();
    !v.is_empty() && v != "cpu"
}

#[cfg(not(feature = "engine-mock"))]
fn near_silent_error(message: &str) -> bool {
    message.contains(NEAR_SILENT_ERROR_PREFIX)
}

#[cfg(not(feature = "engine-mock"))]
fn runtime_fallback_decision(
    error_text: &str,
    forced_non_cpu_ep: bool,
    fallback_already_used: bool,
) -> RuntimeFallbackDecision {
    if !near_silent_error(error_text) {
        return RuntimeFallbackDecision::PropagateOriginal;
    }
    if forced_non_cpu_ep {
        return RuntimeFallbackDecision::ForcedProviderError;
    }
    if fallback_already_used {
        return RuntimeFallbackDecision::PropagateOriginal;
    }
    RuntimeFallbackDecision::RetryOnCpu
}

#[cfg(not(feature = "engine-mock"))]
pub fn run_window_demucs(left: &[f32], right: &[f32]) -> Result<Array3<f32>> {
    if left.len() != right.len() {
        return Err(anyhow!("L/R length mismatch").into());
    }
    if left.len() != DEMUCS_T {
        return Err(anyhow!("Bad window length {} (expected {})", left.len(), DEMUCS_T).into());
    }

    let mut session = SESSION
        .get()
        .expect("engine::preload first")
        .lock()
        .expect("session poisoned");

    let debug_enabled = std::env::var("DEBUG_STEMS").is_ok();

    match run_window_demucs_with_session(&mut session, left, right, debug_enabled) {
        Ok(out) => Ok(out),
        Err(e) => {
            let error_text = e.to_string();
            let forced_non_cpu_ep = is_forced_non_cpu_ep();
            let fallback_already_used = RUNTIME_EP_FALLBACK_USED.load(Ordering::SeqCst);

            match runtime_fallback_decision(&error_text, forced_non_cpu_ep, fallback_already_used) {
                RuntimeFallbackDecision::ForcedProviderError => {
                    if debug_enabled {
                        eprintln!(
                            "⚠️  Runtime EP output was near-silent and STEMMER_EP_FORCE is set; refusing CPU fallback"
                        );
                    }
                    return Err(anyhow!(
                        "Forced execution provider produced near-silent runtime output; refusing CPU fallback"
                    )
                    .into());
                }
                RuntimeFallbackDecision::PropagateOriginal => {
                    if near_silent_error(&error_text) && debug_enabled {
                        eprintln!(
                            "⚠️  Runtime EP output remained near-silent after fallback; propagating original error"
                        );
                    }
                    return Err(e);
                }
                RuntimeFallbackDecision::RetryOnCpu => {}
            }

            RUNTIME_EP_FALLBACK_USED.store(true, Ordering::SeqCst);

            let ctx = ENGINE_CONTEXT
                .get()
                .ok_or_else(|| anyhow!("engine context missing for runtime fallback"))?;

            if ctx.selected_kind != ep::EpKind::Cpu {
                if let Err(cache_err) = ep_cache::mark_unhealthy(
                    ctx.selected_kind.env_name(),
                    &ctx.model_path,
                    &error_text,
                ) {
                    if debug_enabled {
                        eprintln!(
                            "⚠️  Failed to persist unhealthy EP cache entry: {}",
                            cache_err
                        );
                    }
                } else if debug_enabled {
                    eprintln!(
                        "ℹ️  Marked {} as unhealthy for this model (cached for 7 days)",
                        ctx.selected_kind.label()
                    );
                }
            }

            if debug_enabled {
                eprintln!(
                    "⚠️  Runtime EP output was near-silent; switching to CPU and retrying this chunk"
                );
            }

            let cpu_session = commit_cpu_session(&ctx.model_path, ctx.num_threads)?;
            *session = cpu_session;

            match run_window_demucs_with_session(&mut session, left, right, debug_enabled) {
                Ok(out) => {
                    if debug_enabled {
                        eprintln!("✅ Runtime fallback succeeded: CPU is now active");
                    }
                    Ok(out)
                }
                Err(retry_error) => {
                    if debug_enabled {
                        eprintln!("❌ Runtime fallback to CPU failed: {}", retry_error);
                    }
                    Err(retry_error)
                }
            }
        }
    }
}

#[cfg(not(feature = "engine-mock"))]
fn run_window_demucs_with_session(
    session: &mut Session,
    left: &[f32],
    right: &[f32],
    debug_enabled: bool,
) -> Result<Array3<f32>> {
    if left.len() != right.len() {
        return Err(anyhow!("L/R length mismatch").into());
    }
    let t = left.len();
    if t != DEMUCS_T {
        return Err(anyhow!("Bad window length {} (expected {})", t, DEMUCS_T).into());
    }

    let raw = run_demucs_raw_with_session(session, left, right)?;
    let num_sources = raw.num_sources;

    // Debug: Check if model outputs are non-zero
    if debug_enabled {
        eprintln!(
            "Model output stats: time_max={:.6}, freq_max={:.6}",
            raw.time_max, raw.freq_max
        );
    }

    if !input_is_near_silent(left, right) && output_is_near_silent(raw.time_max, raw.freq_max) {
        return Err(anyhow!(
            "{} (time_max={:.3e}, freq_max={:.3e})",
            NEAR_SILENT_ERROR_PREFIX,
            raw.time_max,
            raw.freq_max
        )
        .into());
    }

    let source_specs: Vec<&[f32]> = (0..num_sources)
        .map(|src| {
            let src_freq_offset = src * 4 * DEMUCS_F * DEMUCS_FRAMES;
            &raw.data_freq[src_freq_offset..src_freq_offset + 4 * DEMUCS_F * DEMUCS_FRAMES]
        })
        .collect();

    let istft_results = istft_cac_stereo_parallel(
        &source_specs,
        DEMUCS_F,
        DEMUCS_FRAMES,
        DEMUCS_NFFT,
        DEMUCS_HOP,
        t,
    );

    // Debug: Check iSTFT results
    if debug_enabled {
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
        let left_time = &raw.data_time[src_time_offset..src_time_offset + t];
        let right_time = &raw.data_time[src_time_offset + t..src_time_offset + 2 * t];

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

#[cfg(not(feature = "engine-mock"))]
#[cfg(test)]
mod runtime_policy_tests {
    use super::*;

    #[test]
    fn fallback_retries_on_cpu_when_near_silent_and_not_forced() {
        let decision = runtime_fallback_decision(
            "near-silent execution output (time_max=0, freq_max=0)",
            false,
            false,
        );
        assert!(matches!(decision, RuntimeFallbackDecision::RetryOnCpu));
    }

    #[test]
    fn fallback_refuses_when_forced_provider() {
        let decision = runtime_fallback_decision(
            "near-silent execution output (time_max=0, freq_max=0)",
            true,
            false,
        );
        assert!(matches!(
            decision,
            RuntimeFallbackDecision::ForcedProviderError
        ));
    }

    #[test]
    fn fallback_does_not_retry_twice() {
        let decision = runtime_fallback_decision(
            "near-silent execution output (time_max=0, freq_max=0)",
            false,
            true,
        );
        assert!(matches!(
            decision,
            RuntimeFallbackDecision::PropagateOriginal
        ));
    }

    #[test]
    fn fallback_ignores_non_silent_errors() {
        let decision = runtime_fallback_decision("Model missing input 'x'", false, false);
        assert!(matches!(
            decision,
            RuntimeFallbackDecision::PropagateOriginal
        ));
    }

    #[test]
    fn near_silent_threshold_checks() {
        assert!(output_is_near_silent(1e-7, 1e-4));
        assert!(!output_is_near_silent(1e-4, 1e-4));
        assert!(!output_is_near_silent(1e-7, 1e-2));
    }

    #[test]
    fn input_silence_threshold_checks() {
        let quiet = vec![0.0f32; 16];
        let loud = vec![5e-4f32; 16];
        assert!(input_is_near_silent(&quiet, &quiet));
        assert!(!input_is_near_silent(&loud, &quiet));
    }
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
