#![cfg_attr(feature = "engine-mock", allow(dead_code, unused_imports))]

use crate::{
    core::{
        dsp::{
            istft_cac_stereo_sources_add_into, stft_cac_stereo_centered_into, IstftBatchWorkspace,
        },
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
    value::{TensorRef, Value},
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::Instant,
};

static SESSION: OnceCell<Mutex<Session>> = OnceCell::new();
static MANIFEST: OnceCell<ModelManifest> = OnceCell::new();
static ORT_INIT: OnceCell<()> = OnceCell::new();
#[cfg(not(feature = "engine-mock"))]
static ENGINE_IO_SPEC: OnceCell<EngineIoSpec> = OnceCell::new();
#[cfg(not(feature = "engine-mock"))]
static ENGINE_PERF: OnceCell<EnginePerfConfig> = OnceCell::new();
#[cfg(not(feature = "engine-mock"))]
static INPUT_SCRATCH: OnceCell<Mutex<InferenceScratch>> = OnceCell::new();
#[cfg(not(feature = "engine-mock"))]
static ISTFT_SCRATCH: OnceCell<Mutex<IstftBatchWorkspace>> = OnceCell::new();
#[cfg(not(feature = "engine-mock"))]
static PRELOAD_PROBE_INPUT: OnceCell<(Vec<f32>, Vec<f32>)> = OnceCell::new();
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
#[derive(Clone, Copy)]
struct EngineIoSpec {
    use_positional_inputs: bool,
}

#[cfg(not(feature = "engine-mock"))]
#[derive(Clone, Copy)]
struct EnginePerfConfig {
    enabled: bool,
}

#[cfg(not(feature = "engine-mock"))]
#[derive(Default)]
struct WindowPerf {
    prep_ns: u128,
    stft_ns: u128,
    lock_wait_ns: u128,
    run_ns: u128,
    extract_ns: u128,
    decode_ns: u128,
    istft_ns: u128,
    mix_ns: u128,
    total_ns: u128,
}

#[cfg(not(feature = "engine-mock"))]
#[derive(Default)]
struct InferenceScratch {
    time_branch: Vec<f32>,
    spec_branch: Vec<f32>,
}

#[cfg(not(feature = "engine-mock"))]
impl InferenceScratch {
    fn with_demucs_capacity() -> Self {
        Self {
            time_branch: Vec::with_capacity(2 * DEMUCS_T),
            spec_branch: Vec::with_capacity(4 * DEMUCS_F * DEMUCS_FRAMES),
        }
    }

    fn fill_time_branch(&mut self, left: &[f32], right: &[f32]) {
        self.time_branch.clear();
        self.time_branch.extend_from_slice(left);
        self.time_branch.extend_from_slice(right);
    }
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
fn perf_config() -> &'static EnginePerfConfig {
    ENGINE_PERF.get_or_init(|| EnginePerfConfig {
        enabled: std::env::var("STEMMER_PERF").is_ok(),
    })
}

#[cfg(not(feature = "engine-mock"))]
fn input_scratch() -> &'static Mutex<InferenceScratch> {
    INPUT_SCRATCH.get_or_init(|| Mutex::new(InferenceScratch::with_demucs_capacity()))
}

#[cfg(not(feature = "engine-mock"))]
fn istft_scratch() -> &'static Mutex<IstftBatchWorkspace> {
    ISTFT_SCRATCH.get_or_init(|| Mutex::new(IstftBatchWorkspace::default()))
}

#[cfg(not(feature = "engine-mock"))]
fn io_spec() -> &'static EngineIoSpec {
    ENGINE_IO_SPEC
        .get()
        .expect("engine::preload() must initialize input binding")
}

#[cfg(not(feature = "engine-mock"))]
fn use_positional_inputs(input_names: &[&str]) -> bool {
    matches!(input_names, ["input", "x"])
}

#[cfg(not(feature = "engine-mock"))]
fn inspect_engine_io(session: &Session) -> Result<EngineIoSpec> {
    let input_names: Vec<&str> = session.inputs().iter().map(|input| input.name()).collect();
    let output_names: Vec<&str> = session
        .outputs()
        .iter()
        .map(|output| output.name())
        .collect();

    if !output_names.contains(&"output") {
        return Err(anyhow!("Model missing output 'output' (freq domain)").into());
    }
    if !output_names.contains(&"add_67") {
        return Err(anyhow!("Model missing output 'add_67' (time domain)").into());
    }

    Ok(EngineIoSpec {
        use_positional_inputs: use_positional_inputs(&input_names),
    })
}

#[cfg(not(feature = "engine-mock"))]
fn format_ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

#[cfg(not(feature = "engine-mock"))]
fn log_window_perf(perf: &WindowPerf) {
    eprintln!(
        "⏱️  window total={:.2}ms prep={:.2}ms stft={:.2}ms lock={:.2}ms run={:.2}ms extract={:.2}ms decode={:.2}ms istft={:.2}ms mix={:.2}ms",
        format_ms(perf.total_ns),
        format_ms(perf.prep_ns),
        format_ms(perf.stft_ns),
        format_ms(perf.lock_wait_ns),
        format_ms(perf.run_ns),
        format_ms(perf.extract_ns),
        format_ms(perf.decode_ns),
        format_ms(perf.istft_ns),
        format_ms(perf.mix_ns),
    );
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
        ep::EpKind::Xnnpack => OrtThreading {
            intra_threads: 1,
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

    let mut builder = SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threading.intra_threads)?
        .with_inter_threads(threading.inter_threads)?
        .with_parallel_execution(threading.parallel_execution)?
        .with_execution_providers(vec![provider])?;

    if matches!(kind, ep::EpKind::Xnnpack) {
        builder = builder
            .with_intra_op_spinning(false)?
            .with_inter_op_spinning(false)?;
    }

    Ok(builder.commit_from_file(model_path)?)
}

#[cfg(not(feature = "engine-mock"))]
fn run_demucs_raw_from_inputs(
    session: &mut Session,
    io_spec: &EngineIoSpec,
    t: usize,
    f_bins: usize,
    frames: usize,
    time_branch: &[f32],
    spec_branch: &[f32],
    perf_enabled: bool,
    perf: &mut WindowPerf,
) -> Result<(Value, Value)> {
    let time_value = TensorRef::from_array_view(([1usize, 2, t], time_branch))?;
    let spec_value = TensorRef::from_array_view(([1usize, 4, f_bins, frames], spec_branch))?;

    let run_start = perf_enabled.then(Instant::now);
    let mut outputs = if io_spec.use_positional_inputs {
        session.run(ort::inputs![time_value, spec_value])?
    } else {
        session.run(ort::inputs!["input" => time_value, "x" => spec_value])?
    };
    if let Some(start) = run_start {
        perf.run_ns += start.elapsed().as_nanos();
    }

    let extract_start = perf_enabled.then(Instant::now);
    let out_freq = outputs
        .remove("output")
        .ok_or_else(|| anyhow!("Model did not return 'output' (freq domain)"))?;
    let out_time = outputs
        .remove("add_67")
        .ok_or_else(|| anyhow!("Model did not return 'add_67' (time domain)"))?;
    if let Some(start) = extract_start {
        perf.extract_ns += start.elapsed().as_nanos();
    }

    Ok((out_time, out_freq))
}

#[cfg(not(feature = "engine-mock"))]
fn decode_demucs_outputs(
    out_time: Value,
    out_freq: Value,
    t: usize,
    f_bins: usize,
    frames: usize,
    perf_enabled: bool,
    perf: &mut WindowPerf,
) -> Result<DemucsRawOutput> {
    let decode_start = perf_enabled.then(Instant::now);

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

    let raw = DemucsRawOutput {
        num_sources,
        data_time: data_time.to_vec(),
        data_freq: data_freq.to_vec(),
        time_max,
        freq_max,
    };

    if let Some(start) = decode_start {
        perf.decode_ns += start.elapsed().as_nanos();
    }

    Ok(raw)
}

#[cfg(not(feature = "engine-mock"))]
fn prepare_demucs_inputs(
    left: &[f32],
    right: &[f32],
    scratch: &mut InferenceScratch,
    perf_enabled: bool,
    perf: &mut WindowPerf,
) -> Result<(usize, usize, usize)> {
    if left.len() != right.len() {
        return Err(anyhow!("L/R length mismatch").into());
    }
    let t = left.len();
    if t != DEMUCS_T {
        return Err(anyhow!("Bad window length {} (expected {})", t, DEMUCS_T).into());
    }

    let prep_start = perf_enabled.then(Instant::now);
    scratch.fill_time_branch(left, right);

    let stft_start = perf_enabled.then(Instant::now);
    let (f_bins, frames) = stft_cac_stereo_centered_into(
        left,
        right,
        DEMUCS_NFFT,
        DEMUCS_HOP,
        &mut scratch.spec_branch,
    );
    if let Some(start) = stft_start {
        perf.stft_ns += start.elapsed().as_nanos();
    }
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

    if let Some(start) = prep_start {
        perf.prep_ns += start.elapsed().as_nanos();
    }

    Ok((t, f_bins, frames))
}

#[cfg(not(feature = "engine-mock"))]
fn run_demucs_raw_with_session(
    session: &mut Session,
    io_spec: &EngineIoSpec,
    scratch: &mut InferenceScratch,
    left: &[f32],
    right: &[f32],
    perf_enabled: bool,
    perf: &mut WindowPerf,
) -> Result<DemucsRawOutput> {
    let (t, f_bins, frames) = prepare_demucs_inputs(left, right, scratch, perf_enabled, perf)?;
    let (out_time, out_freq) = run_demucs_raw_from_inputs(
        session,
        io_spec,
        t,
        f_bins,
        frames,
        &scratch.time_branch,
        &scratch.spec_branch,
        perf_enabled,
        perf,
    )?;
    decode_demucs_outputs(out_time, out_freq, t, f_bins, frames, perf_enabled, perf)
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
        probe_session_health,
    )?;

    let selected_io = inspect_engine_io(&selected.session)?;
    ENGINE_IO_SPEC.set(selected_io).ok();
    ENGINE_PERF.set(*perf_config()).ok();
    INPUT_SCRATCH
        .set(Mutex::new(InferenceScratch::with_demucs_capacity()))
        .ok();
    ISTFT_SCRATCH
        .set(Mutex::new(IstftBatchWorkspace::default()))
        .ok();

    if std::env::var("DEBUG_STEMS").is_ok() {
        eprintln!(
            "ℹ️  Engine input binding: {}",
            if selected_io.use_positional_inputs {
                "positional"
            } else {
                "named"
            }
        );
    }

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
fn build_preload_probe_input() -> (Vec<f32>, Vec<f32>) {
    use std::f32::consts::TAU;

    let sample_rate = 44_100.0f32;
    let mut left = Vec::with_capacity(DEMUCS_T);
    let mut right = Vec::with_capacity(DEMUCS_T);

    for i in 0..DEMUCS_T {
        let t = i as f32 / sample_rate;
        left.push(0.22 * (TAU * 220.0 * t).sin() + 0.11 * (TAU * 660.0 * t).sin());
        right.push(0.20 * (TAU * 330.0 * t).sin() + 0.09 * (TAU * 880.0 * t).cos());
    }

    (left, right)
}

#[cfg(not(feature = "engine-mock"))]
fn preload_probe_input() -> &'static (Vec<f32>, Vec<f32>) {
    PRELOAD_PROBE_INPUT.get_or_init(build_preload_probe_input)
}

#[cfg(not(feature = "engine-mock"))]
fn ensure_output_is_not_near_silent(
    left: &[f32],
    right: &[f32],
    raw: &DemucsRawOutput,
) -> Result<()> {
    if !input_is_near_silent(left, right) && output_is_near_silent(raw.time_max, raw.freq_max) {
        return Err(anyhow!(
            "{} (time_max={:.3e}, freq_max={:.3e})",
            NEAR_SILENT_ERROR_PREFIX,
            raw.time_max,
            raw.freq_max
        )
        .into());
    }

    Ok(())
}

#[cfg(not(feature = "engine-mock"))]
fn probe_session_health(session: &mut Session) -> Result<()> {
    let (left, right) = preload_probe_input();
    let io_spec = inspect_engine_io(session)?;
    let mut scratch = InferenceScratch::with_demucs_capacity();
    let mut perf = WindowPerf::default();
    let raw = run_demucs_raw_with_session(
        session,
        &io_spec,
        &mut scratch,
        left,
        right,
        false,
        &mut perf,
    )?;
    ensure_output_is_not_near_silent(left, right, &raw)
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

    let debug_enabled = std::env::var("DEBUG_STEMS").is_ok();
    let perf_enabled = perf_config().enabled;

    match run_window_demucs_once(left, right, debug_enabled, perf_enabled) {
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
            let mut session = SESSION
                .get()
                .expect("engine::preload first")
                .lock()
                .expect("session poisoned");
            *session = cpu_session;
            drop(session);

            match run_window_demucs_once(left, right, debug_enabled, perf_enabled) {
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
fn run_window_demucs_once(
    left: &[f32],
    right: &[f32],
    debug_enabled: bool,
    perf_enabled: bool,
) -> Result<Array3<f32>> {
    let total_start = perf_enabled.then(Instant::now);
    let mut perf = WindowPerf::default();

    let raw = {
        let mut scratch = input_scratch().lock().expect("input scratch poisoned");
        let (t, f_bins, frames) =
            prepare_demucs_inputs(left, right, &mut scratch, perf_enabled, &mut perf)?;

        let lock_start = perf_enabled.then(Instant::now);
        let mut session = SESSION
            .get()
            .expect("engine::preload first")
            .lock()
            .expect("session poisoned");
        if let Some(start) = lock_start {
            perf.lock_wait_ns += start.elapsed().as_nanos();
        }

        let (out_time, out_freq) = run_demucs_raw_from_inputs(
            &mut session,
            io_spec(),
            t,
            f_bins,
            frames,
            &scratch.time_branch,
            &scratch.spec_branch,
            perf_enabled,
            &mut perf,
        )?;
        drop(session);
        drop(scratch);

        decode_demucs_outputs(
            out_time,
            out_freq,
            t,
            f_bins,
            frames,
            perf_enabled,
            &mut perf,
        )?
    };

    let out = postprocess_demucs_output(raw, left, right, debug_enabled, perf_enabled, &mut perf)?;

    if let Some(start) = total_start {
        perf.total_ns = start.elapsed().as_nanos();
        log_window_perf(&perf);
    }

    Ok(out)
}

#[cfg(not(feature = "engine-mock"))]
fn postprocess_demucs_output(
    mut raw: DemucsRawOutput,
    left: &[f32],
    right: &[f32],
    debug_enabled: bool,
    perf_enabled: bool,
    perf: &mut WindowPerf,
) -> Result<Array3<f32>> {
    let t = left.len();
    let num_sources = raw.num_sources;

    if debug_enabled {
        eprintln!(
            "Model output stats: time_max={:.6}, freq_max={:.6}",
            raw.time_max, raw.freq_max
        );
    }

    ensure_output_is_not_near_silent(left, right, &raw)?;

    let source_specs: Vec<&[f32]> = (0..num_sources)
        .map(|src| {
            let src_freq_offset = src * 4 * DEMUCS_F * DEMUCS_FRAMES;
            &raw.data_freq[src_freq_offset..src_freq_offset + 4 * DEMUCS_F * DEMUCS_FRAMES]
        })
        .collect();

    let istft_start = perf_enabled.then(Instant::now);
    {
        let mut istft_ws = istft_scratch().lock().expect("iSTFT scratch poisoned");
        istft_cac_stereo_sources_add_into(
            &source_specs,
            DEMUCS_F,
            DEMUCS_FRAMES,
            DEMUCS_NFFT,
            DEMUCS_HOP,
            t,
            &mut istft_ws,
            &mut raw.data_time,
        );
    }
    if let Some(start) = istft_start {
        perf.istft_ns += start.elapsed().as_nanos();
    }

    if debug_enabled {
        for src_idx in 0..num_sources {
            let src_time_offset = src_idx * 2 * t;
            let left_mix = &raw.data_time[src_time_offset..src_time_offset + t];
            let right_mix = &raw.data_time[src_time_offset + t..src_time_offset + 2 * t];
            let left_max = left_mix.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let right_max = right_mix.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "Combined output [source {}]: left_max={:.6}, right_max={:.6}",
                src_idx, left_max, right_max
            );
        }
    }

    let mix_start = perf_enabled.then(Instant::now);

    if let Some(start) = mix_start {
        perf.mix_ns += start.elapsed().as_nanos();
    }

    Ok(ndarray::Array3::from_shape_vec(
        (num_sources, 2, t),
        raw.data_time,
    )?)
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

    #[test]
    fn preload_probe_input_is_loud_enough_for_health_checks() {
        let (left, right) = build_preload_probe_input();
        assert_eq!(left.len(), DEMUCS_T);
        assert_eq!(right.len(), DEMUCS_T);
        assert!(!input_is_near_silent(&left, &right));
    }

    #[test]
    fn positional_binding_detection_requires_expected_input_order() {
        assert!(use_positional_inputs(&["input", "x"]));
        assert!(!use_positional_inputs(&["x", "input"]));
        assert!(!use_positional_inputs(&["input"]));
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
