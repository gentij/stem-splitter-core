#![cfg_attr(feature = "engine-mock", allow(dead_code))]

use crate::{
    error::Result,
    io::{ep_cache, paths},
};

use anyhow::anyhow;
use ort::{
    execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
    session::Session,
};
use std::{num::NonZeroUsize, path::Path};

// CUDA: Linux and Windows only
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use ort::execution_providers::CUDAExecutionProvider;
// CoreML: macOS only (Apple Silicon)
#[cfg(all(feature = "coreml", target_os = "macos"))]
use ort::execution_providers::coreml::{
    ComputeUnits as CoreMLComputeUnits, ModelFormat as CoreMLModelFormat,
    SpecializationStrategy as CoreMLSpecializationStrategy,
};
#[cfg(all(feature = "coreml", target_os = "macos"))]
use ort::execution_providers::CoreMLExecutionProvider;
// DirectML: Windows only
#[cfg(all(feature = "directml", target_os = "windows"))]
use ort::execution_providers::DirectMLExecutionProvider;
// oneDNN: x86 Linux/Windows only
#[cfg(feature = "onednn")]
use ort::execution_providers::OneDNNExecutionProvider;
// XNNPACK: ARM/x86 CPU acceleration
#[cfg(feature = "xnnpack")]
use ort::execution_providers::XNNPACKExecutionProvider;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpKind {
    Cpu,
    Cuda,
    CoreML,
    DirectML,
    OneDNN,
    Xnnpack,
}

impl EpKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            EpKind::Cpu => "CPU",
            EpKind::Cuda => "CUDA",
            EpKind::CoreML => "CoreML",
            EpKind::DirectML => "DirectML",
            EpKind::OneDNN => "oneDNN",
            EpKind::Xnnpack => "XNNPACK",
        }
    }

    pub(crate) fn env_name(self) -> &'static str {
        match self {
            EpKind::Cpu => "cpu",
            EpKind::Cuda => "cuda",
            EpKind::CoreML => "coreml",
            EpKind::DirectML => "directml",
            EpKind::OneDNN => "onednn",
            EpKind::Xnnpack => "xnnpack",
        }
    }
}

#[derive(Debug)]
struct EpRequest {
    kinds: Vec<EpKind>,
    forced_kind: Option<EpKind>,
    force_cpu: bool,
}

#[derive(Debug)]
struct EpCandidate {
    kind: EpKind,
    dispatch: ExecutionProviderDispatch,
}

impl EpCandidate {
    fn name(&self) -> &'static str {
        self.kind.label()
    }
}

pub(crate) struct SelectedSession {
    pub(crate) session: Session,
    pub(crate) kind: EpKind,
}

pub(crate) fn create_best_session<FCpu, FEp, FProbe>(
    model_path: &Path,
    num_threads: usize,
    mut build_cpu_session: FCpu,
    mut build_ep_session: FEp,
    mut probe_session: FProbe,
) -> Result<SelectedSession>
where
    FCpu: FnMut(&Path, usize) -> Result<Session>,
    FEp: FnMut(&Path, usize, EpKind, ExecutionProviderDispatch) -> Result<Session>,
    FProbe: FnMut(&mut Session) -> Result<()>,
{
    let debug_enabled = is_debug_enabled();
    ep_cache::maybe_reset_from_env()?;
    let cache_bypass = ep_cache::cache_bypass_enabled();
    let request = resolve_ep_request_from_env()?;

    if request.force_cpu {
        if debug_enabled {
            eprintln!(
                "ℹ️  CPU mode forced by environment (STEMMER_FORCE_CPU or STEMMER_EP_FORCE=cpu)"
            );
        }
        if debug_enabled {
            eprintln!("✅ Execution provider selected: CPU");
        }
        return Ok(SelectedSession {
            session: build_cpu_session(model_path, num_threads)?,
            kind: EpKind::Cpu,
        });
    }

    if let Some(kind) = request.forced_kind {
        if debug_enabled {
            eprintln!("ℹ️  Forcing execution provider: {}", kind.label());
        }
    }

    let mut providers: Vec<EpCandidate> = Vec::new();
    for kind in request.kinds {
        let cached_reason = ep_cache::is_unhealthy(kind.env_name(), model_path)?;
        if should_skip_due_to_cache(
            kind,
            request.forced_kind,
            cache_bypass,
            cached_reason.as_deref(),
        ) {
            if debug_enabled {
                eprintln!(
                    "ℹ️  Skipping {} from EP health cache: {} (set STEMMER_EP_CACHE_BYPASS=1 to retry)",
                    kind.label(),
                    cached_reason.unwrap_or_default()
                );
            }
            continue;
        }

        match try_build_execution_provider(kind) {
            Ok(dispatch) => providers.push(EpCandidate { kind, dispatch }),
            Err(reason) => {
                if request.forced_kind == Some(kind) {
                    return Err(anyhow!(
                        "Failed to activate forced execution provider '{}': {}",
                        kind.env_name(),
                        reason
                    )
                    .into());
                }

                if debug_enabled {
                    eprintln!("ℹ️  Skipping {}: {}", kind.label(), reason);
                }
            }
        }
    }

    if debug_enabled {
        let provider_names: Vec<&str> = providers.iter().map(EpCandidate::name).collect();
        eprintln!("ℹ️  Configured EP candidates: {:?}", provider_names);
    }

    if debug_enabled && !providers.is_empty() {
        eprintln!(
            "ℹ️  Trying execution providers sequentially ({} candidates) with CPU fallback",
            providers.len()
        );
    }

    for (idx, candidate) in providers.into_iter().enumerate() {
        let ep_name = candidate.name();
        let mut session =
            match build_ep_session(model_path, num_threads, candidate.kind, candidate.dispatch) {
                Ok(session) => session,
                Err(e) => {
                    if request.forced_kind == Some(candidate.kind) {
                        return Err(anyhow!(
                            "Failed to create forced execution provider '{}': {}",
                            candidate.kind.env_name(),
                            e
                        )
                        .into());
                    }
                    if debug_enabled {
                        eprintln!(
                            "⚠️  EP commit failed for {} (attempt #{}): {}",
                            ep_name,
                            idx + 1,
                            e
                        );
                    }
                    continue;
                }
            };

        let recently_healthy =
            ep_cache::is_recently_healthy(candidate.kind.env_name(), model_path)?;
        if recently_healthy {
            if debug_enabled {
                eprintln!(
                    "✅ Execution provider selected: {} (attempt #{}, cached healthy probe)",
                    ep_name,
                    idx + 1
                );
            }
            return Ok(SelectedSession {
                session,
                kind: candidate.kind,
            });
        }

        match probe_session(&mut session) {
            Ok(()) => {
                if let Err(cache_err) =
                    ep_cache::mark_healthy(candidate.kind.env_name(), model_path)
                {
                    if debug_enabled {
                        eprintln!(
                            "⚠️  Failed to persist healthy EP probe entry for {}: {}",
                            ep_name, cache_err
                        );
                    }
                }

                if debug_enabled {
                    eprintln!(
                        "✅ Execution provider selected: {} (attempt #{})",
                        ep_name,
                        idx + 1
                    );
                }
                return Ok(SelectedSession {
                    session,
                    kind: candidate.kind,
                });
            }
            Err(e) => {
                if request.forced_kind == Some(candidate.kind) {
                    return Err(anyhow!(
                        "Forced execution provider '{}' failed health check: {}",
                        candidate.kind.env_name(),
                        e
                    )
                    .into());
                }

                if let Err(cache_err) =
                    ep_cache::mark_unhealthy(candidate.kind.env_name(), model_path, &e.to_string())
                {
                    if debug_enabled {
                        eprintln!(
                            "⚠️  Failed to persist unhealthy EP cache entry for {}: {}",
                            ep_name, cache_err
                        );
                    }
                } else if debug_enabled {
                    eprintln!(
                        "ℹ️  Marked {} as unhealthy for this model (cached for 7 days)",
                        ep_name
                    );
                }

                if debug_enabled {
                    eprintln!(
                        "⚠️  EP rejected for {} (attempt #{}): {}",
                        ep_name,
                        idx + 1,
                        e
                    );
                }
            }
        }
    }

    if debug_enabled {
        eprintln!(
            "⚠️  All EPs failed or were rejected; falling back to CPU ({} threads)",
            num_threads
        );
    }

    let session = build_cpu_session(model_path, num_threads)?;

    if debug_enabled {
        eprintln!("✅ Execution provider selected: CPU");
    }

    Ok(SelectedSession {
        session,
        kind: EpKind::Cpu,
    })
}

fn is_debug_enabled() -> bool {
    std::env::var("DEBUG_STEMS").is_ok()
}

fn parse_env_usize(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    let parsed = raw.parse::<usize>().ok()?;
    (parsed > 0).then_some(parsed)
}

fn parse_env_bool(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn coreml_compute_units() -> CoreMLComputeUnits {
    match std::env::var("STEMMER_COREML_UNITS")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("all") => CoreMLComputeUnits::All,
        Some("ane") | Some("cpuandneuralengine") | Some("cpu_and_neural_engine") => {
            CoreMLComputeUnits::CPUAndNeuralEngine
        }
        Some("gpu") | Some("cpuandgpu") | Some("cpu_and_gpu") => CoreMLComputeUnits::CPUAndGPU,
        Some("cpu") | Some("cpuonly") | Some("cpu_only") => CoreMLComputeUnits::CPUOnly,
        _ => CoreMLComputeUnits::CPUAndGPU,
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn coreml_model_format() -> CoreMLModelFormat {
    match std::env::var("STEMMER_COREML_MODEL_FORMAT")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("neuralnetwork") | Some("neural_network") | Some("nn") => {
            CoreMLModelFormat::NeuralNetwork
        }
        _ => CoreMLModelFormat::MLProgram,
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn coreml_specialization_strategy() -> CoreMLSpecializationStrategy {
    match std::env::var("STEMMER_COREML_SPECIALIZATION")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("fastprediction") | Some("fast_prediction") | Some("fast") => {
            CoreMLSpecializationStrategy::FastPrediction
        }
        _ => CoreMLSpecializationStrategy::Default,
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn coreml_static_input_shapes() -> bool {
    parse_env_bool("STEMMER_COREML_STATIC_INPUTS").unwrap_or(true)
}

fn xnnpack_threads() -> NonZeroUsize {
    let configured = parse_env_usize("STEMMER_ORT_INTRA_THREADS").or_else(|| {
        std::thread::available_parallelism()
            .ok()
            .map(|threads| threads.get())
    });

    NonZeroUsize::new(configured.unwrap_or(4)).expect("XNNPACK thread count must be non-zero")
}

fn should_skip_due_to_cache(
    kind: EpKind,
    forced_kind: Option<EpKind>,
    cache_bypass: bool,
    cached_reason: Option<&str>,
) -> bool {
    forced_kind != Some(kind) && !cache_bypass && cached_reason.is_some()
}

fn parse_ep_kind(value: &str) -> Option<EpKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(EpKind::Cpu),
        "cuda" => Some(EpKind::Cuda),
        "coreml" => Some(EpKind::CoreML),
        "directml" => Some(EpKind::DirectML),
        "onednn" | "one-dnn" | "one_dnn" | "dnnl" => Some(EpKind::OneDNN),
        "xnnpack" | "xnn" => Some(EpKind::Xnnpack),
        _ => None,
    }
}

fn parse_disabled_ep_list(raw: Option<&str>) -> Result<Vec<EpKind>> {
    let mut disabled = Vec::new();

    let Some(raw) = raw else {
        return Ok(disabled);
    };

    for token in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let kind = parse_ep_kind(token).ok_or_else(|| {
            anyhow!(
                "Unknown execution provider '{}' in STEMMER_EP_DISABLE (valid: cuda, coreml, directml, onednn, xnnpack)",
                token
            )
        })?;

        if kind == EpKind::Cpu {
            return Err(anyhow!("'cpu' is not valid in STEMMER_EP_DISABLE").into());
        }

        if !disabled.contains(&kind) {
            disabled.push(kind);
        }
    }

    Ok(disabled)
}

fn default_ep_order_for_target(os: &str, arch: &str) -> Vec<EpKind> {
    match (os, arch) {
        ("windows", _) => vec![
            EpKind::Cuda,
            EpKind::DirectML,
            EpKind::OneDNN,
            EpKind::Xnnpack,
        ],
        ("macos", "aarch64") => vec![EpKind::CoreML, EpKind::Xnnpack],
        ("macos", _) => vec![EpKind::Xnnpack],
        ("linux", "aarch64") => vec![EpKind::Cuda, EpKind::Xnnpack],
        ("linux", _) => vec![EpKind::Cuda, EpKind::OneDNN, EpKind::Xnnpack],
        _ => vec![EpKind::Xnnpack, EpKind::OneDNN],
    }
}

fn resolve_ep_request_for_target(
    os: &str,
    arch: &str,
    force_cpu: bool,
    forced_ep: Option<&str>,
    disabled_raw: Option<&str>,
) -> Result<EpRequest> {
    let disabled = parse_disabled_ep_list(disabled_raw)?;

    if force_cpu {
        return Ok(EpRequest {
            kinds: Vec::new(),
            forced_kind: None,
            force_cpu: true,
        });
    }

    if let Some(raw_forced) = forced_ep.map(str::trim).filter(|s| !s.is_empty()) {
        let forced_kind = parse_ep_kind(raw_forced).ok_or_else(|| {
            anyhow!(
                "Unknown execution provider '{}' in STEMMER_EP_FORCE (valid: cpu, cuda, coreml, directml, onednn, xnnpack)",
                raw_forced
            )
        })?;

        if forced_kind == EpKind::Cpu {
            return Ok(EpRequest {
                kinds: Vec::new(),
                forced_kind: None,
                force_cpu: true,
            });
        }

        if disabled.contains(&forced_kind) {
            return Err(anyhow!(
                "STEMMER_EP_FORCE={} conflicts with STEMMER_EP_DISABLE",
                forced_kind.env_name()
            )
            .into());
        }

        return Ok(EpRequest {
            kinds: vec![forced_kind],
            forced_kind: Some(forced_kind),
            force_cpu: false,
        });
    }

    let mut kinds = default_ep_order_for_target(os, arch);
    kinds.retain(|kind| !disabled.contains(kind));

    Ok(EpRequest {
        kinds,
        forced_kind: None,
        force_cpu: false,
    })
}

fn resolve_ep_request_from_env() -> Result<EpRequest> {
    let force_cpu = std::env::var_os("STEMMER_FORCE_CPU").is_some();
    let forced_ep = std::env::var("STEMMER_EP_FORCE").ok();
    let disabled_raw = std::env::var("STEMMER_EP_DISABLE").ok();

    resolve_ep_request_for_target(
        std::env::consts::OS,
        std::env::consts::ARCH,
        force_cpu,
        forced_ep.as_deref(),
        disabled_raw.as_deref(),
    )
}

fn check_provider_is_usable<E: ExecutionProvider>(provider: &E) -> std::result::Result<(), String> {
    if !provider.supported_by_platform() {
        return Err("unsupported on this platform".to_string());
    }

    match provider.is_available() {
        Ok(true) => Ok(()),
        Ok(false) => Err("not available in this ONNX Runtime build".to_string()),
        Err(e) => Err(format!("availability check failed: {e}")),
    }
}

fn try_build_execution_provider(
    kind: EpKind,
) -> std::result::Result<ExecutionProviderDispatch, String> {
    match kind {
        EpKind::Cpu => Err("CPU does not require an execution provider registration".to_string()),
        EpKind::Cuda => {
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            {
                let ep = CUDAExecutionProvider::default();
                check_provider_is_usable(&ep)?;
                return Ok(ep.build());
            }
            #[cfg(all(feature = "cuda", not(any(target_os = "linux", target_os = "windows"))))]
            {
                return Err("CUDA is only supported on Linux and Windows targets".to_string());
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err("Cargo feature `cuda` is not enabled".to_string());
            }
        }
        EpKind::CoreML => {
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                let mut ep = CoreMLExecutionProvider::default()
                    .with_model_format(coreml_model_format())
                    .with_compute_units(coreml_compute_units())
                    .with_static_input_shapes(coreml_static_input_shapes())
                    .with_specialization_strategy(coreml_specialization_strategy());

                if let Ok(cache_dir) = paths::coreml_cache_dir() {
                    ep = ep.with_model_cache_dir(cache_dir.to_string_lossy().into_owned());
                }

                if is_debug_enabled() {
                    ep = ep.with_profile_compute_plan(true);
                }

                check_provider_is_usable(&ep)?;
                return Ok(ep.build());
            }
            #[cfg(all(feature = "coreml", not(target_os = "macos")))]
            {
                return Err("CoreML is only supported on macOS targets".to_string());
            }
            #[cfg(not(feature = "coreml"))]
            {
                return Err("Cargo feature `coreml` is not enabled".to_string());
            }
        }
        EpKind::DirectML => {
            #[cfg(all(feature = "directml", target_os = "windows"))]
            {
                let ep = DirectMLExecutionProvider::default();
                check_provider_is_usable(&ep)?;
                return Ok(ep.build());
            }
            #[cfg(all(feature = "directml", not(target_os = "windows")))]
            {
                return Err("DirectML is only supported on Windows targets".to_string());
            }
            #[cfg(not(feature = "directml"))]
            {
                return Err("Cargo feature `directml` is not enabled".to_string());
            }
        }
        EpKind::OneDNN => {
            #[cfg(feature = "onednn")]
            {
                let ep = OneDNNExecutionProvider::default();
                check_provider_is_usable(&ep)?;
                return Ok(ep.build());
            }
            #[cfg(not(feature = "onednn"))]
            {
                return Err("Cargo feature `onednn` is not enabled".to_string());
            }
        }
        EpKind::Xnnpack => {
            #[cfg(feature = "xnnpack")]
            {
                let ep = XNNPACKExecutionProvider::default()
                    .with_intra_op_num_threads(xnnpack_threads());
                check_provider_is_usable(&ep)?;
                return Ok(ep.build());
            }
            #[cfg(not(feature = "xnnpack"))]
            {
                return Err("Cargo feature `xnnpack` is not enabled".to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ep_order_is_platform_specific() {
        assert_eq!(
            default_ep_order_for_target("windows", "x86_64"),
            vec![
                EpKind::Cuda,
                EpKind::DirectML,
                EpKind::OneDNN,
                EpKind::Xnnpack
            ]
        );
        assert_eq!(
            default_ep_order_for_target("macos", "aarch64"),
            vec![EpKind::CoreML, EpKind::Xnnpack]
        );
        assert_eq!(
            default_ep_order_for_target("macos", "x86_64"),
            vec![EpKind::Xnnpack]
        );
        assert_eq!(
            default_ep_order_for_target("linux", "x86_64"),
            vec![EpKind::Cuda, EpKind::OneDNN, EpKind::Xnnpack]
        );
        assert_eq!(
            default_ep_order_for_target("linux", "aarch64"),
            vec![EpKind::Cuda, EpKind::Xnnpack]
        );
    }

    #[test]
    fn force_cpu_overrides_other_flags() {
        let req =
            resolve_ep_request_for_target("linux", "x86_64", true, Some("cuda"), Some("onednn"))
                .unwrap();
        assert!(req.force_cpu);
        assert!(req.kinds.is_empty());
        assert_eq!(req.forced_kind, None);
    }

    #[test]
    fn force_specific_provider() {
        let req =
            resolve_ep_request_for_target("linux", "x86_64", false, Some("CUDA"), None).unwrap();
        assert!(!req.force_cpu);
        assert_eq!(req.kinds, vec![EpKind::Cuda]);
        assert_eq!(req.forced_kind, Some(EpKind::Cuda));
    }

    #[test]
    fn disable_list_filters_auto_order() {
        let req = resolve_ep_request_for_target(
            "windows",
            "x86_64",
            false,
            None,
            Some("directml, onednn"),
        )
        .unwrap();
        assert_eq!(req.kinds, vec![EpKind::Cuda, EpKind::Xnnpack]);
        assert_eq!(req.forced_kind, None);
    }

    #[test]
    fn force_and_disable_conflict_errors() {
        let err = resolve_ep_request_for_target(
            "windows",
            "x86_64",
            false,
            Some("directml"),
            Some("directml"),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("STEMMER_EP_FORCE=directml conflicts with STEMMER_EP_DISABLE"));
    }

    #[test]
    fn invalid_values_error() {
        let err_force =
            resolve_ep_request_for_target("linux", "x86_64", false, Some("invalid"), None)
                .unwrap_err()
                .to_string();
        assert!(err_force.contains("Unknown execution provider 'invalid' in STEMMER_EP_FORCE"));

        let err_disable =
            resolve_ep_request_for_target("linux", "x86_64", false, None, Some("gpu"))
                .unwrap_err()
                .to_string();
        assert!(err_disable.contains("Unknown execution provider 'gpu' in STEMMER_EP_DISABLE"));
    }

    #[test]
    fn force_value_is_case_and_whitespace_insensitive() {
        let req =
            resolve_ep_request_for_target("macos", "aarch64", false, Some("  CoReMl  "), None)
                .unwrap();
        assert_eq!(req.forced_kind, Some(EpKind::CoreML));
        assert_eq!(req.kinds, vec![EpKind::CoreML]);
    }

    #[test]
    fn disable_list_deduplicates_and_handles_aliases() {
        let disabled = parse_disabled_ep_list(Some(" one_dnn, onednn, one-dnn , dnnl ")).unwrap();
        assert_eq!(disabled, vec![EpKind::OneDNN]);
    }

    #[test]
    fn xnnpack_aliases_are_supported() {
        let disabled = parse_disabled_ep_list(Some("xnn, xnnpack")).unwrap();
        assert_eq!(disabled, vec![EpKind::Xnnpack]);
    }

    #[test]
    fn empty_force_uses_default_order() {
        let req =
            resolve_ep_request_for_target("linux", "x86_64", false, Some("   "), None).unwrap();
        assert_eq!(
            req.kinds,
            vec![EpKind::Cuda, EpKind::OneDNN, EpKind::Xnnpack]
        );
        assert_eq!(req.forced_kind, None);
    }

    #[test]
    fn disable_cpu_is_rejected() {
        let err = parse_disabled_ep_list(Some("cpu")).unwrap_err().to_string();
        assert!(err.contains("'cpu' is not valid in STEMMER_EP_DISABLE"));
    }

    #[test]
    fn cache_skip_rules_respect_force_and_bypass() {
        assert!(should_skip_due_to_cache(
            EpKind::CoreML,
            None,
            false,
            Some("near-silent runtime output")
        ));
        assert!(!should_skip_due_to_cache(
            EpKind::CoreML,
            Some(EpKind::CoreML),
            false,
            Some("near-silent runtime output")
        ));
        assert!(!should_skip_due_to_cache(
            EpKind::CoreML,
            None,
            true,
            Some("near-silent runtime output")
        ));
        assert!(!should_skip_due_to_cache(EpKind::CoreML, None, false, None));
    }
}
