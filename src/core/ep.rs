#![cfg_attr(feature = "engine-mock", allow(dead_code))]

use crate::error::Result;

use anyhow::anyhow;
use ort::{
    execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
    session::Session,
};
use std::path::Path;

// CUDA: Linux and Windows only
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use ort::execution_providers::CUDAExecutionProvider;
// CoreML: macOS only (Apple Silicon)
#[cfg(all(feature = "coreml", target_os = "macos"))]
use ort::execution_providers::CoreMLExecutionProvider;
// DirectML: Windows only
#[cfg(all(feature = "directml", target_os = "windows"))]
use ort::execution_providers::DirectMLExecutionProvider;
// oneDNN: x86 Linux/Windows only
#[cfg(feature = "onednn")]
use ort::execution_providers::OneDNNExecutionProvider;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpKind {
    Cpu,
    Cuda,
    CoreML,
    DirectML,
    OneDNN,
}

impl EpKind {
    fn label(self) -> &'static str {
        match self {
            EpKind::Cpu => "CPU",
            EpKind::Cuda => "CUDA",
            EpKind::CoreML => "CoreML",
            EpKind::DirectML => "DirectML",
            EpKind::OneDNN => "oneDNN",
        }
    }

    fn env_name(self) -> &'static str {
        match self {
            EpKind::Cpu => "cpu",
            EpKind::Cuda => "cuda",
            EpKind::CoreML => "coreml",
            EpKind::DirectML => "directml",
            EpKind::OneDNN => "onednn",
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

pub(crate) fn create_best_session<FCpu, FEp, FProbe>(
    model_path: &Path,
    num_threads: usize,
    mut build_cpu_session: FCpu,
    mut build_ep_session: FEp,
    mut probe_session: FProbe,
) -> Result<Session>
where
    FCpu: FnMut(&Path, usize) -> Result<Session>,
    FEp: FnMut(&Path, usize, EpKind, ExecutionProviderDispatch) -> Result<Session>,
    FProbe: FnMut(&mut Session) -> Result<()>,
{
    let debug_enabled = is_debug_enabled();
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
        return build_cpu_session(model_path, num_threads);
    }

    if let Some(kind) = request.forced_kind {
        if debug_enabled {
            eprintln!("ℹ️  Forcing execution provider: {}", kind.label());
        }
    }

    let mut providers: Vec<EpCandidate> = Vec::new();
    for kind in request.kinds {
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

        match probe_session(&mut session) {
            Ok(()) => {
                if debug_enabled {
                    eprintln!(
                        "✅ Execution provider selected: {} (attempt #{})",
                        ep_name,
                        idx + 1
                    );
                }
                return Ok(session);
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

    Ok(session)
}

fn is_debug_enabled() -> bool {
    std::env::var("DEBUG_STEMS").is_ok()
}

fn parse_ep_kind(value: &str) -> Option<EpKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(EpKind::Cpu),
        "cuda" => Some(EpKind::Cuda),
        "coreml" => Some(EpKind::CoreML),
        "directml" => Some(EpKind::DirectML),
        "onednn" | "one-dnn" | "one_dnn" | "dnnl" => Some(EpKind::OneDNN),
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
                "Unknown execution provider '{}' in STEMMER_EP_DISABLE (valid: cuda, coreml, directml, onednn)",
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

fn default_ep_order_for_os(os: &str) -> Vec<EpKind> {
    match os {
        "windows" => vec![EpKind::Cuda, EpKind::DirectML, EpKind::OneDNN],
        "macos" => vec![EpKind::CoreML, EpKind::OneDNN],
        "linux" => vec![EpKind::Cuda, EpKind::OneDNN],
        _ => vec![EpKind::OneDNN],
    }
}

fn resolve_ep_request_for_os(
    os: &str,
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
                "Unknown execution provider '{}' in STEMMER_EP_FORCE (valid: cpu, cuda, coreml, directml, onednn)",
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

    let mut kinds = default_ep_order_for_os(os);
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

    resolve_ep_request_for_os(
        std::env::consts::OS,
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
                let ep = CoreMLExecutionProvider::default();
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ep_order_is_platform_specific() {
        assert_eq!(
            default_ep_order_for_os("windows"),
            vec![EpKind::Cuda, EpKind::DirectML, EpKind::OneDNN]
        );
        assert_eq!(
            default_ep_order_for_os("macos"),
            vec![EpKind::CoreML, EpKind::OneDNN]
        );
        assert_eq!(
            default_ep_order_for_os("linux"),
            vec![EpKind::Cuda, EpKind::OneDNN]
        );
    }

    #[test]
    fn force_cpu_overrides_other_flags() {
        let req = resolve_ep_request_for_os("linux", true, Some("cuda"), Some("onednn")).unwrap();
        assert!(req.force_cpu);
        assert!(req.kinds.is_empty());
        assert_eq!(req.forced_kind, None);
    }

    #[test]
    fn force_specific_provider() {
        let req = resolve_ep_request_for_os("linux", false, Some("CUDA"), None).unwrap();
        assert!(!req.force_cpu);
        assert_eq!(req.kinds, vec![EpKind::Cuda]);
        assert_eq!(req.forced_kind, Some(EpKind::Cuda));
    }

    #[test]
    fn disable_list_filters_auto_order() {
        let req =
            resolve_ep_request_for_os("windows", false, None, Some("directml, onednn")).unwrap();
        assert_eq!(req.kinds, vec![EpKind::Cuda]);
        assert_eq!(req.forced_kind, None);
    }

    #[test]
    fn force_and_disable_conflict_errors() {
        let err = resolve_ep_request_for_os("windows", false, Some("directml"), Some("directml"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("STEMMER_EP_FORCE=directml conflicts with STEMMER_EP_DISABLE"));
    }

    #[test]
    fn invalid_values_error() {
        let err_force = resolve_ep_request_for_os("linux", false, Some("invalid"), None)
            .unwrap_err()
            .to_string();
        assert!(err_force.contains("Unknown execution provider 'invalid' in STEMMER_EP_FORCE"));

        let err_disable = resolve_ep_request_for_os("linux", false, None, Some("gpu"))
            .unwrap_err()
            .to_string();
        assert!(err_disable.contains("Unknown execution provider 'gpu' in STEMMER_EP_DISABLE"));
    }

    #[test]
    fn force_value_is_case_and_whitespace_insensitive() {
        let req = resolve_ep_request_for_os("macos", false, Some("  CoReMl  "), None).unwrap();
        assert_eq!(req.forced_kind, Some(EpKind::CoreML));
        assert_eq!(req.kinds, vec![EpKind::CoreML]);
    }

    #[test]
    fn disable_list_deduplicates_and_handles_aliases() {
        let disabled = parse_disabled_ep_list(Some(" one_dnn, onednn, one-dnn , dnnl ")).unwrap();
        assert_eq!(disabled, vec![EpKind::OneDNN]);
    }

    #[test]
    fn empty_force_uses_default_order() {
        let req = resolve_ep_request_for_os("linux", false, Some("   "), None).unwrap();
        assert_eq!(req.kinds, vec![EpKind::Cuda, EpKind::OneDNN]);
        assert_eq!(req.forced_kind, None);
    }

    #[test]
    fn disable_cpu_is_rejected() {
        let err = parse_disabled_ep_list(Some("cpu")).unwrap_err().to_string();
        assert!(err.contains("'cpu' is not valid in STEMMER_EP_DISABLE"));
    }
}
