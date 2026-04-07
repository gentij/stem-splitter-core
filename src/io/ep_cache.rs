#![cfg_attr(feature = "engine-mock", allow(dead_code))]

use crate::{error::Result, io::paths::ep_cache_file};

use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const CACHE_SCHEMA_VERSION: u32 = 1;
const UNHEALTHY_TTL_SECS: u64 = 7 * 24 * 60 * 60;

#[derive(Debug, Clone)]
struct EpCacheKey {
    provider: String,
    model_id: String,
    os: String,
    arch: String,
    ort_api: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EpHealthEntry {
    provider: String,
    model_id: String,
    os: String,
    arch: String,
    ort_api: u32,
    reason: String,
    updated_at_unix: u64,
    fail_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EpHealthCacheFile {
    version: u32,
    entries: Vec<EpHealthEntry>,
}

impl Default for EpHealthCacheFile {
    fn default() -> Self {
        Self {
            version: CACHE_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }
}

pub(crate) fn cache_bypass_enabled() -> bool {
    env_flag_enabled("STEMMER_EP_CACHE_BYPASS")
}

pub(crate) fn maybe_reset_from_env() -> Result<()> {
    let cache_path = ep_cache_file()?;
    maybe_reset_cache_file(&cache_path, env_flag_enabled("STEMMER_EP_CACHE_RESET"))
}

pub(crate) fn is_unhealthy(provider: &str, model_path: &Path) -> Result<Option<String>> {
    let cache_path = ep_cache_file()?;
    let key = build_key(provider, model_path);
    is_unhealthy_in_file(&cache_path, &key, cache_bypass_enabled())
}

pub(crate) fn mark_unhealthy(provider: &str, model_path: &Path, reason: &str) -> Result<()> {
    let cache_path = ep_cache_file()?;
    let key = build_key(provider, model_path);
    mark_unhealthy_in_file(&cache_path, &key, reason)
}

fn env_flag_enabled(name: &str) -> bool {
    let Some(raw) = std::env::var_os(name) else {
        return false;
    };

    let value = raw.to_string_lossy().trim().to_ascii_lowercase();
    if value.is_empty() {
        return true;
    }

    !matches!(value.as_str(), "0" | "false" | "no" | "off")
}

fn build_key(provider: &str, model_path: &Path) -> EpCacheKey {
    let model_id = model_path
        .file_name()
        .and_then(|s| s.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| model_path.to_string_lossy().into_owned());

    EpCacheKey {
        provider: provider.to_ascii_lowercase(),
        model_id,
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        ort_api: ort::MINOR_VERSION,
    }
}

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_secs()
}

fn load_cache_file(path: &Path) -> Result<EpHealthCacheFile> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(EpHealthCacheFile::default())
        }
        Err(e) => return Err(e.into()),
    };

    let parsed = serde_json::from_str::<EpHealthCacheFile>(&raw).unwrap_or_default();
    if parsed.version != CACHE_SCHEMA_VERSION {
        return Ok(EpHealthCacheFile::default());
    }

    Ok(parsed)
}

fn save_cache_file(path: &Path, cache: &EpHealthCacheFile) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(cache)?;
    fs::write(path, raw)?;
    Ok(())
}

fn entry_matches_key(entry: &EpHealthEntry, key: &EpCacheKey) -> bool {
    entry.provider == key.provider
        && entry.model_id == key.model_id
        && entry.os == key.os
        && entry.arch == key.arch
        && entry.ort_api == key.ort_api
}

fn prune_expired_entries(cache: &mut EpHealthCacheFile, now: u64) -> bool {
    let before = cache.entries.len();
    cache
        .entries
        .retain(|entry| now.saturating_sub(entry.updated_at_unix) <= UNHEALTHY_TTL_SECS);
    cache.entries.len() != before
}

fn maybe_reset_cache_file(path: &Path, reset_enabled: bool) -> Result<()> {
    if !reset_enabled {
        return Ok(());
    }

    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn is_unhealthy_in_file(
    path: &Path,
    key: &EpCacheKey,
    bypass_enabled: bool,
) -> Result<Option<String>> {
    if bypass_enabled {
        return Ok(None);
    }

    let mut cache = load_cache_file(path)?;
    let now = current_unix_secs();
    let changed = prune_expired_entries(&mut cache, now);

    let reason = cache
        .entries
        .iter()
        .find(|entry| entry_matches_key(entry, key))
        .map(|entry| format!("{} ({} failures)", entry.reason, entry.fail_count));

    if changed {
        save_cache_file(path, &cache)?;
    }

    Ok(reason)
}

fn mark_unhealthy_in_file(path: &Path, key: &EpCacheKey, reason: &str) -> Result<()> {
    let mut cache = load_cache_file(path)?;
    let now = current_unix_secs();
    prune_expired_entries(&mut cache, now);

    if let Some(entry) = cache
        .entries
        .iter_mut()
        .find(|entry| entry_matches_key(entry, key))
    {
        entry.reason = reason.to_string();
        entry.updated_at_unix = now;
        entry.fail_count = entry.fail_count.saturating_add(1);
    } else {
        cache.entries.push(EpHealthEntry {
            provider: key.provider.clone(),
            model_id: key.model_id.clone(),
            os: key.os.clone(),
            arch: key.arch.clone(),
            ort_api: key.ort_api,
            reason: reason.to_string(),
            updated_at_unix: now,
            fail_count: 1,
        });
    }

    save_cache_file(path, &cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_roundtrip_marks_and_reads_unhealthy() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_path = tmp.path().join("ep_cache.json");
        let model_path = tmp.path().join("model.onnx");
        let key = build_key("coreml", &model_path);

        mark_unhealthy_in_file(&cache_path, &key, "near-silent runtime output").unwrap();
        let reason = is_unhealthy_in_file(&cache_path, &key, false).unwrap();

        assert!(reason.is_some());
        let text = reason.unwrap();
        assert!(text.contains("near-silent runtime output"));
        assert!(text.contains("1 failures"));
    }

    #[test]
    fn bypass_returns_none_even_when_cached() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_path = tmp.path().join("ep_cache.json");
        let model_path = tmp.path().join("model.onnx");
        let key = build_key("coreml", &model_path);

        mark_unhealthy_in_file(&cache_path, &key, "bad output").unwrap();
        let reason = is_unhealthy_in_file(&cache_path, &key, true).unwrap();

        assert!(reason.is_none());
    }

    #[test]
    fn expired_entries_are_pruned() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_path = tmp.path().join("ep_cache.json");
        let model_path = tmp.path().join("model.onnx");
        let key = build_key("coreml", &model_path);

        let mut cache = EpHealthCacheFile::default();
        cache.entries.push(EpHealthEntry {
            provider: key.provider.clone(),
            model_id: key.model_id.clone(),
            os: key.os.clone(),
            arch: key.arch.clone(),
            ort_api: key.ort_api,
            reason: "old failure".to_string(),
            updated_at_unix: current_unix_secs().saturating_sub(UNHEALTHY_TTL_SECS + 10),
            fail_count: 1,
        });
        save_cache_file(&cache_path, &cache).unwrap();

        let reason = is_unhealthy_in_file(&cache_path, &key, false).unwrap();
        assert!(reason.is_none());

        let reloaded = load_cache_file(&cache_path).unwrap();
        assert!(reloaded.entries.is_empty());
    }

    #[test]
    fn reset_removes_cache_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_path = tmp.path().join("ep_cache.json");
        let model_path = tmp.path().join("model.onnx");
        let key = build_key("coreml", &model_path);

        mark_unhealthy_in_file(&cache_path, &key, "bad output").unwrap();
        assert!(cache_path.exists());

        maybe_reset_cache_file(&cache_path, true).unwrap();
        assert!(!cache_path.exists());
    }
}
