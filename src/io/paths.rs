use crate::error::{Result, StemError};
use directories::ProjectDirs;
use std::path::PathBuf;

pub fn models_cache_dir() -> Result<PathBuf> {
    let proj = ProjectDirs::from("dev", "StemSplitter", "stem-splitter-core")
        .ok_or(StemError::CacheDirUnavailable)?;
    let mut p = PathBuf::from(proj.cache_dir());
    p.push("models");
    Ok(p)
}

pub fn ep_cache_file() -> Result<PathBuf> {
    let proj = ProjectDirs::from("dev", "StemSplitter", "stem-splitter-core")
        .ok_or(StemError::CacheDirUnavailable)?;
    let mut p = PathBuf::from(proj.cache_dir());
    p.push("ep_health_v1.json");
    Ok(p)
}

pub fn ep_probe_cache_file() -> Result<PathBuf> {
    let proj = ProjectDirs::from("dev", "StemSplitter", "stem-splitter-core")
        .ok_or(StemError::CacheDirUnavailable)?;
    let mut p = PathBuf::from(proj.cache_dir());
    p.push("ep_probe_success_v1.json");
    Ok(p)
}

pub fn coreml_cache_dir() -> Result<PathBuf> {
    let proj = ProjectDirs::from("dev", "StemSplitter", "stem-splitter-core")
        .ok_or(StemError::CacheDirUnavailable)?;
    let mut p = PathBuf::from(proj.cache_dir());
    p.push("coreml");
    Ok(p)
}
