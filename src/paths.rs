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
