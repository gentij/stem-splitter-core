use crate::{
    crypto::verify_sha256,
    error::{Result, StemError},
    net::{download_with_progress, http_client},
    paths::models_cache_dir,
    registry::resolve_manifest_url,
};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelManifest {
    pub name: String,
    pub version: String,
    pub backend: String,
    pub sample_rate: u32,
    pub window: usize,
    pub hop: usize,
    pub stems: Vec<String>,
    pub input_layout: String,
    pub output_layout: String,
    pub url: String,
    pub sha256: String,
    pub filesize: u64,
}

pub struct ModelHandle {
    pub manifest: ModelManifest,
    pub local_path: PathBuf,
}

pub fn ensure_model(model_name: &str, manifest_url_override: Option<&str>) -> Result<ModelHandle> {
    let manifest_url = if let Some(url) = manifest_url_override {
        url.to_string()
    } else {
        resolve_manifest_url(model_name)?
    };

    let client = http_client();
    let manifest: ModelManifest = client
        .get(&manifest_url)
        .send()?
        .error_for_status()?
        .json()?;

    let cache_dir = models_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;
    let file_name = format!("{}-{}.onnx", manifest.name, &manifest.sha256[..8]);
    let local_path = cache_dir.join(file_name);

    let need_download = match verify_sha256(&local_path, &manifest.sha256) {
        Ok(true) => false,
        _ => true,
    };

    if need_download {
        download_with_progress(&client, &manifest.url, &local_path)?;
        if !verify_sha256(&local_path, &manifest.sha256)? {
            return Err(StemError::Checksum {
                path: local_path.display().to_string(),
            });
        }
    }

    Ok(ModelHandle {
        manifest,
        local_path,
    })
}
