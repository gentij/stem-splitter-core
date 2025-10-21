use crate::{
    error::{Result, StemError},
    io::{
        crypto::verify_sha256,
        net::{download_with_progress, http_client},
        paths::models_cache_dir,
    },
    model::registry::resolve_manifest_url,
    types::ModelManifest,
};

use std::{fs, path::PathBuf};

pub struct ModelHandle {
    pub manifest: ModelManifest,
    pub local_path: PathBuf,
}

pub fn ensure_model(model_name: &str, manifest_url_override: Option<&str>) -> Result<ModelHandle> {
    let manifest_url = manifest_url_override
        .map(|s| s.to_string())
        .unwrap_or_else(|| resolve_manifest_url(model_name).expect("resolve_manifest_url failed"));

    let client = http_client();
    let manifest: ModelManifest = client
        .get(&manifest_url)
        .send()?
        .error_for_status()?
        .json()?;

    let a = manifest
        .resolve_primary_artifact()
        .map_err(|msg| StemError::Manifest(msg))?;

    let cache_dir = models_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;
    let ext = a
        .file
        .rsplit('.')
        .next()
        .map(|s| format!(".{s}"))
        .unwrap_or_default();
    let file_name = format!("{}-{}{}", manifest.name, &a.sha256[..8], ext);
    let local_path = cache_dir.join(file_name);

    let need_download = !matches!(verify_sha256(&local_path, &a.sha256), Ok(true));
    if need_download {
        download_with_progress(&client, &a.url, &local_path)?;
        if !verify_sha256(&local_path, &a.sha256)? {
            return Err(StemError::Checksum {
                path: local_path.display().to_string(),
            });
        }
        if a.size_bytes > 0 {
            let size = fs::metadata(&local_path).map(|m| m.len()).unwrap_or(0);
            if size != a.size_bytes {
                eprintln!(
                    "warn: size mismatch for {}, expected {}, got {}",
                    local_path.display(),
                    a.size_bytes,
                    size
                );
            }
        }
    }

    Ok(ModelHandle {
        manifest,
        local_path,
    })
}
