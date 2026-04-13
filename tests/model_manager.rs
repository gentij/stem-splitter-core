use rand::{rngs::StdRng, RngCore, SeedableRng};
use sha2::{Digest, Sha256};
use std::sync::{Mutex, OnceLock};
use tempfile::tempdir;

use httpmock::prelude::*;

use stem_splitter_core::model::model_manager::ensure_model;

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct CacheHomeGuard {
    previous: Option<String>,
}

impl CacheHomeGuard {
    fn set(path: &std::path::Path) -> Self {
        let previous = std::env::var("XDG_CACHE_HOME").ok();
        std::env::set_var("XDG_CACHE_HOME", path);
        Self { previous }
    }
}

impl Drop for CacheHomeGuard {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            std::env::set_var("XDG_CACHE_HOME", previous);
        } else {
            std::env::remove_var("XDG_CACHE_HOME");
        }
    }
}

fn make_fake_model_bytes(len: usize) -> (Vec<u8>, String, u64) {
    let mut data = vec![0u8; len];

    let mut rng = StdRng::seed_from_u64(42);
    rng.fill_bytes(&mut data);

    let mut h = Sha256::new();
    h.update(&data);
    let sha = hex::encode(h.finalize());

    (data, sha, len as u64)
}

fn manifest_json(
    model_name: &str,
    file_name: &str,
    model_url: &str,
    sha256_hex: &str,
    size: u64,
) -> String {
    format!(
        r#"{{
  "name": "{name}",
  "version": "1.0.0",
  "backend": "onnx",
  "sample_rate": 44100,
  "window": 441000,
  "hop": 220500,
  "stems": ["vocals", "drums", "bass", "other"],
  "input_layout": "BCT",
  "output_layout": "BSCT",
  "artifacts": [
    {{
      "file": "{file}",
      "url": "{url}",
      "sha256": "{sha}",
      "size_bytes": {size}
    }}
  ]
}}"#,
        name = model_name,
        file = file_name,
        url = model_url,
        sha = sha256_hex,
        size = size
    )
}

#[test]
fn downloads_and_caches_model_then_reuses_cache() {
    let _lock = env_lock().lock().unwrap();
    let tmp_cache = tempdir().unwrap();
    let _cache_home = CacheHomeGuard::set(tmp_cache.path());

    let unique = tmp_cache
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let (model_bytes, sha_hex, size) = make_fake_model_bytes(256 * 1024);

    let server = MockServer::start();

    let model_name = format!("mdx_4stem_v1_{unique}");
    let file_name = format!("mdx_4stem_v1_{unique}.onnx");
    let model_path = format!("/{file_name}");
    let model_url = format!("{}/{}", server.base_url(), file_name);

    let model_mock = server.mock(|when, then| {
        when.method(GET).path(model_path.as_str());
        then.status(200)
            .header("Content-Length", size.to_string().as_str())
            .body(model_bytes.clone());
    });

    let manifest_body = manifest_json(&model_name, &file_name, &model_url, &sha_hex, size);
    let manifest_path = format!("/{model_name}.json");

    let manifest_mock = server.mock(|when, then| {
        when.method(GET).path(manifest_path.as_str());
        then.status(200)
            .header("Content-Type", "application/json")
            .body(manifest_body.clone());
    });

    let manifest_url = format!("{}{}", server.base_url(), manifest_path);

    let handle = ensure_model("ignored", Some(&manifest_url)).expect("first ensure_model failed");
    assert!(handle.local_path.exists(), "cached model should exist");

    assert!(manifest_mock.hits() >= 1);
    model_mock.assert_hits(1);

    let handle2 = ensure_model("ignored", Some(&manifest_url)).expect("second ensure_model failed");
    assert_eq!(
        handle.local_path, handle2.local_path,
        "cache path should be stable"
    );

    model_mock.assert_hits(1);
}

#[test]
fn checksum_mismatch_returns_error() {
    let _lock = env_lock().lock().unwrap();
    let tmp_cache = tempdir().unwrap();
    let _cache_home = CacheHomeGuard::set(tmp_cache.path());

    let unique = tmp_cache
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let (model_bytes, sha_hex, size) = make_fake_model_bytes(64 * 1024);
    let mut bad_sha = sha_hex.clone();
    let first = &bad_sha[0..1];
    bad_sha.replace_range(0..1, if first == "a" { "b" } else { "a" });

    let server = MockServer::start();

    let model_name = format!("bad_model_{unique}");
    let file_name = format!("bad_{unique}.onnx");
    let model_path = format!("/{file_name}");
    let model_url = format!("{}/{}", server.base_url(), file_name);

    let _model_mock = server.mock(|when, then| {
        when.method(GET).path(model_path.as_str());
        then.status(200)
            .header("Content-Length", size.to_string().as_str())
            .body(model_bytes.clone());
    });

    let manifest_body = manifest_json(&model_name, &file_name, &model_url, &bad_sha, size);
    let manifest_path = format!("/bad_{unique}.json");

    let _manifest_mock = server.mock(|when, then| {
        when.method(GET).path(manifest_path.as_str());
        then.status(200)
            .header("Content-Type", "application/json")
            .body(manifest_body.clone());
    });

    let manifest_url = format!("{}{}", server.base_url(), manifest_path);

    match ensure_model("ignored", Some(&manifest_url)) {
        Ok(_) => panic!("expected checksum error, but got Ok"),
        Err(e) => {
            let msg = e.to_string().to_lowercase();
            assert!(
                msg.contains("checksum"),
                "expected checksum error, got: {msg}"
            );
        }
    }
}
