// tests/model_manager.rs
use rand::{rngs::StdRng, RngCore, SeedableRng};
use sha2::{Digest, Sha256};

use httpmock::prelude::*;

use stem_splitter_core::ensure_model;

/// Build a random "model" file in-memory and return (bytes, sha256_hex, size).
fn make_fake_model_bytes(len: usize) -> (Vec<u8>, String, u64) {
    let mut data = vec![0u8; len];
    // deterministic random so the test is reproducible
    let mut rng = StdRng::seed_from_u64(42);
    rng.fill_bytes(&mut data);

    let mut h = Sha256::new();
    h.update(&data);
    let sha = hex::encode(h.finalize());

    (data, sha, len as u64)
}

/// Construct a manifest JSON string for the given URL/hash/size.
fn manifest_json(model_url: &str, sha256_hex: &str, size: u64) -> String {
    format!(
        r#"{{
  "name": "mdx_4stem_v1",
  "version": "1.0.0",
  "backend": "onnx",
  "sample_rate": 44100,
  "window": 441000,
  "hop": 220500,
  "stems": ["vocals", "drums", "bass", "other"],
  "input_layout": "BCT",
  "output_layout": "BSCT",
  "url": "{url}",
  "sha256": "{sha}",
  "filesize": {size}
}}"#,
        url = model_url,
        sha = sha256_hex,
        size = size
    )
}

#[test]
fn downloads_and_caches_model_then_reuses_cache() {
    // Arrange: fake model bytes + hash
    let (model_bytes, sha_hex, size) = make_fake_model_bytes(256 * 1024); // 256 KiB

    // Start mock server
    let server = MockServer::start();

    // Route for the model file (.onnx)
    let model_mock = server.mock(|when, then| {
        when.method(GET).path("/mdx_4stem_v1.onnx");
        then.status(200)
            .header("Content-Length", size.to_string().as_str())
            .body(model_bytes.clone()); // serve bytes
    });

    // Manifest pointing at the model URL
    let manifest_body = manifest_json(
        &format!("{}/mdx_4stem_v1.onnx", server.base_url()),
        &sha_hex,
        size,
    );

    // Route for the manifest JSON
    let manifest_mock = server.mock(|when, then| {
        when.method(GET).path("/mdx_4stem_v1.json");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(manifest_body.clone());
    });

    let manifest_url = format!("{}/mdx_4stem_v1.json", server.base_url());

    // Act 1: first call should download (hit both manifest + model)
    let handle = ensure_model("ignored", Some(&manifest_url)).expect("first ensure_model failed");
    assert!(handle.local_path.exists(), "cached model should exist");

    // Assert:
    // - Manifest fetched at least once
    assert!(manifest_mock.hits() >= 1);
    // - Model fetched exactly once
    model_mock.assert_hits(1);

    // Act 2: second call should use cache (no extra model GET)
    let handle2 = ensure_model("ignored", Some(&manifest_url)).expect("second ensure_model failed");
    assert_eq!(
        handle.local_path, handle2.local_path,
        "cache path should be stable"
    );

    // Manifest may be fetched again (that’s fine), but model must not
    model_mock.assert_hits(1); // still exactly one hit total
}

#[test]
fn checksum_mismatch_returns_error() {
    // Arrange: create model bytes and WRONG sha (flip one nibble)
    let (model_bytes, sha_hex, size) = make_fake_model_bytes(64 * 1024);
    let mut bad_sha = sha_hex.clone();
    let first = &bad_sha[0..1];
    bad_sha.replace_range(0..1, if first == "a" { "b" } else { "a" });

    let server = MockServer::start();

    // Serve the model
    let _model_mock = server.mock(|when, then| {
        when.method(GET).path("/bad.onnx");
        then.status(200)
            .header("Content-Length", size.to_string().as_str())
            .body(model_bytes.clone());
    });

    // Manifest with BAD sha256
    let manifest_body = manifest_json(&format!("{}/bad.onnx", server.base_url()), &bad_sha, size);

    // Serve manifest
    let _manifest_mock = server.mock(|when, then| {
        when.method(GET).path("/bad.json");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(manifest_body.clone());
    });

    let manifest_url = format!("{}/bad.json", server.base_url());

    // Act: ensure_model should error due to checksum mismatch.
    // Avoid unwrap_err() (which requires Debug on the Ok type).
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
