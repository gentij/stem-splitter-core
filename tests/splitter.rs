#![cfg(feature = "engine-mock")]

use httpmock::prelude::*;
use sha2::{Digest, Sha256};
use std::f32::consts::PI;
use std::fs;
use tempfile::tempdir;

use hound;
use stem_splitter_core::core::audio::write_audio;
use stem_splitter_core::core::splitter::split_file;
use stem_splitter_core::{AudioData, SplitOptions};

// Compute hex sha256 for arbitrary bytes
fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn manifest_json(model_url: &str, sha_hex: &str) -> String {
    format!(
        r#"{{
  "name": "mdx_mock",
  "version": "1.0.0",
  "backend": "onnx",
  "sample_rate": 44100,
  "window": 4096,
  "hop": 2048,
  "stems": ["vocals","drums","bass","other"],
  "input_layout": "BCT",
  "output_layout": "BSCT",
  "artifacts": [
    {{
      "file": "mock.onnx",
      "url": "{url}",
      "sha256": "{sha}",
      "size_bytes": 0
    }}
  ]
}}"#,
        url = model_url,
        sha = sha_hex
    )
}

#[test]
fn split_file_produces_four_stems() {
    let tmp = tempdir().unwrap();
    std::env::set_var("XDG_CACHE_HOME", tmp.path());

    let in_wav = tmp.path().join("in.wav");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&out_dir).unwrap();

    let sr = 44_100u32;
    let frames = 8000usize;
    let mut samples = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let t = i as f32 / sr as f32;
        samples.push((2.0 * PI * 440.0 * t).sin() * 0.2);
        samples.push((2.0 * PI * 660.0 * t).sin() * 0.2);
    }
    let audio = AudioData {
        samples,
        sample_rate: sr,
        channels: 2,
    };
    write_audio(in_wav.to_str().unwrap(), &audio).unwrap();

    let server = MockServer::start();
    let model_body = b"this is the mock onnx payload";
    let model_sha = sha256_hex(model_body);

    let _model = server.mock(|when, then| {
        when.method(GET).path("/mock.onnx");
        then.status(200)
            .header("Content-Length", model_body.len().to_string().as_str())
            .body(model_body.as_slice());
    });

    let _manifest = server.mock(|when, then| {
        when.method(GET).path("/m.json");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(manifest_json(
                &format!("{}/mock.onnx", server.base_url()),
                &model_sha,
            ));
    });

    let manifest_url = format!("{}/m.json", server.base_url());

    let opts = SplitOptions {
        model_name: "ignored".into(),
        manifest_url_override: Some(manifest_url),
        output_dir: out_dir.to_string_lossy().into(),
    };

    let res = split_file(in_wav.to_str().unwrap(), opts).expect("split_file failed");

    for p in [
        &res.vocals_path,
        &res.drums_path,
        &res.bass_path,
        &res.other_path,
    ] {
        assert!(std::path::Path::new(p).exists(), "missing stem {p}");
        let r = hound::WavReader::open(p).unwrap();
        assert_eq!(r.spec().channels, 2);
        assert_eq!(r.spec().sample_rate, sr);
        assert!(r.into_samples::<i16>().count() > 0);
    }
}
