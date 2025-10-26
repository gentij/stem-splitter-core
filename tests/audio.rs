use std::f32::consts::PI;

use tempfile::tempdir;

use hound;
use stem_splitter_core::core::audio::{read_audio, write_audio};
use stem_splitter_core::AudioData;

fn mono_sine(sample_rate: u32, freq: f32, seconds: f32) -> Vec<f32> {
    let n = (sample_rate as f32 * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * freq * t).sin() * 0.5 // amplitude 0.5 for headroom
        })
        .collect()
}

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn write_then_read_roundtrip_mono_sine() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("out/roundtrip.wav"); // includes parent dir to test mkdirs
    let path_str = path.to_string_lossy().to_string();

    let sr = 44_100;
    let samples = mono_sine(sr, 440.0, 0.25);
    let audio = AudioData {
        samples: samples.clone(),
        sample_rate: sr,
        channels: 1,
    };

    write_audio(&path_str, &audio).expect("write_audio failed");
    let decoded = read_audio(&path).expect("read_audio failed");

    assert_eq!(decoded.sample_rate, sr);
    assert_eq!(decoded.channels, 1);

    assert_eq!(decoded.samples.len(), samples.len());

    let tol = 1e-3;
    let mut mismatches = 0usize;
    for (a, b) in samples.iter().zip(decoded.samples.iter()) {
        if !approx_eq(*a, *b, tol) {
            mismatches += 1;
        }
    }

    let ratio = mismatches as f32 / samples.len() as f32;
    assert!(
        ratio < 0.01,
        "Too many mismatches after roundtrip: {mismatches}/{}",
        samples.len()
    );
}

#[test]
fn read_audio_stereo_interleaved() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("stereo/test.wav");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let sr = 48_000;
    let frames = 1024usize;
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    {
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..frames {
            // scale to i16
            let l = (0.5 * i16::MAX as f32) as i16;
            let r = (-0.5 * i16::MAX as f32) as i16;
            writer.write_sample(l).unwrap();
            writer.write_sample(r).unwrap();
        }
        writer.finalize().unwrap();
    }

    let decoded = read_audio(&path).expect("read_audio failed");

    assert_eq!(decoded.sample_rate, sr);
    assert_eq!(decoded.channels, 2);
    assert_eq!(decoded.samples.len(), frames * 2);

    let tol = 2e-3;
    for i in 0..frames {
        let l = decoded.samples[2 * i];
        let r = decoded.samples[2 * i + 1];
        assert!(approx_eq(l, 0.5, tol), "L[{}] ~= 0.5, got {}", i, l);
        assert!(approx_eq(r, -0.5, tol), "R[{}] ~= -0.5, got {}", i, r);
    }
}

#[test]
fn write_audio_clamps_samples_to_16bit() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("clamp/out.wav");
    let path_str = path.to_string_lossy().to_string();

    let audio = AudioData {
        samples: vec![2.0, -2.0, 0.0],
        sample_rate: 44_100,
        channels: 1,
    };

    write_audio(&path_str, &audio).expect("write_audio failed");

    let mut reader = hound::WavReader::open(&path).unwrap();
    let mut raw: Vec<i16> = Vec::new();
    for s in reader.samples::<i16>() {
        raw.push(s.unwrap());
    }

    assert_eq!(raw.len(), 3);
    assert_eq!(raw[0], i16::MAX, "2.0 should clamp to i16::MAX");
    assert_eq!(raw[1], i16::MIN, "-2.0 should clamp to i16::MIN");
    assert_eq!(raw[2], 0, "0.0 should round to 0");
}

#[test]
fn write_audio_creates_parent_directories() {
    let tmp = tempdir().unwrap();
    let deep = tmp.path().join("a/b/c/out.wav");
    let path_str = deep.to_string_lossy().to_string();

    let audio = AudioData {
        samples: vec![0.0; 64],
        sample_rate: 22_050,
        channels: 1,
    };

    write_audio(&path_str, &audio).expect("write_audio failed");

    assert!(deep.exists(), "output file should exist");
}

#[test]
fn read_audio_nonexistent_file_returns_error() {
    let tmp = tempdir().unwrap();
    let missing = tmp.path().join("nope/does_not_exist.wav");
    let err = read_audio(&missing).unwrap_err();
    let msg = format!("{:#}", err).to_lowercase();
    assert!(
        msg.contains("failed to open audio file") || msg.contains("no such file"),
        "expected an open/read error, got: {msg}"
    );
}
