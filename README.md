# 🎵 Stem Splitter Core

**High-performance, pure-Rust audio stem separation library powered by ONNX Runtime**

[![Crates.io](https://img.shields.io/crates/v/stem-splitter-core.svg)](https://crates.io/crates/stem-splitter-core)
[![License](https://img.shields.io/crates/l/stem-splitter-core.svg)](LICENSE-MIT)

---

## 🎧 Overview

`stem-splitter-core` is a Rust library for splitting audio tracks into isolated stems (vocals, drums, bass, and other instruments) using state-of-the-art AI models. Built entirely in Rust with ONNX Runtime, it provides:

- **No Python dependency** - Pure Rust implementation
- **High-quality separation** - Uses the Hybrid Transformer Demucs (htdemucs) model
- **Automatic model management** - Downloads and caches models with registry support
- **Fast inference** - Optimized ONNX Runtime with GPU acceleration and multi-threading
- **Progress tracking** - Built-in callbacks for download and processing progress
- **Production-ready** - Memory-safe, performant, and battle-tested

Perfect for music production tools, DJ software, karaoke apps, or any application requiring audio source separation.

---

## ✨ Features

- 🎵 **4-Stem Separation** — Isolate vocals, drums, bass, and other instruments
- 🧠 **State-of-the-art AI** — Hybrid Transformer Demucs model (htdemucs)
- 🚀 **GPU Acceleration** — CUDA, CoreML, DirectML, and oneDNN support (auto-detected)
- 📦 **Model Registry** — Built-in model registry with support for multiple models
- 🎚️ **Multiple Formats** — Supports WAV, MP3, FLAC, OGG, and more via Symphonia
- 📊 **Progress Tracking** — Real-time callbacks for download and split progress
- 🔒 **Type-safe** — Strong compile-time guarantees with Rust's type system
- 💾 **Smart Caching** — Models cached in user directories with SHA-256 verification

---

## 🔧 CLI & Distribution

While `stem-splitter-core` is primarily a Rust library, this repository also provides a
first-party CLI (`stem-splitter`) and prebuilt binaries for common platforms.

### CLI

The CLI is built on top of `stem-splitter-core` and exposes the same high-performance
audio stem separation features via the command line.

The CLI source lives in:

src/bin/stem-splitter.rs

### Prebuilt Binaries

Prebuilt binaries are published with each GitHub release:

https://github.com/gentij/stem-splitter-core/releases

These binaries are suitable for:
- Arch Linux (via AUR)
- Debian / Ubuntu (manual install)
- Any glibc-based Linux distribution

### Platform Packages

- macOS: Homebrew
- Arch Linux: AUR (`stem-splitter-bin`)
- Linux (generic): tar.gz binary from GitHub Releases

See the `packaging/` directory for reference packaging files.

---

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
stem-splitter-core = "1.0.0"
```

### System Requirements

- **Rust 1.70+**
- **~200MB disk space** for model storage (first run only)
- **4GB+ RAM** recommended for processing

No external dependencies or Python installation required!

---

## 🚀 Quick Start

### Basic Usage

```rust
use stem_splitter_core::{split_file, SplitOptions};

fn main() -> anyhow::Result<()> {
    // Configure the split operation
    let options = SplitOptions {
        output_dir: "./output".to_string(),
        model_name: "htdemucs_ort_v1".to_string(),
        manifest_url_override: None,
    };

    // Split the audio file
    let result = split_file("song.mp3", options)?;

    // Access the separated stems
    println!("Vocals: {}", result.vocals_path);
    println!("Drums: {}", result.drums_path);
    println!("Bass: {}", result.bass_path);
    println!("Other: {}", result.other_path);

    Ok(())
}
```

Or even simpler with defaults:

```rust
use stem_splitter_core::{split_file, SplitOptions};

fn main() -> anyhow::Result<()> {
    // Use default options (htdemucs_ort_v1 model, current directory)
    let result = split_file("song.mp3", SplitOptions::default())?;
    
    println!("Vocals: {}", result.vocals_path);
    Ok(())
}
```

### With Progress Tracking

```rust
use stem_splitter_core::{split_file, SplitOptions, SplitProgress};

fn main() -> anyhow::Result<()> {
    // Set download progress callback
    stem_splitter_core::set_download_progress_callback(|downloaded, total| {
        let percent = if total > 0 {
            (downloaded as f64 / total as f64 * 100.0) as u64
        } else {
            0
        };
        if total > 0 {
            eprint!("\rDownloading model… {}% ({}/{} bytes)", percent, downloaded, total);
            if downloaded >= total {
                eprintln!();
            }
        }
    });

    // Set split progress callback
    stem_splitter_core::set_split_progress_callback(|progress| {
        match progress {
            SplitProgress::Stage(stage) => {
                eprintln!("> Stage: {}", stage);
            }
            SplitProgress::Writing { stem, percent, .. } => {
                eprintln!("Writing {}: {:.0}%", stem, percent);
            }
            SplitProgress::Finished => {
                eprintln!("Split finished!");
            }
            _ => {}
        }
    });

    let options = SplitOptions {
        output_dir: "./output".to_string(),
        ..Default::default()  // Uses htdemucs_ort_v1 by default
    };

    split_file("song.mp3", options)?;
    Ok(())
}
```

### Pre-loading Models

For applications that need to minimize latency, pre-load the model:

```rust
use stem_splitter_core::prepare_model;

fn main() -> anyhow::Result<()> {
    // Download and load model at startup
    prepare_model("htdemucs_ort_v1", None)?;
    
    // Now splitting will be instant (no download delay)
    // ... use split_file() as normal
    
    Ok(())
}
```

---

## 📖 API Reference

### `split_file(input_path: &str, opts: SplitOptions) -> Result<SplitResult>`

Main function to split an audio file into stems.

**Parameters:**
- `input_path`: Path to the audio file (supports WAV, MP3, FLAC, OGG, etc.)
- `opts`: Configuration options (see `SplitOptions`)

**Returns:**
- `SplitResult` containing paths to the separated stem files

### `SplitOptions`

Configuration struct for the separation process.

```rust
pub struct SplitOptions {
    /// Directory where output stems will be saved
    pub output_dir: String,
    
    /// Name of the model to use (e.g., "htdemucs_ort_v1")
    pub model_name: String,
    
    /// Optional: Override the model manifest URL
    /// (useful for custom models or specific versions)
    pub manifest_url_override: Option<String>,
}
```

**Default values:**
- `output_dir`: `"."`
- `model_name`: `"htdemucs_ort_v1"`
- `manifest_url_override`: `None`

### `SplitResult`

Result struct containing paths to the separated stems.

```rust
pub struct SplitResult {
    pub vocals_path: String,
    pub drums_path: String,
    pub bass_path: String,
    pub other_path: String,
}
```

### `prepare_model(model_name: &str, manifest_url_override: Option<&str>) -> Result<()>`

Pre-loads and caches a model for faster subsequent splits.

**Parameters:**
- `model_name`: Name of the model to prepare
- `manifest_url_override`: Optional URL to override the manifest location

### `ensure_model(model_name: &str, manifest_url_override: Option<&str>) -> Result<ModelHandle>`

Downloads and verifies a model, returning a handle with metadata.

**Parameters:**
- `model_name`: Name of the model to ensure
- `manifest_url_override`: Optional URL to override the manifest location

**Returns:**
- `ModelHandle` containing the manifest and local path to the model

### `set_download_progress_callback(callback: F)`

Set a callback to track model download progress.

```rust
pub fn set_download_progress_callback<F>(callback: F)
where
    F: Fn(u64, u64) + Send + 'static,
```

**Callback parameters:**
- `downloaded`: Bytes downloaded so far
- `total`: Total bytes to download (0 if unknown)

### `set_split_progress_callback(callback: F)`

Set a callback to track split processing progress.

```rust
pub fn set_split_progress_callback<F>(callback: F)
where
    F: Fn(SplitProgress) + Send + 'static,
```

**SplitProgress variants:**
- `Stage(&'static str)`: Current processing stage (e.g., "resolve_model", "read_audio", "infer")
- `Chunks { done, total, percent }`: Progress through audio chunks
- `Writing { stem, done, total, percent }`: Progress writing a specific stem
- `Finished`: Processing complete

---

## 🎯 Supported Audio Formats

The library supports a wide range of audio formats through the [Symphonia](https://github.com/pdeljanov/Symphonia) decoder:

- **WAV** - Uncompressed audio (best quality)
- **MP3** - MPEG Layer 3
- **FLAC** - Free Lossless Audio Codec
- **OGG Vorbis** - Open-source lossy format
- **AAC** - Advanced Audio Coding
- And more...

**Output Format:** All stems are saved as 16-bit PCM WAV files at 44.1kHz stereo.

---

## 🧠 Model Information

### HTDemucs-ORT (htdemucs_ort_v1)

This is the default and currently supported model:

- **Architecture:** Hybrid Transformer Demucs
- **Format:** ONNX Runtime optimized
- **Size:** ~200MB (~209MB to be precise)
- **Quality:** State-of-the-art separation quality
- **Sources:** 4 stems (drums, bass, other, vocals)
- **Sample Rate:** 44.1kHz
- **Window Size:** 343,980 samples (~7.8 seconds)
- **Hop Size:** 171,990 samples (50% overlap)
- **Origin:** Converted from [Meta's Demucs v4](https://github.com/facebookresearch/demucs)

The model is automatically downloaded from [HuggingFace](https://huggingface.co/gentij/htdemucs-ort/resolve/main/manifest.json) on first use and cached locally in your system's cache directory with SHA-256 verification.

### Model Registry

The library includes a built-in model registry (`models/registry.json`) that maps model names to their manifest URLs. This allows users to simply specify `"htdemucs_ort_v1"` without needing to remember or provide the full HuggingFace URL.

### Custom Models

You can use custom models by providing a manifest URL override:

```rust
let options = SplitOptions {
    output_dir: "./output".to_string(),
    model_name: "my_custom_model".to_string(),
    manifest_url_override: Some(
        "https://example.com/path/to/manifest.json".to_string()
    ),
};
```

---

## 🔧 Advanced Usage

### Error Handling

```rust
use stem_splitter_core::{split_file, SplitOptions};

match split_file("song.mp3", SplitOptions::default()) {
    Ok(result) => {
        println!("Success! Vocals: {}", result.vocals_path);
    }
    Err(e) => {
        eprintln!("Error during separation: {}", e);
    }
}
```

### Working with Model Handles

For advanced use cases, you can manually manage models:

```rust
use stem_splitter_core::{ensure_model, ModelHandle};

fn main() -> anyhow::Result<()> {
    // Get a handle to the model
    let handle: ModelHandle = ensure_model("htdemucs_ort_v1", None)?;
    
    // Access model metadata
    println!("Model path: {}", handle.local_path.display());
    println!("Sample rate: {}", handle.manifest.sample_rate);
    println!("Window size: {}", handle.manifest.window);
    println!("Stems: {:?}", handle.manifest.stems);
    
    Ok(())
}
```

---

## 🧪 Development

### Running Examples

The library includes two examples demonstrating key features:

#### `split_one` - Complete stem separation with progress tracking

```bash
# Split an audio file into stems
cargo run --release --example split_one -- input.mp3 ./output

# Usage: split_one <audio_file> [output_dir]
# Default output directory is ./out
```

This example demonstrates:
- Download progress callbacks
- Split progress callbacks (stages, chunks, writing)
- Custom model manifest URLs
- Complete stem separation workflow

#### `ensure_model` - Model download and caching

```bash
# Download and cache a model
cargo run --release --example ensure_model
```

This example demonstrates:
- Model download with progress tracking
- Model metadata inspection
- Model registry usage

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test model_manager

# With output
cargo test -- --nocapture
```

### Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

---

## 🤔 FAQ

**Q: Why is the first run slow?**  
A: The model (~200MB) is downloaded on first use. Subsequent runs are instant.

**Q: Where are models stored?**  
A: Models are cached in your system's standard cache directory with SHA-256 verification for integrity.

**Q: Can I use GPU acceleration?**  
A: Yes! GPU acceleration is enabled by default and works across all platforms:
- **NVIDIA GPUs**: CUDA (Linux, Windows)
- **Apple Silicon**: CoreML (M1/M2/M3/M4 Macs)
- **Windows (any GPU)**: DirectML (NVIDIA, AMD, Intel)
- **Intel**: oneDNN optimizations

The library automatically detects available hardware and uses the best execution provider, falling back to CPU if no GPU is available. GPU mode significantly reduces CPU usage (~5x less) while maintaining similar processing speed.

**Q: What's the quality compared to Python Demucs?**  
A: Identical quality - we use the same model architecture, just optimized for ONNX.

**Q: Can I use my own custom model?**  
A: Yes! Use the `manifest_url_override` option to point to your own model manifest.

**Q: Does it work offline?**  
A: Yes, after the initial model download, everything works offline.

**Q: What sample rates are supported?**  
A: Input audio is automatically resampled to 44.1kHz for processing.

---

## 🗺️ Roadmap

- [x] GPU acceleration (CUDA, CoreML, DirectML, oneDNN)
- [ ] Additional model support (6-stem models with guitar/piano)
- [ ] Real-time processing mode
- [ ] Streaming API support

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Clone the repository
2. Install Rust (1.70+): https://rustup.rs
3. Run `cargo build`
4. Run tests: `cargo test`

---

## 📄 License

Licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

---

## 🙏 Acknowledgments

- **Meta Research** - Original [Demucs](https://github.com/facebookresearch/demucs) model
- **[demucs.onnx](https://github.com/sevagh/demucs.onnx)** - ONNX conversion reference
- **ONNX Runtime** - High-performance inference engine
- **Symphonia** - Pure Rust audio decoding

---

## 📞 Support

- 📧 Issues: [GitHub Issues](https://github.com/gentij/stem-splitter-core/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/gentij/stem-splitter-core/discussions)
- 📚 Documentation: [docs.rs](https://docs.rs/stem-splitter-core)

---

**Made with ❤️ and 🦀 Rust**
