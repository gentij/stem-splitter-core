# 🎵 Stem Splitter Core

**High-performance, pure-Rust audio stem separation library powered by ONNX Runtime**

[![Crates.io](https://img.shields.io/crates/v/stem-splitter-core.svg)](https://crates.io/crates/stem-splitter-core)
[![License](https://img.shields.io/crates/l/stem-splitter-core.svg)](LICENSE-MIT)

---

## 🎧 Overview

`stem-splitter-core` is a Rust library for splitting audio tracks into isolated stems (vocals, drums, bass, and other instruments) using state-of-the-art AI models. Built entirely in Rust with ONNX Runtime, it provides:

- **No Python dependency** - Pure Rust implementation
- **High-quality separation** - Uses the Hybrid Transformer Demucs (htdemucs) model
- **Automatic model management** - Downloads and caches models from HuggingFace
- **Fast inference** - Optimized ONNX Runtime with multi-threading support
- **Production-ready** - Memory-safe, performant, and battle-tested

Perfect for music production tools, DJ software, karaoke apps, or any application requiring audio source separation.

---

## ✨ Features

- 🎵 **4-Stem Separation** — Isolate vocals, drums, bass, and other instruments
- 🧠 **State-of-the-art AI** — Hybrid Transformer Demucs model (htdemucs)
- 📦 **Auto Model Management** — Automatically downloads and verifies models from HuggingFace
- 🎚️ **Multiple Formats** — Supports WAV, MP3, FLAC, OGG, and more via Symphonia
- 🔒 **Type-safe** — Strong compile-time guarantees with Rust's type system
- 💾 **Smart Caching** — Models cached in user directories, downloaded once

---

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
stem-splitter-core = "0.1"
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
    /// (useful for custom or local models)
    pub manifest_url_override: Option<String>,
}
```

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

### `set_download_progress_callback`

Set a callback to track model download progress.

```rust
pub fn set_download_progress_callback<F>(callback: F)
where
    F: Fn(u64, u64) + Send + Sync + 'static,
```

**Callback parameters:**
- `downloaded`: Bytes downloaded so far
- `total`: Total bytes to download

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

### Default Model: htdemucs_ort_v1

- **Architecture:** Hybrid Transformer Demucs
- **Format:** ONNX Runtime optimized
- **Size:** ~190MB
- **Quality:** State-of-the-art separation quality
- **Sources:** 4 stems (vocals, drums, bass, other)
- **Sample Rate:** 44.1kHz
- **Origin:** Converted from [Meta's Demucs v4](https://github.com/facebookresearch/demucs)

The model is automatically downloaded from HuggingFace on first use and cached locally.

### Custom Models

You can use custom models by providing a manifest URL:

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

## ⚙️ Performance

### Benchmark (Apple M1, 3-minute song)

- **Processing Time:** ~2m 40s
- **Memory Usage:** ~800MB peak
- **Model Load Time:** <5s (after initial download)
- **Chunk Processing:** ~7 chunks @ 343,980 samples each

### Optimization Tips

1. **Use Release Mode:** Always build with `--release` for ~10x speedup
2. **Pre-load Models:** Call `prepare_model()` at startup to avoid download delays
3. **Batch Processing:** Reuse the loaded model for multiple files
4. **Hardware:** More CPU cores = faster processing (auto-detected)

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
        // Handle different error types
        if e.to_string().contains("Model") {
            eprintln!("Model download/load failed");
        } else if e.to_string().contains("audio") {
            eprintln!("Audio file reading failed");
        }
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
    println!("Model path: {}", handle.local_path);
    println!("Sample rate: {}", handle.manifest.sample_rate);
    println!("Window size: {}", handle.manifest.window);
    println!("Stems: {:?}", handle.manifest.stems);
    
    Ok(())
}
```

---

## 🧪 Development

### Running Examples

```bash
# Basic example
cargo run --release --example split_one -- input.mp3 ./output

# With custom model
cargo run --release --example split_one -- song.wav ./stems
```

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
A: The model (~190MB) is downloaded on first use. Subsequent runs are instant.

**Q: Can I use GPU acceleration?**  
A: Currently CPU-only. GPU support via ONNX Runtime execution providers is planned.

**Q: What's the quality compared to Python Demucs?**  
A: Identical quality - we use the same model architecture, just optimized for ONNX.

**Q: Can I separate more than 4 stems?**  
A: The current model supports 4 stems. 6-stem models (adding guitar/piano) can be added.

**Q: Does it work offline?**  
A: Yes, after the initial model download, everything works offline.

**Q: What sample rates are supported?**  
A: Input audio is automatically resampled to 44.1kHz for processing.

---

## 🗺️ Roadmap

- [ ] GPU acceleration (CUDA, Metal, DirectML)
- [ ] 6-stem model support (guitar, piano)
- [ ] Real-time processing mode

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
