# Stem Splitter Core

A Rust library for high-quality, AI-powered audio stem separation.

---

## 🎧 Overview

`stem-splitter-core` provides the core functionality for splitting full audio tracks into individual stems such as vocals, drums, bass, and other instruments. It's designed to serve as the foundational backend for music production tools, remix apps, or DJ software.

Behind the scenes, it uses external machine learning models (like [Demucs](https://github.com/facebookresearch/demucs)) to perform the separation locally on the user's machine.

---

## 🚀 Features

- 🎵 **Audio Stem Separation** — Split full tracks into vocals, drums, bass, and more  
- 🧠 **AI-Powered** — Uses external models like Demucs (via Python) for state-of-the-art quality  
- ⚡ **Fast + Safe** — Built in Rust with strong safety guarantees and performance  
- 🎚️ **Mono & Stereo Input** — Supports mono and stereo WAV/MP3 files  
- 🛠️ **Pluggable Backends** — Trait-based model interface allows future integration of native or other AI inference engines  
- 📂 **Output as WAV** — Results are saved in `.wav` format for easy post-processing  

---

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
stem-splitter-core = "0.1.0"
```

> ⚠️ This crate depends on Python and external AI models. See [Setup](#-setup) for details.

---

## ⚡ Quick Start

```rust
use stem_splitter_core::{split_file, SplitConfig};

let result = split_file("example.mp3", SplitConfig {
    output_dir: "./output".to_string(),
}).expect("Failed to split stems");

println!("Vocals: {} samples", result.vocals.len());
```

---

## 🧰 Setup

To use this crate, you must install:

### 1. ✅ Python 3.8+

Ensure Python is installed and accessible:

```bash
python3 --version
```

---

### 2. ✅ Install Python Dependencies

You need to install the following Python packages:

```bash
pip install demucs torch torchaudio
```

Optionally, you can use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install demucs torch torchaudio
```

---

### 3. ✅ Python Script Setup

By default, this crate uses a Python script named `demucs_runner.py` located at the root of the project.

If you wish to override it with your own script, set the `STEM_SPLITTER_PYTHON_SCRIPT` environment variable to point to your custom script:

```bash
export STEM_SPLITTER_PYTHON_SCRIPT=./scripts/your_custom_script.py
```

The script must:

- Accept `--input` and `--output` arguments
- Use Demucs (or another model) to process the audio file
- Save 4 WAV files: `vocals.wav`, `drums.wav`, `bass.wav`, and `other.wav` in the specified output directory

Basic stub example:

```python
# demucs_runner.py
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

subprocess.run(["demucs", "--two-stems=vocals", args.input, "-o", args.output])
```

---

## 📁 Supported Input Formats

- `.wav`
- `.mp3`

Other formats (like `.flac`, `.ogg`, etc.) may work depending on `symphonia` backend support.

---

## 🧪 Development Status

- ✅ MP3/WAV input decoding
- ✅ Python subprocess integration
- ✅ WAV stem writing
- ✅ Mono/stereo support
- 🛠️ Extensible architecture for custom inference backends

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions, issues, or feature requests, feel free to open an issue or submit a pull request.

---

## 🪪 License

Licensed under either of:

- MIT ([LICENSE-MIT](LICENSE-MIT))
- Apache 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

At your option.
