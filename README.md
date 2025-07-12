# Stem Splitter Core

A Rust library for AI-powered audio stem separation.

## Overview

This crate provides the core functionality for separating audio tracks into individual stems (vocals, drums, bass, etc.) using machine learning models. It's designed to be the foundation for audio stem separation applications.

## Features

- 🎵 **Audio Stem Separation**: Split audio tracks into individual components
- 🤖 **AI-Powered**: Uses machine learning models for high-quality separation
- ⚡ **Fast**: Built with Rust for performance
- 🔧 **Configurable**: Flexible configuration options for different use cases

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
stem-splitter-core = "0.1.0"
```

## Quick Start

```rust
use stem_splitter_core::{StemSplitter, StemSplitterConfig};

// Create a stem splitter with default configuration
let splitter = StemSplitter::new();

// Or with custom configuration
let config = StemSplitterConfig {
    model_type: "demucs".to_string(),
    quality: 0.9,
};
let splitter = StemSplitter::with_config(config);

println!("Stem splitter ready: {:?}", splitter);
```

## Development Status

🚧 **This is an early development version.** The actual stem separation functionality is still being implemented. Currently, this crate provides the basic structure and API that will be expanded upon.

## Roadmap

- [ ] Audio file loading and processing
- [ ] Integration with AI models (Demucs, Spleeter)
- [ ] Multiple output formats support
- [ ] Real-time processing capabilities
- [ ] Advanced configuration options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
