# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2024-11-29

### ⚡ Performance & GPU Acceleration

This release adds GPU acceleration support and significant performance improvements.

### Added
- **GPU Acceleration**: Cross-platform GPU support via ONNX Runtime execution providers
  - CUDA for NVIDIA GPUs (Linux, Windows)
  - CoreML for Apple Silicon (macOS)
  - DirectML for Windows (NVIDIA, AMD, Intel)
  - oneDNN for Intel optimizations (all platforms)
  - Automatic fallback to CPU if no GPU available
- **Parallel iSTFT Processing**: All 4 stems now processed in parallel using rayon
- **Cached FFT Planner**: FFT planners and Hann windows cached globally for reuse

### Changed
- ~14% faster processing time (1:49 vs 2:07 on test files)
- ~5x reduction in CPU usage when GPU acceleration is active
- Reduced memory allocations in hot paths

### Dependencies
- Added `rayon` for parallel processing

## [1.0.0] - 2024-10-26

### 🎉 Major Release - Complete Architecture Rewrite

This is a **major architectural change** moving from Python-based processing to pure Rust with ONNX Runtime.

### Changed
- **BREAKING**: Removed Python dependency entirely - now uses ONNX Runtime for inference
- **BREAKING**: Now uses converted HTDemucs ONNX model instead of Python-based Demucs
- Pure Rust implementation with no external language dependencies
- Significantly improved performance and reduced system requirements
- Simplified deployment - no need to install Python or Python packages

### Added
- ONNX Runtime integration for high-performance inference
- HTDemucs-ORT model (htdemucs_ort_v1) - converted from Meta's Demucs v4
- Built-in model registry for simplified model management
- Automatic model downloading with SHA-256 verification
- Progress tracking with `set_split_progress_callback()` and `SplitProgress` enum
- Model caching in system cache directories
- `prepare_model()` function for pre-loading models
- Comprehensive API documentation

### Previous Versions

Earlier versions (< 1.0.0) used a Python-based approach and are not compatible with this release.

[Unreleased]: https://github.com/gentij/stem-splitter-core/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/gentij/stem-splitter-core/releases/tag/v1.1.0
[1.0.0]: https://github.com/gentij/stem-splitter-core/releases/tag/v1.0.0

