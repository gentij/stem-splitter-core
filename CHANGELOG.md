# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/gentij/stem-splitter-core/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/gentij/stem-splitter-core/releases/tag/v1.0.0

