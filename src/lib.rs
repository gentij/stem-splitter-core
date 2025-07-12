//! # Stem Splitter Core
//!
//! A Rust library for AI-powered audio stem separation.
//! 
//! This crate provides the core functionality for separating audio tracks
//! into individual stems (vocals, drums, bass, etc.) using machine learning models.
//!
//! ## Example
//!
//! ```rust
//! use stem_splitter_core::StemSplitter;
//!
//! // This is a placeholder example - actual implementation coming soon!
//! let splitter = StemSplitter::new();
//! println!("Stem splitter initialized: {:?}", splitter);
//! ```

/// A simple struct representing our stem splitter
#[derive(Debug)]
pub struct StemSplitter {
    /// Configuration for the stem splitter
    pub config: StemSplitterConfig,
}

/// Configuration for the stem splitter
#[derive(Debug)]
pub struct StemSplitterConfig {
    /// The model to use for stem separation
    pub model_type: String,
    /// Quality setting (0.0 to 1.0)
    pub quality: f32,
}

impl Default for StemSplitterConfig {
    fn default() -> Self {
        Self {
            model_type: "demucs".to_string(),
            quality: 0.8,
        }
    }
}

impl StemSplitter {
    /// Create a new stem splitter with default configuration
    pub fn new() -> Self {
        Self {
            config: StemSplitterConfig::default(),
        }
    }

    /// Create a new stem splitter with custom configuration
    pub fn with_config(config: StemSplitterConfig) -> Self {
        Self { config }
    }

    /// Get the current configuration
    pub fn config(&self) -> &StemSplitterConfig {
        &self.config
    }
}

impl Default for StemSplitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stem_splitter_creation() {
        let splitter = StemSplitter::new();
        assert_eq!(splitter.config.model_type, "demucs");
        assert_eq!(splitter.config.quality, 0.8);
    }

    #[test]
    fn test_stem_splitter_with_config() {
        let config = StemSplitterConfig {
            model_type: "spleeter".to_string(),
            quality: 0.9,
        };
        let splitter = StemSplitter::with_config(config);
        assert_eq!(splitter.config.model_type, "spleeter");
        assert_eq!(splitter.config.quality, 0.9);
    }
} 