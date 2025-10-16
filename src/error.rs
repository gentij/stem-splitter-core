use thiserror::Error;

/// Central error type for the stem-splitter-core crate.
#[derive(Debug, Error)]
pub enum StemError {
    // Generic fallback (wraps anyhow)
    #[error("{0}")]
    Anyhow(#[from] anyhow::Error),

    // Domain-specific variants
    #[error("Registry error: {0}")]
    Registry(String),

    #[error("Checksum mismatch for {path}")]
    Checksum { path: String },

    #[error("Cache dir not available")]
    CacheDirUnavailable,
}

// --- Implement From conversions for common errors ---
impl From<std::io::Error> for StemError {
    fn from(e: std::io::Error) -> Self {
        StemError::Anyhow(e.into())
    }
}

impl From<serde_json::Error> for StemError {
    fn from(e: serde_json::Error) -> Self {
        StemError::Anyhow(e.into())
    }
}

impl From<reqwest::Error> for StemError {
    fn from(e: reqwest::Error) -> Self {
        StemError::Anyhow(e.into())
    }
}

impl From<hex::FromHexError> for StemError {
    fn from(e: hex::FromHexError) -> Self {
        StemError::Anyhow(e.into())
    }
}

pub type Result<T> = std::result::Result<T, StemError>;
