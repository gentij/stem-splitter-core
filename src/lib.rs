mod error;
mod types;

pub mod core {
    pub mod audio;
    pub mod dsp;
    pub mod engine;
    pub mod splitter;
}

pub mod model {
    pub mod model_manager;
    pub mod registry;
}

pub mod io {
    pub mod crypto;
    pub mod net;
    pub mod paths;
    pub mod progress;
}

// Public API
pub use crate::core::splitter::split_file;
pub use crate::io::progress::set_download_progress_callback;
pub use crate::model::model_manager::{ensure_model, ModelHandle, ModelManifest};
pub use crate::types::{SplitOptions, SplitResult};

pub fn prepare_model(model_name: &str, manifest_url_override: Option<&str>) -> error::Result<()> {
    let handle = ensure_model(model_name, manifest_url_override)?;
    crate::core::engine::preload(&handle)?;
    Ok(())
}
