//! # stem-splitter-core
//!
//! Core audio stem splitting functionality for reading files,
//! processing them with models, and returning separated stems.

mod audio;
mod model;
mod pipeline;
mod types;
mod utils;

pub use crate::{
    audio::{read_audio, write_audio},
    pipeline::split_file,
    types::{SplitConfig, StemResult},
};
