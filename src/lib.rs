//! # stem-splitter-core
//!
//! Core audio stem splitting functionality for reading files,
//! processing them with models, and returning separated stems.

mod audio;
mod model;
mod pipeline;
mod types;

pub use crate::{
    pipeline::split_file,
    types::{SplitConfig, StemResult},
};
