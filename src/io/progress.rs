// src/core/progress.rs
use std::sync::{Mutex, OnceLock};

static DOWNLOAD_PROGRESS_CB: OnceLock<Mutex<Option<Box<dyn Fn(u64, u64) + Send + 'static>>>> =
    OnceLock::new();
static SPLIT_PROGRESS_CB: OnceLock<Mutex<Option<Box<dyn Fn(SplitProgress) + Send + 'static>>>> =
    OnceLock::new();

#[derive(Debug, Clone, serde::Serialize)]
pub enum SplitProgress {
    Stage(&'static str),
    Chunks {
        done: usize,
        total: usize,
        percent: f32,
    },
    Writing {
        stem: String,
        done: usize,
        total: usize,
        percent: f32,
    },
    Finished,
}

pub fn set_download_progress_callback(cb: impl Fn(u64, u64) + Send + 'static) {
    let _ = DOWNLOAD_PROGRESS_CB.set(Mutex::new(Some(Box::new(cb))));
}

pub fn emit_download_progress(done: u64, total: u64) {
    if let Some(m) = DOWNLOAD_PROGRESS_CB.get() {
        if let Ok(g) = m.lock() {
            if let Some(cb) = &*g {
                cb(done, total);
            }
        }
    }
}

pub fn set_split_progress_callback(cb: impl Fn(SplitProgress) + Send + 'static) {
    let _ = SPLIT_PROGRESS_CB.set(Mutex::new(Some(Box::new(cb))));
}

pub fn emit_split_progress(p: SplitProgress) {
    if let Some(m) = SPLIT_PROGRESS_CB.get() {
        if let Ok(g) = m.lock() {
            if let Some(cb) = &*g {
                cb(p);
            }
        }
    }
}
