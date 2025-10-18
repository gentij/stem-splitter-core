use std::sync::{Mutex, OnceLock};

static DOWNLOAD_PROGRESS_CB: OnceLock<Mutex<Option<Box<dyn Fn(u64, u64) + Send + 'static>>>> =
    OnceLock::new();

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
