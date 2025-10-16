use std::{
    env,
    path::{Path, PathBuf},
};

pub fn script_path() -> String {
    env::var("STEM_SPLITTER_PYTHON_SCRIPT")
        .unwrap_or_else(|_| format!("{}/demucs_runner.py", env!("CARGO_MANIFEST_DIR")))
}

pub fn tmp_dir() -> PathBuf {
    env::var("STEM_SPLITTER_TMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("tmp"))
}
