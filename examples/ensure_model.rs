fn main() -> anyhow::Result<()> {
    stem_splitter_core::set_download_progress_callback(|done, total| {
        let pct = if total > 0 {
            (done as f64 / total as f64 * 100.0).round() as u64
        } else {
            0
        };
        eprint!("\rDownloading model… {}% ({}/{} bytes)", pct, done, total);
    });

    let handle = stem_splitter_core::ensure_model("mdx_4stem_v1", None)?;
    eprintln!("\nOK: cached at {}", handle.local_path.display());
    eprintln!(
        "Manifest says {} stems: {:?}",
        handle.manifest.stems.len(),
        handle.manifest.stems
    );

    Ok(())
}
