fn main() -> anyhow::Result<()> {
    stem_splitter_core::set_download_progress_callback(|done, total| {
        let pct = if total > 0 {
            (done as f64 / total as f64 * 100.0).round() as u64
        } else {
            0
        };

        if total > 0 {
            eprint!(
                "\rDownloading model… {:>3}% ({}/{} bytes)",
                pct, done, total
            );
        } else {
            eprint!("\rDownloading model… {} bytes", done);
        }
        if total > 0 && done >= total {
            eprintln!();
        }
    });

    let handle = stem_splitter_core::ensure_model("htdemucs_ort_v1", None)?;
    eprintln!("OK: cached at {}", handle.local_path.display());
    eprintln!(
        "Manifest says {} stems: {:?}",
        handle.manifest.stems.len(),
        handle.manifest.stems
    );
    Ok(())
}
