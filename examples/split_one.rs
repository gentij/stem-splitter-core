use stem_splitter_core::SplitProgress;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().expect("usage: split_one <wav> [out_dir]");
    let out = args.next().unwrap_or_else(|| "./out".into());

    stem_splitter_core::set_download_progress_callback(|d, t| {
        let pct = if t > 0 {
            (d as f64 / t as f64 * 100.0).round() as u64
        } else {
            0
        };
        if t > 0 {
            eprint!("\rModel: {:>3}% ({}/{})", pct, d, t);
            if d >= t {
                eprintln!();
            }
        } else {
            eprint!("\rModel: {} bytes", d);
        }
    });

    stem_splitter_core::set_split_progress_callback(|p| match p {
        SplitProgress::Stage(s) => {
            eprintln!("> {}", s);
        }
        SplitProgress::Chunks {
            done,
            total,
            percent,
        } => {
            eprint!("\rSplit: {}/{} ({:.0}%)", done, total, percent);
            if done >= total {
                eprintln!();
            }
        }
        SplitProgress::Writing {
            ref stem,
            done,
            total,
            percent,
        } => {
            eprintln!("Writing {}: {}/{} ({:.0}%)", stem, done, total, percent);
        }
        SplitProgress::Finished => {
            eprintln!("Split finished.");
        }
    });

    let opts = stem_splitter_core::SplitOptions {
        output_dir: out,
        model_name: "htdemucs_ort_v1".into(),
        manifest_url_override: None,
    };

    let res = stem_splitter_core::split_file(&input, opts)?;
    eprintln!(
        "Done:\n{}\n{}\n{}\n{}",
        res.vocals_path, res.drums_path, res.bass_path, res.other_path
    );
    Ok(())
}
