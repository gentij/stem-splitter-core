use clap::{Parser, Subcommand};
use stem_splitter_core::{prepare_model, set_download_progress_callback, set_split_progress_callback, split_file, SplitOptions, SplitProgress};
use std::process;

#[derive(Parser)]
#[command(name = "stem-splitter")]
#[command(about = "AI-powered audio stem separation tool", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Split {
        #[arg(short, long)]
        input: String,
        
        #[arg(short, long, default_value = ".")]
        output: String,
        
        #[arg(short, long, default_value = "htdemucs_ort_v1")]
        model: String,
        
        #[arg(long)]
        manifest_url: Option<String>,
        
        #[arg(short, long)]
        quiet: bool,
    },
    
    Prepare {
        #[arg(short, long, default_value = "htdemucs_ort_v1")]
        model: String,
        
        #[arg(long)]
        manifest_url: Option<String>,
        
        #[arg(short, long)]
        quiet: bool,
    },
    
    /// List available models
    List,
}

fn main() {
    let cli = Cli::parse();
    
    let result = match cli.command {
        Commands::Split {
            input,
            output,
            model,
            manifest_url,
            quiet,
        } => handle_split(input, output, model, manifest_url, quiet),
        Commands::Prepare {
            model,
            manifest_url,
            quiet,
        } => handle_prepare(model, manifest_url, quiet),
        Commands::List => handle_list(),
    };
    
    match result {
        Ok(()) => process::exit(0),
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    }
}

fn handle_split(
    input: String,
    output: String,
    model: String,
    manifest_url: Option<String>,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !std::path::Path::new(&input).exists() {
        return Err(format!("Input file not found: {}", input).into());
    }
    
    if !quiet {
        setup_progress_callbacks();
    }
    
    let opts = SplitOptions {
        output_dir: output.clone(),
        model_name: model.clone(),
        manifest_url_override: manifest_url,
    };
    
    if !quiet {
        eprintln!("🎵 Stem Splitter");
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("Input:  {}", input);
        eprintln!("Output: {}", output);
        eprintln!("Model:  {}", model);
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!();
    }
    
    let result = split_file(&input, opts)?;
    
    if !quiet {
        eprintln!();
        eprintln!("✅ Split completed successfully!");
        eprintln!();
        eprintln!("Output files:");
        eprintln!("  🎤 Vocals: {}", result.vocals_path);
        eprintln!("  🥁 Drums:  {}", result.drums_path);
        eprintln!("  🎸 Bass:   {}", result.bass_path);
        eprintln!("  🎹 Other:  {}", result.other_path);
    } else {
        // Quiet mode: just print paths
        println!("{}", result.vocals_path);
        println!("{}", result.drums_path);
        println!("{}", result.bass_path);
        println!("{}", result.other_path);
    }
    
    Ok(())
}

fn handle_prepare(
    model: String,
    manifest_url: Option<String>,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        eprintln!("📦 Preparing model: {}", model);
        eprintln!();
        
        set_download_progress_callback(|downloaded, total| {
            if total > 0 {
                let percent = (downloaded as f64 / total as f64 * 100.0).round() as u64;
                let downloaded_mb = downloaded as f64 / 1_000_000.0;
                let total_mb = total as f64 / 1_000_000.0;
                eprint!(
                    "\rDownloading model: {:>3}% ({:.2} MB / {:.2} MB)",
                    percent, downloaded_mb, total_mb
                );
                if downloaded >= total {
                    eprintln!();
                }
            } else {
                eprint!("\rDownloading model: {:.2} MB", downloaded as f64 / 1_000_000.0);
            }
        });
    }
    
    prepare_model(&model, manifest_url.as_deref())?;
    
    if !quiet {
        eprintln!("✅ Model prepared successfully!");
    }
    
    Ok(())
}

fn handle_list() -> Result<(), Box<dyn std::error::Error>> {
    let registry_json = include_str!("../../models/registry.json");
    let registry: serde_json::Value = serde_json::from_str(registry_json)?;
    
    eprintln!("📋 Available Models");
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    if let Some(models) = registry.get("models").and_then(|m| m.as_array()) {
        let default = registry.get("default")
            .and_then(|d| d.as_str())
            .unwrap_or("");
        
        for model in models {
            if let Some(name) = model.get("name").and_then(|n| n.as_str()) {
                let is_default = name == default;
                let marker = if is_default { " (default)" } else { "" };
                eprintln!("  • {}{}", name, marker);
            }
        }
    }
    
    eprintln!();
    eprintln!("Use --model <name> to specify a model");
    
    Ok(())
}

fn setup_progress_callbacks() {
    set_download_progress_callback(|downloaded, total| {
        if total > 0 {
            let percent = (downloaded as f64 / total as f64 * 100.0).round() as u64;
            let downloaded_mb = downloaded as f64 / 1_000_000.0;
            let total_mb = total as f64 / 1_000_000.0;
            eprint!(
                "\r📥 Downloading model: {:>3}% ({:.2} MB / {:.2} MB)",
                percent, downloaded_mb, total_mb
            );
            if downloaded >= total {
                eprintln!();
            }
        } else {
            eprint!("\r📥 Downloading model: {:.2} MB", downloaded as f64 / 1_000_000.0);
        }
    });
    
    set_split_progress_callback(|progress| {
        match progress {
            SplitProgress::Stage(stage) => {
                let stage_name = match stage {
                    "resolve_model" => "Resolving model",
                    "engine_preload" => "Loading model",
                    "read_audio" => "Reading audio file",
                    "infer" => "Processing audio",
                    "write_stems" => "Writing stems",
                    "finalize" => "Finalizing",
                    _ => stage,
                };
                eprintln!("⏳ {}", stage_name);
            }
            SplitProgress::Chunks {
                done,
                total,
                percent,
            } => {
                eprint!("\r🔄 Processing: {}/{} chunks ({:.0}%)", done, total, percent);
                if done >= total {
                    eprintln!();
                }
            }
            SplitProgress::Writing {
                stem,
                done,
                total,
                percent,
            } => {
                eprint!(
                    "\r💾 Writing {}: {}/{} ({:.0}%)",
                    stem, done, total, percent
                );
                if done >= total {
                    eprintln!();
                }
            }
            SplitProgress::Finished => {
                // This is handled in the main function
            }
        }
    });
}

