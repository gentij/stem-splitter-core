use crate::{
    core::dsp::stft_cac_stereo_centered,
    error::{Result, StemError},
    model::model_manager::ModelHandle,
    types::ModelManifest,
};

use anyhow::{anyhow, Context};
use ndarray::Array3;
use once_cell::sync::OnceCell;
use ort::{
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
    value::{Tensor, Value},
};
use std::sync::Mutex;

static SESSION: OnceCell<Mutex<Session>> = OnceCell::new();
static MANIFEST: OnceCell<ModelManifest> = OnceCell::new();
static ORT_INIT: OnceCell<()> = OnceCell::new();

const DEMUCS_T: usize = 343_980;
const DEMUCS_F: usize = 2048;
const DEMUCS_FRAMES: usize = 336;
const DEMUCS_NFFT: usize = 4096;
const DEMUCS_HOP: usize = 1024;

pub fn preload(h: &ModelHandle) -> Result<()> {
    // Pin error type so `?` is unambiguous.
    ORT_INIT.get_or_try_init::<_, StemError>(|| {
        ort::init().commit().map_err(StemError::from)?;
        Ok(())
    })?;

    let session = SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(&h.local_path)?;

    SESSION.set(Mutex::new(session)).ok();
    MANIFEST.set(h.manifest.clone()).ok();
    Ok(())
}

pub fn manifest() -> &'static ModelManifest {
    MANIFEST
        .get()
        .expect("engine::preload() must be called once before using the engine")
}

pub fn run_window_demucs(left: &[f32], right: &[f32]) -> Result<Array3<f32>> {
    if left.len() != right.len() {
        return Err(anyhow!("L/R length mismatch").into());
    }
    let t = left.len();
    if t != DEMUCS_T {
        return Err(anyhow!("Bad window length {} (expected {})", t, DEMUCS_T).into());
    }

    // Build time branch [1,2,T], planar
    let mut planar = Vec::with_capacity(2 * t);
    planar.extend_from_slice(left);
    planar.extend_from_slice(right);
    let time_value: Value = Tensor::from_array((vec![1, 2, t], planar))?.into_dyn();

    // Build spec branch [1,4,F,Frames] with center padding, Hann, 4096/1024
    let (spec_cac, f_bins, frames) = stft_cac_stereo_centered(left, right, DEMUCS_NFFT, DEMUCS_HOP);
    if f_bins != DEMUCS_F || frames != DEMUCS_FRAMES {
        return Err(anyhow!(
            "Spec dims mismatch: got F={},Frames={}, expected F={},Frames={}",
            f_bins,
            frames,
            DEMUCS_F,
            DEMUCS_FRAMES
        )
        .into());
    }
    let spec_value: Value = Tensor::from_array((vec![1, 4, f_bins, frames], spec_cac))
        .context("spec tensor")?
        .into_dyn();

    let mut session = SESSION
        .get()
        .expect("engine::preload first")
        .lock()
        .expect("session poisoned");

    // Bind inputs by the names we saw in check_io.py
    let in_time = session
        .inputs
        .iter()
        .find(|i| i.name == "input")
        .map(|i| i.name.clone())
        .ok_or_else(|| anyhow!("Model missing input 'input'"))?;

    let in_spec = session
        .inputs
        .iter()
        .find(|i| i.name == "x")
        .map(|i| i.name.clone())
        .ok_or_else(|| anyhow!("Model missing input 'x'"))?;

    // Run
    let outputs = session.run(vec![(in_time, time_value), (in_spec, spec_value)])?;

    // Pick the time-domain stems: name "add_67", shape [1,4,2,T]
    let out_td: Value = outputs
        .into_iter()
        .find_map(|(name, v)| if name == "add_67" { Some(v) } else { None })
        .ok_or_else(|| anyhow!("Model did not return 'add_67' output"))?;

    // Extract [1,4,2,T] and squeeze to [4,2,T]
    let (_shape, data) = out_td.try_extract_tensor::<f32>()?;
    if data.len() != 1 * 4 * 2 * t {
        return Err(anyhow!(
            "Unexpected add_67 length {} (expected {})",
            data.len(),
            1 * 4 * 2 * t
        )
        .into());
    }
    let out = ndarray::Array3::from_shape_vec((4, 2, t), data.to_vec())?;
    Ok(out)
}
