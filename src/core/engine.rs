use crate::{
    error::{Result, StemError},
    model::model_manager::{ModelHandle, ModelManifest},
};

use anyhow::anyhow;
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

pub fn run_window(left: &[f32], right: &[f32]) -> Result<Array3<f32>> {
    let mf = manifest();
    let input_is_bct = mf.input_layout.eq_ignore_ascii_case("BCT");
    let output_is_bsct = mf.output_layout.eq_ignore_ascii_case("BSCT");

    let t = left.len();
    if t != right.len() {
        return Err(anyhow!("L/R length mismatch").into());
    }

    // --- Build ONNX input tensor without ndarray interop ---
    let input_value: Value = if input_is_bct {
        // [1, 2, T] planar L then R
        let mut planar = Vec::with_capacity(2 * t);
        planar.extend_from_slice(left);
        planar.extend_from_slice(right);
        Tensor::from_array((vec![1, 2, t], planar))?.into_dyn()
    } else {
        // [1, T, 2] interleaved
        let mut inter = Vec::with_capacity(2 * t);
        for i in 0..t {
            inter.push(left[i]);
            inter.push(right[i]);
        }
        Tensor::from_array((vec![1, t, 2], inter))?.into_dyn()
    };

    let mut session = SESSION
        .get()
        .expect("engine::preload first")
        .lock()
        .expect("session poisoned");

    // In rc.10, `inputs` is a field.
    let input_name = session
        .inputs
        .get(0)
        .ok_or_else(|| anyhow!("Model has no inputs"))?
        .name
        .clone();

    // Named input -> named outputs
    let outputs = session.run(vec![(input_name, input_value)])?;

    // Take first output value
    let out0: Value = outputs
        .into_iter()
        .next()
        .map(|(_, v)| v)
        .ok_or_else(|| anyhow!("Model returned no outputs"))?;

    // --- Extract raw tensor view: (shape, data) ---
    // We ignore `shape` to avoid API differences; reconstruct sizes from data length + known T.
    let (_shape, data) = out0.try_extract_tensor::<f32>()?;
    let n = data.len();

    // --- Normalize to [S, 2, T] ---
    let out: Array3<f32> = if output_is_bsct {
        // Expect original layout [1, S, 2, T] (B=1).
        // data.len() must be S * 2 * T  ->  S = n / (2 * T)
        if t == 0 || n % (2 * t) != 0 {
            return Err(anyhow!(
                "Output buffer length {} is not divisible by 2*T (T = {})",
                n,
                t
            )
            .into());
        }
        let stems = n / (2 * t);
        if stems == 0 {
            return Err(anyhow!("Computed 0 stems from output tensor").into());
        }

        // Reorder [1,S,2,T] -> [S,2,T]
        let mut buf = vec![0f32; stems * 2 * t];
        // Indices for source [B,S,C,T] (B=1 -> b=0):
        // src = (((0 * S + s) * 2 + c) * T) + i  ==  ((s * 2 + c) * T) + i
        // dst [S,2,T] = ((s * 2 + c) * T) + i    (same formula)
        // -> layout is already contiguous per (s,c,i), so this copy can be a memcpy;
        // we still loop for clarity/robustness.
        for s in 0..stems {
            for c in 0..2 {
                let src_off = (s * 2 + c) * t;
                let dst_off = (s * 2 + c) * t;
                buf[dst_off..dst_off + t].copy_from_slice(&data[src_off..src_off + t]);
            }
        }
        Array3::from_shape_vec((stems, 2, t), buf)?
    } else {
        // Expect original layout [1, 2, T] (B=1).
        // data.len() must be 2 * T
        if n != 2 * t {
            return Err(anyhow!(
                "Expected stereo [1,2,T]; got {} samples but 2*T = {}",
                n,
                2 * t
            )
            .into());
        }

        // Reorder [1,2,T] -> [1,2,T] (no change, just place into Array3)
        let mut buf = vec![0f32; 1 * 2 * t];
        // src (b=0): ((0 * 2 + c) * T) + i == (c * T) + i
        for c in 0..2 {
            let src_off = c * t;
            let dst_off = c * t; // [1,2,T] flattened uses same stride ordering here
            buf[dst_off..dst_off + t].copy_from_slice(&data[src_off..src_off + t]);
        }
        Array3::from_shape_vec((1, 2, t), buf)?
    };

    Ok(out)
}
