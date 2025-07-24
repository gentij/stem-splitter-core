use crate::types::StemResult;
use anyhow::Result;

pub trait StemModel {
    fn separate(&self, input: &[f32]) -> Result<StemResult>;
}

pub struct OnnxModel {
    pub model_name: String,
}

impl OnnxModel {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
        }
    }
}

impl StemModel for OnnxModel {
    fn separate(&self, input: &[f32]) -> Result<StemResult> {
        println!(
            "[MOCK] Running ONNX model: {} on {} samples",
            self.model_name,
            input.len()
        );

        let quarter = input.len() / 4;

        Ok(StemResult {
            vocals: input[..quarter].to_vec(),
            drums: input[quarter..quarter * 2].to_vec(),
            bass: input[quarter * 2..quarter * 3].to_vec(),
            other: input[quarter * 3..].to_vec(),
        })
    }
}
