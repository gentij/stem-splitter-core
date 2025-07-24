use crate::types::StemResult;
use anyhow::Result;

pub trait StemModel {
    fn separate(&self, input: &[f32]) -> Result<StemResult>;
}

pub struct DummyModel;

impl StemModel for DummyModel {
    fn separate(&self, input: &[f32]) -> Result<StemResult> {
        Ok(StemResult {
            vocals: input.to_vec(),
            drums: vec![],
            bass: vec![],
            other: vec![],
        })
    }
}
