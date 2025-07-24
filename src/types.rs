#[derive(Debug, Clone)]
pub struct StemResult {
    pub vocals: Vec<f32>,
    pub drums: Vec<f32>,
    pub bass: Vec<f32>,
    pub other: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct StemSplitterConfig {
    pub model_type: String,
    pub quality: f32,
}

impl Default for StemSplitterConfig {
    fn default() -> Self {
        Self {
            model_type: "demucs".into(),
            quality: 0.8,
        }
    }
}

#[derive(Debug)]
pub struct StemSplitter {
    pub config: StemSplitterConfig,
}

impl StemSplitter {
    pub fn new() -> Self {
        Self {
            config: StemSplitterConfig::default(),
        }
    }
}
