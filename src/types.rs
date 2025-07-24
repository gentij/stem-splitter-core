#[derive(Debug, Clone)]
pub struct StemResult {
    pub vocals: Vec<f32>,
    pub drums: Vec<f32>,
    pub bass: Vec<f32>,
    pub other: Vec<f32>,
}

#[derive(Debug)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone)]
pub struct SplitConfig {
    pub model_name: String,
    pub output_dir: String,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            model_name: "demucs".to_string(),
            output_dir: ".".to_string(),
        }
    }
}

impl SplitConfig {
    pub fn model(mut self, name: &str) -> Self {
        self.model_name = name.to_string();
        self
    }

    pub fn output_dir(mut self, dir: &str) -> Self {
        self.output_dir = dir.to_string();
        self
    }
}
