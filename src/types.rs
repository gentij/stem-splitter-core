use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitOptions {
    pub output_dir: String,
    pub model_name: String,
    pub manifest_url_override: Option<String>,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            output_dir: ".".into(),
            model_name: "mdx_4stem_v1".into(),
            manifest_url_override: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SplitResult {
    pub vocals_path: String,
    pub drums_path: String,
    pub bass_path: String,
    pub other_path: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Artifact {
    pub file: String,
    pub sha256: String,
    #[serde(alias = "size_bytes")]
    pub size_bytes: u64,
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IODesc {
    pub name: String,
    #[serde(default)]
    pub layout: String,
    #[serde(default)]
    pub dtype: String,
    #[serde(default)]
    pub shape: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelManifest {
    pub name: String,
    #[serde(default)]
    pub version: String,

    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub opset: Option<u32>,

    #[serde(alias = "sample_rate_hz")]
    pub sample_rate: u32,
    pub window: usize,
    pub hop: usize,

    #[serde(default)]
    pub stems: Vec<String>,

    #[serde(default)]
    pub input_layout: String,
    #[serde(default)]
    pub output_layout: String,

    #[serde(default)]
    pub inputs: Vec<IODesc>,
    #[serde(default)]
    pub outputs: Vec<IODesc>,

    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    #[serde(default)]
    pub entry: String,

    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub filesize: u64,
}

#[derive(Debug, Clone)]
pub struct ResolvedArtifact {
    pub file: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub url: String,
}

impl ModelManifest {
    pub fn resolve_primary_artifact(&self) -> Result<ResolvedArtifact, String> {
        if !self.artifacts.is_empty() {
            if !self.entry.is_empty() {
                if let Some(a) = self.artifacts.iter().find(|a| a.file == self.entry) {
                    return Ok(ResolvedArtifact {
                        file: a.file.clone(),
                        sha256: a.sha256.clone(),
                        size_bytes: a.size_bytes,
                        url: a.url.clone(),
                    });
                }
                return Err(format!("entry '{}' not found in artifacts[]", self.entry));
            }
            if self.artifacts.len() == 1 {
                let a = &self.artifacts[0];
                return Ok(ResolvedArtifact {
                    file: a.file.clone(),
                    sha256: a.sha256.clone(),
                    size_bytes: a.size_bytes,
                    url: a.url.clone(),
                });
            }
            return Err("multiple artifacts present but no 'entry' specified".into());
        }

        if self.url.is_empty() || self.sha256.is_empty() || self.filesize == 0 {
            return Err("manifest missing artifacts and legacy url/sha256/filesize".into());
        }
        let file = infer_filename_from_url(&self.url)
            .unwrap_or_else(|| format!("{}-{}.bin", self.name, &self.sha256[..8]));
        Ok(ResolvedArtifact {
            file,
            sha256: self.sha256.clone(),
            size_bytes: self.filesize,
            url: self.url.clone(),
        })
    }
}

fn infer_filename_from_url(url: &str) -> Option<String> {
    url.rsplit('/').next().map(|s| s.to_string())
}
