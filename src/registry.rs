use crate::error::{Result, StemError};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RegistryEntry {
    pub name: String,
    pub manifest: String,
}

#[derive(Debug, Deserialize)]
pub struct Registry {
    pub default: String,
    pub models: Vec<RegistryEntry>,
}

const REGISTRY_JSON: &str = include_str!("../models/registry.json");

pub fn resolve_manifest_url(model_name: &str) -> Result<String> {
    let reg: Registry = serde_json::from_str(REGISTRY_JSON)?;
    let target = if model_name.is_empty() {
        reg.default
    } else {
        model_name.to_string()
    };

    reg.models
        .into_iter()
        .find(|m| m.name == target)
        .map(|m| m.manifest)
        .ok_or_else(|| StemError::Registry(format!("Model `{target}` not found in registry")))
}
