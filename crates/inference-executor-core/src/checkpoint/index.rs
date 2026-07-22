use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use serde::Deserialize;

use crate::def::ModelExecutorError;

#[derive(Clone, Debug, Deserialize)]
pub struct SafeTensorIndex {
    weight_map: HashMap<String, String>,
}

pub enum SafeTensorIndexAction {
    Keep,
    Rename(String),
    Remove,
}

impl SafeTensorIndex {
    pub fn new(weight_map: HashMap<String, String>) -> Result<Self, ModelExecutorError> {
        if weight_map.is_empty() {
            return Err(ModelExecutorError::custom(
                "safetensors index must not have an empty weight_map",
            ));
        }
        Ok(Self { weight_map })
    }

    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, ModelExecutorError> {
        let index_path = model_dir.as_ref().join("model.safetensors.index.json");
        let file = File::open(&index_path).map_err(|err| {
            ModelExecutorError::custom(format!(
                "unable to open safetensors index file {:?}, err: {err:?}",
                index_path
            ))
        })?;
        let index = serde_json::from_reader::<_, SafeTensorIndex>(file).map_err(|err| {
            ModelExecutorError::custom(format!(
                "unable to parse safetensors index file {:?}, err: {err:?}",
                index_path
            ))
        })?;
        Self::new(index.weight_map).map_err(|err| {
            ModelExecutorError::custom(format!("invalid safetensors index file {:?}: {err}", index_path))
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    pub fn file_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.values().map(String::as_str)
    }

    pub fn contains(&self, tensor_name: &str) -> bool {
        self.weight_map.contains_key(tensor_name)
    }

    pub fn map_tensors(self, mut map: impl FnMut(&str) -> SafeTensorIndexAction) -> Result<Self, ModelExecutorError> {
        let mut weight_map = HashMap::with_capacity(self.weight_map.len());
        for (source_name, file_name) in self.weight_map {
            let target_name = match map(&source_name) {
                SafeTensorIndexAction::Keep => source_name,
                SafeTensorIndexAction::Rename(target_name) => target_name,
                SafeTensorIndexAction::Remove => continue,
            };
            if weight_map.insert(target_name.clone(), file_name).is_some() {
                return Err(ModelExecutorError::custom(format!(
                    "safetensors index mapping produced duplicate target name {target_name:?}"
                )));
            }
        }
        if weight_map.is_empty() {
            return Err(ModelExecutorError::custom(
                "safetensors index mapping produced an empty weight_map",
            ));
        }
        Self::new(weight_map)
    }

    pub fn file_name_for(&self, tensor_name: &str) -> Result<&str, ModelExecutorError> {
        self.weight_map
            .get(tensor_name)
            .map(String::as_str)
            .ok_or_else(|| ModelExecutorError::custom(format!("missing safetensor {tensor_name:?} in weight index")))
    }
}
