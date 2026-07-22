use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use safetensors::SafeTensors;

use crate::checkpoint::index::SafeTensorIndex;
use crate::checkpoint::mapped_file::MappedFile;
use crate::checkpoint::tensor::TensorBytes;
use crate::def::ModelExecutorError;

pub struct SafeTensorStore {
    model_dir: PathBuf,
    index: SafeTensorIndex,
    mapped_files: HashMap<PathBuf, MappedFile>,
}

impl SafeTensorStore {
    pub fn new(model_dir: impl AsRef<Path>, index: SafeTensorIndex) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            index,
            mapped_files: HashMap::new(),
        }
    }

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, ModelExecutorError> {
        let index = SafeTensorIndex::load(&model_dir)?;
        Ok(Self::new(model_dir, index))
    }

    pub fn load(&mut self, file_name: &str) -> Result<(), ModelExecutorError> {
        let file_path = self.file_path(file_name);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.mapped_files.entry(file_path) {
            let mapped = MappedFile::open(entry.key())?;
            entry.insert(mapped);
        }
        Ok(())
    }

    pub fn unload(&mut self, file_name: &str) {
        let file_path = self.file_path(file_name);
        self.mapped_files.remove(&file_path);
    }

    pub fn load_all(&mut self) -> Result<(), ModelExecutorError> {
        for file_name in self.index.file_names().map(ToOwned::to_owned).collect::<Vec<_>>() {
            self.load(&file_name)?;
        }
        Ok(())
    }

    pub fn unload_all(&mut self) {
        self.mapped_files.clear();
    }

    pub fn index(&self) -> &SafeTensorIndex {
        &self.index
    }

    pub fn tensor_bytes(
        &mut self,
        tensor_name: &str,
        dtype: safetensors::Dtype,
    ) -> Result<TensorBytes, ModelExecutorError> {
        let file_name = self.index().file_name_for(tensor_name)?.to_string();
        self.load(&file_name)?;
        let file_path = self.file_path(&file_name);
        let mapped = self
            .mapped_files
            .get(&file_path)
            .expect("safetensors file must be inserted before reading");
        let tensors = SafeTensors::deserialize(mapped.as_bytes()).map_err(|err| {
            ModelExecutorError::custom(format!(
                "unable to deserialize safetensors file {:?}, err: {err:?}",
                file_path
            ))
        })?;
        let view = tensors.tensor(tensor_name).map_err(|err| {
            ModelExecutorError::custom(format!(
                "unable to read tensor {tensor_name:?} from safetensors file {:?}, err: {err:?}",
                file_path
            ))
        })?;
        TensorBytes::from_view(tensor_name, dtype, &view)
    }

    fn file_path(&self, file_name: &str) -> PathBuf {
        self.model_dir.join(file_name)
    }
}
