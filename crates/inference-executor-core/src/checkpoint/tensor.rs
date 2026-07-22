use safetensors::tensor::TensorView;

use crate::def::ModelExecutorError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedTensorBindings {
    pub weight: String,
    pub scales: String,
    pub biases: String,
}

pub struct TensorBytes {
    name: String,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl TensorBytes {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    pub fn from_view(
        name: &str,
        expected_dtype: safetensors::Dtype,
        view: &TensorView<'_>,
    ) -> Result<Self, ModelExecutorError> {
        if view.dtype() != expected_dtype {
            return Err(ModelExecutorError::custom(format!(
                "unexpected dtype for tensor {name:?}: expected {:?}, got {:?}",
                expected_dtype,
                view.dtype()
            )));
        }
        Ok(Self {
            name: name.to_string(),
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        })
    }
}
