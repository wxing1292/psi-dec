use std::collections::HashMap;

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

pub type TensorMap = HashMap<String, TensorBytes>;

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

    pub fn expect_dtype(self, expected: safetensors::Dtype) -> Result<Self, ModelExecutorError> {
        if self.dtype != expected {
            return Err(ModelExecutorError::custom(format!(
                "unexpected dtype for tensor {:?}: expected {:?}, got {:?}",
                self.name, expected, self.dtype
            )));
        }
        Ok(self)
    }

    pub fn from_view(name: &str, view: &TensorView<'_>) -> Self {
        Self {
            name: name.to_string(),
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        }
    }
}

pub fn remove_tensor(
    tensors: &mut TensorMap,
    name: &str,
    expected_dtype: safetensors::Dtype,
) -> Result<TensorBytes, ModelExecutorError> {
    tensors
        .remove(name)
        .ok_or_else(|| ModelExecutorError::custom(format!("missing loaded tensor {name:?}")))?
        .expect_dtype(expected_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_tensor_transfers_ownership() {
        let mut tensors = TensorMap::from([(
            "layers.0.weight".to_string(),
            TensorBytes {
                name: "layers.0.weight".to_string(),
                dtype: safetensors::Dtype::BF16,
                shape: vec![2, 3],
                data: vec![0; 12],
            },
        )]);

        let tensor = remove_tensor(&mut tensors, "layers.0.weight", safetensors::Dtype::BF16).unwrap();

        assert_eq!(tensor.name(), "layers.0.weight");
        assert_eq!(tensor.shape(), [2, 3]);
        assert!(tensors.is_empty());
    }

    #[test]
    fn test_remove_tensor_validates_actual_dtype() {
        let mut tensors = TensorMap::from([(
            "weight".to_string(),
            TensorBytes {
                name: "weight".to_string(),
                dtype: safetensors::Dtype::U32,
                shape: vec![1],
                data: vec![0; 4],
            },
        )]);

        let error = match remove_tensor(&mut tensors, "weight", safetensors::Dtype::BF16) {
            Ok(_) => panic!("dtype mismatch must fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("expected BF16, got U32"));
        assert!(tensors.is_empty());
    }
}
