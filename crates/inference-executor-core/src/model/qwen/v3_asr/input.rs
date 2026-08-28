use crate::def::ModelExecutorError;
use crate::model::qwen::v3_asr::audio_output_rows;

#[derive(Debug)]
pub struct Qwen3ASRAudioSource {
    features: Vec<f32>,
    num_mel_bins: usize,
    num_frames: usize,
}

impl Qwen3ASRAudioSource {
    pub fn new(features: Vec<f32>, num_mel_bins: usize, num_frames: usize) -> Result<Self, ModelExecutorError> {
        if num_mel_bins == 0 || num_frames == 0 {
            return Err(ModelExecutorError::custom(
                "Qwen3-ASR prepared audio dimensions must be positive",
            ));
        }
        let num_features = num_mel_bins
            .checked_mul(num_frames)
            .ok_or_else(|| ModelExecutorError::custom("Qwen3-ASR prepared audio size must fit usize"))?;
        if features.len() != num_features {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3-ASR prepared audio has {} features; expected {num_features}",
                features.len()
            )));
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err(ModelExecutorError::custom(
                "Qwen3-ASR prepared audio features must be finite",
            ));
        }
        Ok(Self {
            features,
            num_mel_bins,
            num_frames,
        })
    }

    pub fn features(&self) -> &[f32] {
        &self.features
    }

    pub const fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }

    pub const fn num_frames(&self) -> usize {
        self.num_frames
    }

    pub fn num_resource_tokens(&self) -> usize {
        audio_output_rows(self.num_frames)
    }
}
