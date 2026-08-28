mod config;
mod input;

pub use config::Qwen3ASRAudioConfig;
pub use config::Qwen3ASRGenerationConfig;
pub use config::Qwen3ASRModelConfig;
pub use config::Qwen3ASRPreprocessorConfig;
pub use config::audio_output_rows;
pub use config::init_qwen3_asr_config;
pub use input::Qwen3ASRAudioSource;

pub mod weight_layout;

pub const QWEN3_ASR_AUDIO_RESOURCE_TYPE: inference_runtime_core::runtime::ResourceTypeID =
    inference_runtime_core::runtime::ResourceTypeID::new(1);
