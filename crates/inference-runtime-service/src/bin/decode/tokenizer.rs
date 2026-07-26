use inference_runtime_core::tokenizer::huggingface::HFTokenizer;

use crate::config::ModelConfig;
use crate::error::DecodeCliResult;

pub fn load_tokenizer(config: &ModelConfig) -> DecodeCliResult<HFTokenizer> {
    if let Some(tokenizer_file) = config.tokenizer_file() {
        let tokenizer = HFTokenizer::from_file(tokenizer_file)
            .map_err(|err| format!("unable to initialize tokenizer from {tokenizer_file:?}: {err:?}"))?;
        return Ok(tokenizer);
    }

    if let Some(hf_model_dir) = config.hf_model_dir() {
        let tokenizer_path = hf_model_dir.join("tokenizer.json");
        let tokenizer = HFTokenizer::from_file(&tokenizer_path)
            .map_err(|err| format!("unable to initialize tokenizer from {tokenizer_path:?}: {err:?}"))?;
        return Ok(tokenizer);
    }

    Err("config validation should require tokenizer assets".into())
}
