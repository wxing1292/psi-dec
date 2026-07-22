use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use crate::config::InputConfig;
use crate::error::DecodeCliResult;

#[derive(Clone, Debug)]
pub struct DecodeInput {
    request_id: u64,
    prompt: String,
}

impl DecodeInput {
    pub fn from_config(config: &InputConfig) -> DecodeCliResult<Self> {
        let prompt = match (config.prompt_str(), config.prompt_file()) {
            (Some(prompt), None) => prompt.to_string(),
            (None, Some(prompt_file)) => {
                std::fs::read_to_string(prompt_file)
                    .map_err(|err| format!("unable to read prompt file {prompt_file:?}: {err}"))?
            },
            _ => return Err("exactly one of --prompt-str or --prompt-file must be provided".into()),
        };
        Ok(Self {
            request_id: match config.request_id() {
                Some(request_id) => request_id,
                None => default_request_id()?,
            },
            prompt,
        })
    }

    pub fn request_id(&self) -> u64 {
        self.request_id
    }
    pub fn prompt(&self) -> &str {
        &self.prompt
    }
}

fn default_request_id() -> DecodeCliResult<u64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system clock is before UNIX_EPOCH: {err}"))?;
    Ok(elapsed.as_millis() as u64)
}
