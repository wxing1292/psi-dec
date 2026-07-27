use std::path::Path;

use hf_chat_template::ChatTemplate;
use hf_chat_template::TokenizerConfig;

use crate::Result;
use crate::log_err_internal;
use crate::log_info_invalid_argument;

pub fn load(model_dir: &Path) -> Result<ChatTemplate> {
    let config = load_tokenizer_config(model_dir)?;
    let template_path = model_dir.join("chat_template.jinja");
    if template_path.exists() {
        let source = std::fs::read_to_string(&template_path)
            .map_err(|error| log_err_internal!("unable to read chat template {template_path:?}: {error}"))?;
        return ChatTemplate::from_template_and_config(&source, &config)
            .map_err(|error| log_err_internal!("unable to compile chat template {template_path:?}: {error}"));
    }
    ChatTemplate::from_tokenizer_config(&config)
        .map_err(|error| log_err_internal!("unable to compile chat template from {model_dir:?}: {error}"))
}

pub fn compile(source: &str, model_dir: Option<&Path>) -> Result<ChatTemplate> {
    let template = match model_dir {
        Some(model_dir) => {
            let config = load_tokenizer_config(model_dir)?;
            ChatTemplate::from_template_and_config(source, &config)
        },
        None => ChatTemplate::from_str(source),
    };
    template.map_err(|error| log_info_invalid_argument!("unable to compile explicit chat template: {error}"))
}

fn load_tokenizer_config(model_dir: &Path) -> Result<TokenizerConfig> {
    let config_path = model_dir.join("tokenizer_config.json");
    let config = std::fs::read_to_string(&config_path)
        .map_err(|error| log_err_internal!("unable to read tokenizer config {config_path:?}: {error}"))?;
    serde_json::from_str(&config)
        .map_err(|error| log_err_internal!("unable to parse tokenizer config {config_path:?}: {error}"))
}

#[cfg(test)]
#[path = "mod_test.rs"]
mod tests;
