use std::path::Path;

use serde_json::Value;

use crate::config::ChatTemplateConfig;
use crate::config::ModelConfig;
use crate::error::DecodeCliResult;

pub fn load_explicit(config: &ChatTemplateConfig) -> DecodeCliResult<Option<String>> {
    if let Some(template) = config.template_str() {
        return Ok(Some(template.to_string()));
    }
    let Some(template_file) = config.template_file() else {
        return Ok(None);
    };
    std::fs::read_to_string(template_file)
        .map(Some)
        .map_err(|err| format!("unable to read chat template file {template_file:?}: {err}").into())
}

pub fn load_model_template(config: &ModelConfig) -> Option<DecodeCliResult<String>> {
    if let Some(hf_model_dir) = config.hf_model_dir() {
        match load_model_dir_chat_template(hf_model_dir) {
            Some(Ok(template)) => return Some(Ok(template)),
            Some(Err(err)) => return Some(Err(err)),
            None => {},
        }
    }

    None
}

fn load_model_dir_chat_template(model_dir: &Path) -> Option<DecodeCliResult<String>> {
    let jinja_path = model_dir.join("chat_template.jinja");
    if jinja_path.exists() {
        return Some(
            std::fs::read_to_string(&jinja_path)
                .map_err(|err| format!("unable to read chat template file {jinja_path:?}: {err}").into()),
        );
    }

    let tokenizer_config_path = model_dir.join("tokenizer_config.json");
    if !tokenizer_config_path.exists() {
        return None;
    }

    let tokenizer_config = match std::fs::read_to_string(&tokenizer_config_path) {
        Ok(tokenizer_config) => tokenizer_config,
        Err(err) => {
            return Some(Err(format!(
                "unable to read tokenizer config {tokenizer_config_path:?}: {err}"
            )
            .into()));
        },
    };

    let tokenizer_config: Value = match serde_json::from_str(&tokenizer_config) {
        Ok(tokenizer_config) => tokenizer_config,
        Err(err) => {
            return Some(Err(format!(
                "unable to parse tokenizer config {tokenizer_config_path:?}: {err}"
            )
            .into()));
        },
    };

    tokenizer_config
        .get("chat_template")
        .and_then(Value::as_str)
        .map(|template| Ok(template.to_owned()))
}
