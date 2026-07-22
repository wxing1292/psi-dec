use crate::args::ChatTemplateMode;
use crate::config::ChatTemplateConfig;
use crate::config::ModelConfig;
use crate::error::DecodeCliResult;

mod qwen_fixed;
mod render;
mod source;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct RenderedPrompt {
    text: String,
}

impl RenderedPrompt {
    pub fn new(text: String) -> Self {
        Self { text }
    }
    pub fn text(&self) -> &str {
        &self.text
    }
}

pub struct ChatTemplateRenderer {
    config: ChatTemplateConfig,
    model: ModelConfig,
}

impl ChatTemplateRenderer {
    pub fn new(config: ChatTemplateConfig, model: ModelConfig) -> Self {
        Self { config, model }
    }

    pub fn render(&self, user_prompt: &str) -> DecodeCliResult<RenderedPrompt> {
        let text = match self.config.mode() {
            ChatTemplateMode::Raw => user_prompt.to_string(),
            ChatTemplateMode::QwenFixed => qwen_fixed::render(&self.config, user_prompt)?,
            ChatTemplateMode::Custom => self.render_required_template(user_prompt)?,
            ChatTemplateMode::Auto => self.render_auto(user_prompt)?,
        };
        Ok(RenderedPrompt::new(text))
    }

    fn render_required_template(&self, user_prompt: &str) -> DecodeCliResult<String> {
        let template = source::load_explicit(&self.config)?
            .ok_or_else(|| "--chat-template custom requires --chat-template-str or --chat-template-file".to_string())?;
        render_template("custom", &template, &self.config, user_prompt)
    }

    fn render_auto(&self, user_prompt: &str) -> DecodeCliResult<String> {
        let template = if let Some(template) = source::load_explicit(&self.config)? {
            template
        } else {
            source::load_model_template(&self.model).transpose()?.ok_or_else(|| {
                "--chat-template auto requires a chat template from --chat-template-str, --chat-template-file, or \
                 --hf-model-dir; use --chat-template raw to bypass"
                    .to_string()
            })?
        };
        render_template("auto", &template, &self.config, user_prompt)
    }
}

fn render_template(
    mode_name: &str,
    template: &str,
    config: &ChatTemplateConfig,
    user_prompt: &str,
) -> DecodeCliResult<String> {
    if qwen_fixed::looks_like_qwen_template(template) {
        return qwen_fixed::render(config, user_prompt);
    }

    render::render(template, config, user_prompt)
        .map_err(|err| format!("unable to render {mode_name} chat template: {err}").into())
}
