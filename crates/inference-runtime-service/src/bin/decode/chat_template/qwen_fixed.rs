use crate::config::ChatTemplateConfig;
use crate::error::DecodeCliResult;

/// Render the Qwen 3.5/3.6 chat prefix used by this decode binary.
///
/// The upstream Qwen templates use Python/HF-Jinja syntax that minijinja does
/// not fully support (`messages[::-1]`, string methods, namespace mutation,
/// etc.). For this CLI we only expose the single-turn text path, so keeping a
/// small explicit renderer is both cleaner and more correct than partially
/// normalizing the full HF template.
pub fn render(config: &ChatTemplateConfig, user_prompt: &str) -> DecodeCliResult<String> {
    let mut prompt = String::new();

    if !config.system_prompt().is_empty() {
        push_message(&mut prompt, "system", config.system_prompt());
    }
    push_message(&mut prompt, "user", user_prompt);

    prompt.push_str("<|im_start|>assistant\n");
    if config.enable_thinking() {
        prompt.push_str("<think>\n");
    } else {
        prompt.push_str("<think>\n\n</think>\n\n");
    }

    Ok(prompt)
}

pub fn looks_like_qwen_template(chat_template: &str) -> bool {
    chat_template.contains("<|im_start|>")
        && chat_template.contains("<|im_end|>")
        && chat_template.contains("enable_thinking")
        && (chat_template.contains("messages[::-1]")
            || chat_template.contains("reasoning_content")
            || chat_template.contains("<think>"))
}

fn push_message(prompt: &mut String, role: &str, content: &str) {
    prompt.push_str("<|im_start|>");
    prompt.push_str(role);
    prompt.push('\n');
    prompt.push_str(content);
    prompt.push_str("<|im_end|>\n");
}
