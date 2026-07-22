use minijinja::Environment;
use minijinja::context;
use serde_json::Value;
use serde_json::json;

use crate::config::ChatTemplateConfig;

pub fn render(chat_template: &str, config: &ChatTemplateConfig, user_prompt: &str) -> Result<String, minijinja::Error> {
    let mut env = Environment::new();
    add_hf_compat_filters(&mut env);
    let chat_template = normalize_hf_template_for_minijinja(chat_template);
    env.add_template("chat", &chat_template)?;

    let messages = build_messages(config, user_prompt);
    let template = env.get_template("chat")?;
    template.render(context! {
        messages => messages,
        tools => Vec::<Value>::new(),
        add_generation_prompt => true,
        enable_thinking => config.enable_thinking(),
        preserve_thinking => config.preserve_thinking(),
        add_vision_id => false,
    })
}

fn add_hf_compat_filters(env: &mut Environment<'_>) {
    env.add_filter("startswith", |value: String, prefix: String| value.starts_with(&prefix));
    env.add_filter("starts_with", |value: String, prefix: String| {
        value.starts_with(&prefix)
    });
    env.add_filter("endswith", |value: String, suffix: String| value.ends_with(&suffix));
    env.add_filter("ends_with", |value: String, suffix: String| value.ends_with(&suffix));
    env.add_filter("split", |value: String, pattern: String| {
        value.split(&pattern).map(str::to_string).collect::<Vec<_>>()
    });
    env.add_filter("strip", strip_chars);
    env.add_filter("trim", strip_chars);
    env.add_filter("lstrip", lstrip_chars);
    env.add_filter("rstrip", rstrip_chars);
    env.add_filter("lower", |value: String| value.to_lowercase());
    env.add_filter("upper", |value: String| value.to_uppercase());
}

fn normalize_hf_template_for_minijinja(chat_template: &str) -> String {
    chat_template
        .replace(".startswith(", "|startswith(")
        .replace(".endswith(", "|endswith(")
        .replace(".split(", "|split(")
        .replace(".strip()", "|strip")
        .replace(".lstrip()", "|lstrip")
        .replace(".rstrip()", "|rstrip")
        .replace(".lstrip(", "|lstrip(")
        .replace(".rstrip(", "|rstrip(")
        .replace(".lower()", "|lower")
        .replace(".upper()", "|upper")
}

fn strip_chars(value: String, chars: Option<String>) -> String {
    let Some(chars) = chars else {
        return value.trim().to_string();
    };
    value.trim_matches(|c| chars.contains(c)).to_string()
}

fn lstrip_chars(value: String, chars: Option<String>) -> String {
    let Some(chars) = chars else {
        return value.trim_start().to_string();
    };
    value.trim_start_matches(|c| chars.contains(c)).to_string()
}

fn rstrip_chars(value: String, chars: Option<String>) -> String {
    let Some(chars) = chars else {
        return value.trim_end().to_string();
    };
    value.trim_end_matches(|c| chars.contains(c)).to_string()
}

fn build_messages(config: &ChatTemplateConfig, user_prompt: &str) -> Vec<Value> {
    let mut messages = Vec::new();
    if !config.system_prompt().is_empty() {
        messages.push(json!({"role": "system", "content": config.system_prompt()}));
    }
    messages.push(json!({"role": "user", "content": user_prompt}));
    messages
}
