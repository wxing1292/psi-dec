use std::path::Path;
use std::path::PathBuf;

use crate::args::Args;
use crate::args::ChatTemplateMode;
use crate::chat_template::ChatTemplateRenderer;
use crate::config::DecodeConfig;

#[test]
fn raw_mode_returns_user_prompt_without_template() {
    let config = config_with(ChatTemplateMode::Raw, None, true);
    let renderer = ChatTemplateRenderer::new(config.chat_template().clone(), config.model().clone());
    let prompt = renderer.render("hello").expect("raw render should succeed");
    assert_eq!(prompt.text(), "hello");
}

#[test]
fn qwen_fixed_mode_respects_enable_thinking() {
    let config = config_with(ChatTemplateMode::QwenFixed, None, true);
    let renderer = ChatTemplateRenderer::new(config.chat_template().clone(), config.model().clone());
    let prompt = renderer.render("hello").expect("qwen fixed render should succeed");
    assert!(prompt.text().contains("<|im_start|>user\nhello<|im_end|>"));
    assert!(prompt.text().contains(
        "<|im_start|>assistant
<think>"
    ));
}

#[test]
fn qwen_fixed_mode_disable_thinking_closes_empty_think_block() {
    let config = config_with(ChatTemplateMode::QwenFixed, None, false);
    let renderer = ChatTemplateRenderer::new(config.chat_template().clone(), config.model().clone());
    let prompt = renderer.render("hello").expect("qwen fixed render should succeed");
    assert!(
        prompt
            .text()
            .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    );
}

#[test]
fn custom_template_renders_enable_thinking_flag() {
    let template = "{{ messages[0].content }} {{ enable_thinking }}";
    let config = config_with(ChatTemplateMode::Custom, Some(template.to_string()), true);
    let renderer = ChatTemplateRenderer::new(config.chat_template().clone(), config.model().clone());
    let prompt = renderer.render("hello").expect("custom render should succeed");
    assert_eq!(prompt.text(), "hello true");
}

#[test]
fn auto_template_str_precedes_model_template() {
    let model_dir = temp_model_dir("auto-precedence");
    write_tokenizer_config(&model_dir, "MODEL: {{ messages[0].content }}");

    let mut args = args_with(
        ChatTemplateMode::Auto,
        Some("EXPLICIT: {{ messages[0].content }}".to_string()),
        true,
    );
    args.hf_model_dir = Some(model_dir.clone());
    args.tokenizer_file = None;
    let config = DecodeConfig::from_args(args).expect("config should be valid");
    let renderer = ChatTemplateRenderer::new(config.chat_template().clone(), config.model().clone());
    let prompt = renderer.render("hello").expect("auto render should succeed");

    assert_eq!(prompt.text(), "EXPLICIT: hello");
    remove_dir(&model_dir);
}

#[test]
fn auto_template_loads_hf_tokenizer_config_template() {
    let model_dir = temp_model_dir("auto-hf-tokenizer-config");
    write_tokenizer_config(&model_dir, "HF: {{ messages[0].content }} {{ enable_thinking }}");

    let mut args = args_with(ChatTemplateMode::Auto, None, false);
    args.hf_model_dir = Some(model_dir.clone());
    args.tokenizer_file = None;
    let config = DecodeConfig::from_args(args).expect("config should be valid");
    let renderer = ChatTemplateRenderer::new(config.chat_template().clone(), config.model().clone());
    let prompt = renderer.render("hello").expect("auto render should succeed");

    assert_eq!(prompt.text(), "HF: hello false");
    remove_dir(&model_dir);
}

#[test]
fn auto_template_reports_invalid_hf_tokenizer_config() {
    let model_dir = temp_model_dir("auto-hf-invalid-config");
    std::fs::write(model_dir.join("tokenizer_config.json"), "not json").expect("write invalid config");

    let mut args = args_with(ChatTemplateMode::Auto, None, true);
    args.hf_model_dir = Some(model_dir.clone());
    args.tokenizer_file = None;
    let config = DecodeConfig::from_args(args).expect("config should be valid");
    let renderer = ChatTemplateRenderer::new(config.chat_template().clone(), config.model().clone());
    let err = renderer
        .render("hello")
        .expect_err("invalid tokenizer config should error");

    assert!(err.to_string().contains("unable to parse tokenizer config"));
    remove_dir(&model_dir);
}

fn config_with(mode: ChatTemplateMode, template: Option<String>, enable_thinking: bool) -> DecodeConfig {
    DecodeConfig::from_args(args_with(mode, template, enable_thinking)).expect("config should be valid")
}

fn args_with(mode: ChatTemplateMode, template: Option<String>, enable_thinking: bool) -> Args {
    Args {
        server_url: "http://127.0.0.1:50051".to_string(),
        request_id: Some(1),
        hf_model_dir: None,
        tokenizer_file: Some(PathBuf::from("tokenizer.json")),
        prompt_str: Some("hello".to_string()),
        prompt_file: None,
        system_prompt: String::new(),
        chat_template: mode,
        chat_template_str: template,
        chat_template_file: None,
        enable_thinking,
        disable_thinking: !enable_thinking,
        preserve_thinking: true,
        max_sampled_tokens: 16,
        max_total_tokens: None,
        temperature: 0.7,
        top_k: 20,
        top_p: 0.8,
        seed: Some(42),
        timeout_ms: None,
        output_str: true,
        no_output_str: false,
        output_file: None,
        show_stats: false,
        print_prompt: false,
        raw: false,
    }
}

fn temp_model_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "decode-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).expect("create temp model dir");
    path
}

fn write_tokenizer_config(model_dir: &Path, template: &str) {
    let escaped = template.replace('\\', "\\\\").replace('"', "\\\"");
    std::fs::write(
        model_dir.join("tokenizer_config.json"),
        format!("{{\"chat_template\":\"{escaped}\"}}"),
    )
    .expect("write tokenizer config");
}

fn remove_dir(path: &Path) {
    let _ = std::fs::remove_dir_all(path);
}
