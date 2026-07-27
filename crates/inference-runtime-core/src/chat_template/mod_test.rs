use std::path::Path;
use std::path::PathBuf;

use hf_chat_template::Message;

use super::compile;
use super::load;

#[test]
fn test_load_inline_template() {
    let model_dir = new_model_dir("inline");
    write_tokenizer_config(&model_dir, r#"{"chat_template":"inline: {{ messages[0].content }}"}"#);

    let template = load(&model_dir).unwrap();
    assert_eq!(
        template.render_messages(&[Message::user("hello")], false).unwrap(),
        "inline: hello"
    );
    remove_model_dir(&model_dir);
}

#[test]
fn test_load_standalone_template_precedence() {
    let model_dir = new_model_dir("standalone");
    write_tokenizer_config(&model_dir, r#"{"chat_template":"inline: {{ messages[0].content }}"}"#);
    std::fs::write(
        model_dir.join("chat_template.jinja"),
        "standalone: {{ messages[0].content }}",
    )
    .unwrap();

    let template = load(&model_dir).unwrap();
    assert_eq!(
        template.render_messages(&[Message::user("hello")], false).unwrap(),
        "standalone: hello"
    );
    remove_model_dir(&model_dir);
}

#[test]
fn test_load_invalid_tokenizer_config() {
    let model_dir = new_model_dir("invalid");
    write_tokenizer_config(&model_dir, "not json");

    let error = load(&model_dir).unwrap_err();
    assert!(error.to_string().contains("unable to parse tokenizer config"));
    remove_model_dir(&model_dir);
}

#[test]
fn test_compile_uses_explicit_template() {
    let model_dir = new_model_dir("explicit");
    write_tokenizer_config(&model_dir, r#"{"chat_template":"inline: {{ messages[0].content }}"}"#);

    let template = compile("explicit: {{ messages[0].content }}", Some(&model_dir)).unwrap();
    assert_eq!(
        template.render_messages(&[Message::user("hello")], false).unwrap(),
        "explicit: hello"
    );
    remove_model_dir(&model_dir);
}

#[test]
fn test_compile_without_checkpoint_config() {
    let template = compile("explicit: {{ messages[0].content }}", None).unwrap();
    assert_eq!(
        template.render_messages(&[Message::user("hello")], false).unwrap(),
        "explicit: hello"
    );
}

fn new_model_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "chat-template-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn write_tokenizer_config(model_dir: &Path, content: &str) {
    std::fs::write(model_dir.join("tokenizer_config.json"), content).unwrap();
}

fn remove_model_dir(model_dir: &Path) {
    std::fs::remove_dir_all(model_dir).unwrap();
}
