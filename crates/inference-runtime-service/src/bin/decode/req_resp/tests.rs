use std::path::PathBuf;
use std::time::Duration;

use inference_runtime_service::perf_metrics::DecodePerfMetrics;

use crate::args::Args;
use crate::args::ChatTemplateMode;
use crate::config::DecodeConfig;
use crate::req_resp::DecodeInput;
use crate::req_resp::DecodeOutput;
use crate::req_resp::DecodeResponse;

#[test]
fn decode_input_reads_prompt_str() {
    let config = DecodeConfig::from_args(test_args()).expect("config should be valid");
    let input = DecodeInput::from_config(config.input()).expect("input should be valid");
    assert_eq!(input.request_id(), 7);
    assert_eq!(input.prompt(), "hello");
}

#[test]
fn decode_output_writes_full_text_to_file() {
    let mut args = test_args();
    args.output_str = false;
    args.output_file = Some(temp_output_path());
    let output_path = args.output_file.clone().expect("path should exist");
    let config = DecodeConfig::from_args(args).expect("config should be valid");

    let response = DecodeResponse::new("final".to_string(), test_metrics());
    let output =
        DecodeOutput::from_response(response, config.output(), false).expect("finish should write output file");

    assert_eq!(output.text(), "final");
    assert_eq!(std::fs::read_to_string(&output_path).expect("read output"), "final");
    let _ = std::fs::remove_file(output_path);
}

fn test_metrics() -> DecodePerfMetrics {
    DecodePerfMetrics {
        request_id: 7,
        input_tokens: 3,
        max_sampled_tokens: 2,
        sampled_tokens: 2,
        chunk_count: 1,
        elapsed: Duration::from_millis(10),
        ttft: Some(Duration::from_millis(2)),
        decode_elapsed: Some(Duration::from_millis(8)),
        inter_token_latencies: Vec::new(),
    }
}

fn temp_output_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "decode-output-{}.txt",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ))
}

fn test_args() -> Args {
    Args {
        server_url: "http://127.0.0.1:50051".to_string(),
        request_id: Some(7),
        hf_model_dir: None,
        tokenizer_file: Some(PathBuf::from("tokenizer.json")),
        prompt_str: Some("hello".to_string()),
        prompt_file: None,
        system_prompt: String::new(),
        chat_template: ChatTemplateMode::Raw,
        chat_template_str: None,
        chat_template_file: None,
        enable_thinking: true,
        disable_thinking: false,
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
