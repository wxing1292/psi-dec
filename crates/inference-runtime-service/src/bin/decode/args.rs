use std::path::PathBuf;

use clap::ArgAction;
use clap::Parser;
use clap::ValueEnum;
use inference_runtime_core::config::DEFAULT_SAMPLING_TEMPERATURE;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_K;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_P;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ChatTemplateMode {
    Auto,
    Raw,
    QwenFixed,
    Custom,
}

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(long, default_value = "http://127.0.0.1:50061")]
    pub server_url: String,

    #[arg(long)]
    pub request_id: Option<u64>,

    #[arg(long, value_name = "DIR")]
    pub hf_model_dir: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    pub tokenizer_file: Option<PathBuf>,

    #[arg(long, conflicts_with = "prompt_file")]
    pub prompt_str: Option<String>,

    #[arg(long, value_name = "FILE", conflicts_with = "prompt_str")]
    pub prompt_file: Option<PathBuf>,

    #[arg(long, default_value = "")]
    pub system_prompt: String,

    #[arg(long, value_enum, default_value = "auto")]
    pub chat_template: ChatTemplateMode,

    #[arg(long, conflicts_with = "chat_template_file")]
    pub chat_template_str: Option<String>,

    #[arg(long, value_name = "FILE", conflicts_with = "chat_template_str")]
    pub chat_template_file: Option<PathBuf>,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub enable_thinking: bool,

    #[arg(long, conflicts_with = "enable_thinking")]
    pub disable_thinking: bool,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub preserve_thinking: bool,

    #[arg(long, default_value_t = 16)]
    pub max_sampled_tokens: u32,

    #[arg(long)]
    pub max_total_tokens: Option<u32>,

    #[arg(long, default_value_t = DEFAULT_SAMPLING_TEMPERATURE)]
    pub temperature: f32,

    #[arg(long, default_value_t = DEFAULT_SAMPLING_TOP_K)]
    pub top_k: usize,

    #[arg(long, default_value_t = DEFAULT_SAMPLING_TOP_P)]
    pub top_p: f32,

    #[arg(long)]
    pub seed: Option<u32>,

    #[arg(long)]
    pub timeout_ms: Option<u64>,

    #[arg(long = "output-str", default_value_t = true, action = ArgAction::Set, conflicts_with = "no_output_str")]
    pub output_str: bool,

    #[arg(long = "no-output-str", action = ArgAction::SetTrue)]
    pub no_output_str: bool,

    #[arg(long, value_name = "FILE")]
    pub output_file: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub show_stats: bool,

    #[arg(long, default_value_t = false)]
    pub print_prompt: bool,

    /// Preserve chat-template/special-token text when printing prompt/output.
    #[arg(long, default_value_t = false)]
    pub raw: bool,
}
