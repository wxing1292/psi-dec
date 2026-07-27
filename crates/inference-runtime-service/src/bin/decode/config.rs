use std::path::Path;
use std::path::PathBuf;

use hf_chat_template::ChatTemplate;
use inference_runtime_core::chat_template;
use inference_runtime_core::config::MAX_SAMPLING_TOP_K;

use crate::args::Args;
use crate::args::ChatTemplateMode;
use crate::error::DecodeCliResult;

#[derive(Debug)]
pub struct DecodeConfig {
    model: ModelConfig,
    input: InputConfig,
    chat_template: ChatTemplateConfig,
    sampling: DecodeSamplingConfig,
    runtime: RuntimeConfig,
    output: OutputConfig,
}

#[derive(Debug)]
pub struct ModelConfig {
    hf_model_dir: Option<PathBuf>,
    tokenizer_file: Option<PathBuf>,
}

#[derive(Debug)]
pub struct InputConfig {
    prompt_str: Option<String>,
    prompt_file: Option<PathBuf>,
}

#[derive(Debug)]
pub struct ChatTemplateConfig {
    mode: ChatTemplateMode,
    template_str: Option<String>,
    template_file: Option<PathBuf>,
    system_prompt: String,
    enable_thinking: bool,
    preserve_thinking: bool,
}

#[derive(Debug)]
pub struct DecodeSamplingConfig {
    max_sampled_tokens: u32,
    max_total_tokens: Option<u32>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u32>,
}

#[derive(Debug)]
pub struct RuntimeConfig {
    server_url: String,
    timeout_ms: Option<u64>,
}

#[derive(Debug)]
pub struct OutputConfig {
    output_str: bool,
    output_file: Option<PathBuf>,
    show_stats: bool,
    print_prompt: bool,
}

impl DecodeConfig {
    pub fn from_args(args: Args) -> DecodeCliResult<Self> {
        validate_prompt(&args)?;
        validate_tokenizer(&args)?;
        validate_chat_template(&args)?;
        validate_sampling(&args)?;

        Ok(Self {
            model: ModelConfig {
                hf_model_dir: args.hf_model_dir,
                tokenizer_file: args.tokenizer_file,
            },
            input: InputConfig {
                prompt_str: args.prompt_str,
                prompt_file: args.prompt_file,
            },
            chat_template: ChatTemplateConfig {
                mode: args.chat_template,
                template_str: args.chat_template_str,
                template_file: args.chat_template_file,
                system_prompt: args.system_prompt,
                enable_thinking: !args.disable_thinking && args.enable_thinking,
                preserve_thinking: args.preserve_thinking,
            },
            sampling: DecodeSamplingConfig {
                max_sampled_tokens: args.max_sampled_tokens,
                max_total_tokens: args.max_total_tokens,
                temperature: args.temperature,
                top_k: args.top_k,
                top_p: args.top_p,
                seed: args.seed,
            },
            runtime: RuntimeConfig {
                server_url: args.server_url,
                timeout_ms: args.timeout_ms,
            },
            output: OutputConfig {
                output_str: args.output_str && !args.no_output_str,
                output_file: args.output_file,
                show_stats: args.show_stats,
                print_prompt: args.print_prompt,
            },
        })
    }

    pub fn model(&self) -> &ModelConfig {
        &self.model
    }
    pub fn input(&self) -> &InputConfig {
        &self.input
    }
    pub fn chat_template(&self) -> &ChatTemplateConfig {
        &self.chat_template
    }
    pub fn sampling(&self) -> &DecodeSamplingConfig {
        &self.sampling
    }
    pub fn runtime(&self) -> &RuntimeConfig {
        &self.runtime
    }
    pub fn output(&self) -> &OutputConfig {
        &self.output
    }
}

impl ModelConfig {
    pub fn hf_model_dir(&self) -> Option<&Path> {
        self.hf_model_dir.as_deref()
    }
    pub fn tokenizer_file(&self) -> Option<&Path> {
        self.tokenizer_file.as_deref()
    }
}

impl InputConfig {
    pub fn read_prompt(&self) -> DecodeCliResult<String> {
        if let Some(prompt) = &self.prompt_str {
            return Ok(prompt.clone());
        }
        let path = self
            .prompt_file
            .as_ref()
            .expect("validated input config must contain one prompt source");
        std::fs::read_to_string(path).map_err(|error| format!("unable to read prompt file {path:?}: {error}").into())
    }
}

impl ChatTemplateConfig {
    pub fn load(&self, model: &ModelConfig) -> DecodeCliResult<Option<ChatTemplate>> {
        match self.mode {
            ChatTemplateMode::Raw => Ok(None),
            ChatTemplateMode::Auto | ChatTemplateMode::Custom => {
                if let Some(source) = self.read_source()? {
                    let template = chat_template::compile(&source, model.hf_model_dir())?;
                    return Ok(Some(template));
                }
                let model_dir = model
                    .hf_model_dir()
                    .expect("validated auto template config must include a model directory");
                chat_template::load(model_dir).map(Some).map_err(Into::into)
            },
        }
    }

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
    pub fn enable_thinking(&self) -> bool {
        self.enable_thinking
    }
    pub fn preserve_thinking(&self) -> bool {
        self.preserve_thinking
    }

    fn read_source(&self) -> DecodeCliResult<Option<String>> {
        if let Some(template) = &self.template_str {
            return Ok(Some(template.clone()));
        }
        let Some(template_file) = &self.template_file else {
            return Ok(None);
        };
        std::fs::read_to_string(template_file)
            .map(Some)
            .map_err(|error| format!("unable to read chat template file {template_file:?}: {error}").into())
    }
}

impl DecodeSamplingConfig {
    pub fn max_sampled_tokens(&self) -> u32 {
        self.max_sampled_tokens
    }
    pub fn max_total_tokens(&self) -> Option<u32> {
        self.max_total_tokens
    }
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
    pub fn top_k(&self) -> usize {
        self.top_k
    }
    pub fn top_p(&self) -> f32 {
        self.top_p
    }
    pub fn seed(&self) -> Option<u32> {
        self.seed
    }
}

impl RuntimeConfig {
    pub fn server_url(&self) -> &str {
        &self.server_url
    }
    pub fn timeout_ms(&self) -> Option<u64> {
        self.timeout_ms
    }
}

impl OutputConfig {
    pub fn output_str(&self) -> bool {
        self.output_str
    }
    pub fn output_file(&self) -> Option<&Path> {
        self.output_file.as_deref()
    }
    pub fn show_stats(&self) -> bool {
        self.show_stats
    }
    pub fn print_prompt(&self) -> bool {
        self.print_prompt
    }
}

fn validate_prompt(args: &Args) -> DecodeCliResult<()> {
    match (&args.prompt_str, &args.prompt_file) {
        (Some(_), None) | (None, Some(_)) => Ok(()),
        (None, None) => Err("exactly one of --prompt-str or --prompt-file must be provided".into()),
        (Some(_), Some(_)) => Err("--prompt-str and --prompt-file are mutually exclusive".into()),
    }
}

fn validate_tokenizer(args: &Args) -> DecodeCliResult<()> {
    if args.tokenizer_file.is_none() && args.hf_model_dir.is_none() {
        return Err("prompt decode requires tokenizer assets; pass --tokenizer-file or --hf-model-dir".into());
    }
    Ok(())
}

fn validate_chat_template(args: &Args) -> DecodeCliResult<()> {
    if args.chat_template_str.is_some() && args.chat_template_file.is_some() {
        return Err("--chat-template-str and --chat-template-file are mutually exclusive".into());
    }

    let has_explicit_template = args.chat_template_str.is_some() || args.chat_template_file.is_some();
    match args.chat_template {
        ChatTemplateMode::Raw if has_explicit_template => {
            Err(format!(
                "--chat-template {:?} does not accept --chat-template-str or --chat-template-file",
                args.chat_template,
            )
            .into())
        },
        ChatTemplateMode::Raw => Ok(()),
        ChatTemplateMode::Custom if !has_explicit_template => {
            Err("--chat-template custom requires --chat-template-str or --chat-template-file".into())
        },
        ChatTemplateMode::Auto if !has_explicit_template && args.hf_model_dir.is_none() => {
            Err("--chat-template auto requires --hf-model-dir or an explicit chat template".into())
        },
        ChatTemplateMode::Custom | ChatTemplateMode::Auto => Ok(()),
    }
}

fn validate_sampling(args: &Args) -> DecodeCliResult<()> {
    if args.max_sampled_tokens == 0 {
        return Err("--max-sampled-tokens must be greater than 0".into());
    }
    if !args.temperature.is_finite() || args.temperature < 0.0 {
        return Err(format!(
            "--temperature must be finite and non-negative, got {}",
            args.temperature
        )
        .into());
    }
    if args.top_k == 0 || args.top_k > MAX_SAMPLING_TOP_K {
        return Err(format!("--top-k must be in [1, {MAX_SAMPLING_TOP_K}], got {}", args.top_k).into());
    }
    if !args.top_p.is_finite() || !(0.0..=1.0).contains(&args.top_p) {
        return Err(format!("--top-p must be finite and in [0, 1], got {}", args.top_p).into());
    }
    Ok(())
}
