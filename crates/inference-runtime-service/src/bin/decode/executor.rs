use inference_runtime_core::runtime::Token;
use inference_runtime_core::tokenizer::Tokenizer;
use inference_runtime_core::tokenizer::huggingface::HFTokenizer;
use inference_runtime_proto::inference_runtime_service::inference_runtime_client::InferenceRuntimeClient;
use tonic::transport::Channel;

use crate::chat_template::ChatTemplateRenderer;
use crate::config::DecodeConfig;
use crate::error::DecodeCliResult;
use crate::stream::DecodeStreamExecutor;
use crate::stream::DecodeStreamResult;

pub struct DecodeExecutor {
    config: DecodeConfig,
    client: InferenceRuntimeClient<Channel>,
    tokenizer: HFTokenizer,
}

impl DecodeExecutor {
    pub async fn connect(config: DecodeConfig, tokenizer: HFTokenizer) -> DecodeCliResult<Self> {
        let client = InferenceRuntimeClient::connect(config.runtime().server_url().to_string())
            .await
            .map_err(|err| format!("unable to connect to {}: {err}", config.runtime().server_url()))?;
        Ok(Self {
            config,
            client,
            tokenizer,
        })
    }

    pub async fn execute(mut self, prompt: &str) -> DecodeCliResult<()> {
        let renderer = ChatTemplateRenderer::new(self.config.chat_template().clone(), self.config.model().clone());
        let rendered_prompt = renderer.render(prompt)?;
        let request = DecodeRequest::from_input(rendered_prompt, &self.tokenizer, self.config.sampling())?;

        if self.config.output().print_prompt() {
            let prompt_text = render_prompt_for_output(&self.tokenizer, &request)?;
            println!("input: {prompt_text}");
        }

        let response = DecodeStreamExecutor::new(
            &mut self.client,
            &self.tokenizer,
            self.config.runtime(),
            self.config.output().output_str(),
        )
        .execute(&request)
        .await?;

        output_response(response, self.config.output())
    }
}

#[derive(Clone, Debug)]
pub struct DecodeRequest {
    pub prompt: crate::chat_template::RenderedPrompt,
    pub tokens: Vec<Token>,
    pub max_sampled_tokens: u32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u32>,
}

impl DecodeRequest {
    fn from_input<T: Tokenizer>(
        prompt: crate::chat_template::RenderedPrompt,
        tokenizer: &T,
        config: &crate::config::DecodeSamplingConfig,
    ) -> DecodeCliResult<Self> {
        let tokens = tokenizer
            .encode(prompt.text())
            .map_err(|err| format!("unable to tokenize prompt: {err}"))?;
        let max_sampled_tokens = resolve_max_sampled_tokens(tokens.len(), config)?;
        Ok(Self {
            prompt,
            tokens,
            max_sampled_tokens,
            temperature: config.temperature(),
            top_k: config.top_k(),
            top_p: config.top_p(),
            seed: config.seed(),
        })
    }
}

fn resolve_max_sampled_tokens(
    input_tokens: usize,
    config: &crate::config::DecodeSamplingConfig,
) -> DecodeCliResult<u32> {
    let Some(max_total_tokens) = config.max_total_tokens() else {
        return Ok(config.max_sampled_tokens());
    };
    let max_total_tokens = usize::try_from(max_total_tokens)
        .map_err(|_| "max_total_tokens should fit into usize on this platform".to_string())?;
    if input_tokens >= max_total_tokens {
        return Err(format!("input has {input_tokens} tokens but --max-total-tokens is {max_total_tokens}").into());
    }
    let remaining_tokens = max_total_tokens - input_tokens;
    Ok(config
        .max_sampled_tokens()
        .min(u32::try_from(remaining_tokens).map_err(|_| "remaining token count should fit into u32".to_string())?))
}

fn output_response(response: DecodeStreamResult, config: &crate::config::OutputConfig) -> DecodeCliResult<()> {
    if config.output_str() && !response.text.is_empty() && !response.streamed {
        println!("{}", response.text);
    }
    if let Some(output_file) = config.output_file() {
        std::fs::write(output_file, &response.text)
            .map_err(|err| format!("unable to write decode output to {output_file:?}: {err}"))?;
    }
    if config.show_stats() {
        eprint!("{}", crate::output::format_stats(&response.metrics));
        eprintln!("{}", response.metrics.json_line());
    }
    Ok(())
}

fn render_prompt_for_output<T: Tokenizer>(tokenizer: &T, request: &DecodeRequest) -> DecodeCliResult<String> {
    tokenizer
        .decode(&request.tokens)
        .map_err(|err| format!("unable to decode prompt text: {err:?}").into())
}
