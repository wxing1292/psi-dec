use inference_runtime_core::tokenizer::HFTokenizer;

use crate::chat_template::RenderedPrompt;
use crate::config::DecodeSamplingConfig;
use crate::error::DecodeCliResult;
use crate::req_resp::DecodeInput;

#[derive(Clone, Debug)]
pub struct DecodeRequest {
    request_id: u64,
    prompt: RenderedPrompt,
    tokens: Vec<u32>,
    max_sampled_tokens: u32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u32>,
}

impl DecodeRequest {
    pub fn from_input(
        input: DecodeInput,
        prompt: RenderedPrompt,
        tokenizer: &HFTokenizer,
        config: &DecodeSamplingConfig,
    ) -> DecodeCliResult<Self> {
        let tokens = tokenizer
            .encode(prompt.text().to_string(), true)
            .map_err(|err| format!("unable to tokenize prompt: {err}"))?
            .get_ids()
            .to_vec();
        let max_sampled_tokens = resolve_max_sampled_tokens(tokens.len(), config)?;
        Ok(Self {
            request_id: input.request_id(),
            prompt,
            tokens,
            max_sampled_tokens,
            temperature: config.temperature(),
            top_k: config.top_k(),
            top_p: config.top_p(),
            seed: config.seed(),
        })
    }

    pub fn request_id(&self) -> u64 {
        self.request_id
    }
    pub fn prompt(&self) -> &RenderedPrompt {
        &self.prompt
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn max_sampled_tokens(&self) -> u32 {
        self.max_sampled_tokens
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

fn resolve_max_sampled_tokens(input_tokens: usize, config: &DecodeSamplingConfig) -> DecodeCliResult<u32> {
    let Some(max_total_tokens) = config.max_total_tokens() else {
        return Ok(config.max_sampled_tokens());
    };
    let max_total_tokens = usize::try_from(max_total_tokens)
        .map_err(|_| "max_total_tokens should fit into usize on this platform".to_string())?;
    if input_tokens >= max_total_tokens {
        return Err(format!(
            "input has {input_tokens} tokens but --max-total-tokens is {max_total_tokens}; increase \
             --max-total-tokens or shorten the prompt"
        )
        .into());
    }
    let remaining_tokens = max_total_tokens - input_tokens;
    Ok(config
        .max_sampled_tokens()
        .min(u32::try_from(remaining_tokens).map_err(|_| "remaining token count should fit into u32".to_string())?))
}
