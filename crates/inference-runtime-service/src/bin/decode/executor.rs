use std::sync::Arc;

use inference_runtime_core::tokenizer::HFTokenizer;
use inference_runtime_proto::inference_runtime_service::inference_runtime_client::InferenceRuntimeClient;
use tonic::transport::Channel;

use crate::chat_template::ChatTemplateRenderer;
use crate::config::DecodeConfig;
use crate::error::DecodeCliResult;
use crate::req_resp::DecodeInput;
use crate::req_resp::DecodeOutput;
use crate::req_resp::DecodeRequest;
use crate::stream::DecodeStreamExecutor;

pub struct DecodeExecutor {
    config: DecodeConfig,
    client: InferenceRuntimeClient<Channel>,
    tokenizer: Arc<HFTokenizer>,
}

impl DecodeExecutor {
    pub async fn connect(config: DecodeConfig, tokenizer: Arc<HFTokenizer>) -> DecodeCliResult<Self> {
        let client = InferenceRuntimeClient::connect(config.runtime().server_url().to_string())
            .await
            .map_err(|err| format!("unable to connect to {}: {err}", config.runtime().server_url()))?;
        Ok(Self {
            config,
            client,
            tokenizer,
        })
    }

    pub async fn execute(mut self, input: DecodeInput) -> DecodeCliResult<DecodeOutput> {
        let renderer = ChatTemplateRenderer::new(self.config.chat_template().clone(), self.config.model().clone());
        let rendered_prompt = renderer.render(input.prompt())?;
        let request = DecodeRequest::from_input(input, rendered_prompt, &self.tokenizer, self.config.sampling())?;

        if self.config.output().print_prompt() {
            let prompt_text = render_prompt_for_output(&self.tokenizer, &request, self.config.output().raw())?;
            println!("input: {prompt_text}");
        }

        let response = DecodeStreamExecutor::new(
            &mut self.client,
            &self.tokenizer,
            self.config.runtime(),
            self.config.output().raw(),
            self.config.output().output_str(),
        )
        .execute(&request)
        .await?;

        DecodeOutput::from_response(response, self.config.output(), self.config.output().output_str())
    }
}

fn render_prompt_for_output(tokenizer: &HFTokenizer, request: &DecodeRequest, raw: bool) -> DecodeCliResult<String> {
    if raw {
        return Ok(request.prompt().text().to_string());
    }

    tokenizer
        .decode(request.tokens(), true)
        .map_err(|err| format!("unable to decode prompt text: {err:?}").into())
}
