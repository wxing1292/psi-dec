use std::sync::Arc;

use axum::Json;
use axum::extract::Multipart;
use axum::extract::State;
use axum::extract::multipart::Field;
use axum::extract::multipart::MultipartRejection;
use axum::response::IntoResponse;
use axum::response::Response;

use crate::asr::PreparedTranscription;
use crate::asr::Qwen3ASRService;
use crate::rpc::http::TranscriptionsServer;
use crate::rpc::http::error::HTTPError;
use crate::rpc::http::error::invalid_request;
use crate::rpc::http::error::map_error;

const MAX_OUTPUT_TOKENS: usize = 512;

struct Input {
    file: Vec<u8>,
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
}

pub async fn handle<const N: usize, const L: usize, const P: usize>(
    State(server): State<TranscriptionsServer<N, L, P>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, HTTPError> {
    let mut multipart =
        multipart.map_err(|error| invalid_request(format!("invalid multipart transcription request: {error}")))?;
    let mut file = None;
    let mut model = None;
    let mut language = None;
    let mut prompt = None;
    let mut response_format = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| invalid_request(format!("invalid multipart transcription request: {error}")))?
    {
        let name = field
            .name()
            .ok_or_else(|| invalid_request("transcription multipart field must have a name"))?
            .to_string();
        match name.as_str() {
            "file" => {
                if file.is_some() {
                    return Err(invalid_request("transcription request must contain one file field"));
                }
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|error| invalid_request(format!("unable to read transcription file: {error}")))?
                        .to_vec(),
                );
            },
            "model" => {
                if model.is_some() {
                    return Err(invalid_request("transcription request must contain one model field"));
                }
                model = Some(field_text(field, "model").await?);
            },
            "language" => {
                if language.is_some() {
                    return Err(invalid_request(
                        "transcription request must contain at most one language field",
                    ));
                }
                language = Some(field_text(field, "language").await?);
            },
            "prompt" => {
                if prompt.is_some() {
                    return Err(invalid_request(
                        "transcription request must contain at most one prompt field",
                    ));
                }
                prompt = Some(field_text(field, "prompt").await?);
            },
            "response_format" => {
                if response_format.is_some() {
                    return Err(invalid_request(
                        "transcription request must contain at most one response_format field",
                    ));
                }
                response_format = Some(field_text(field, "response_format").await?);
            },
            _ => {
                return Err(invalid_request(format!("unsupported transcription field {name:?}")));
            },
        }
    }

    let input = Input {
        file: file.ok_or_else(|| invalid_request("transcription request requires a WAV file field"))?,
        model: model.ok_or_else(|| invalid_request("transcription request requires a model field"))?,
        language,
        prompt,
        response_format,
    };
    let (prepared, response_format) = validate_input(Arc::clone(&server.asr), input).await?;
    let transcription = server
        .asr
        .transcribe(&server.inference, prepared, MAX_OUTPUT_TOKENS)
        .await
        .map_err(map_error)?;
    if response_format == "text" {
        Ok(transcription.text.into_response())
    } else {
        Ok(Json(transcription).into_response())
    }
}

async fn validate_input(asr: Arc<Qwen3ASRService>, input: Input) -> Result<(PreparedTranscription, String), HTTPError> {
    if input.model.trim().is_empty() {
        return Err(invalid_request("model must not be empty"));
    }
    let response_format = input.response_format.unwrap_or_else(|| "json".to_string());
    if !matches!(response_format.as_str(), "json" | "text") {
        return Err(invalid_request("Qwen3-ASR response_format must be json or text"));
    }

    let prepared = tokio::task::spawn_blocking(move || {
        asr.prepare_wav(&input.file, input.language.as_deref(), input.prompt.as_deref())
    })
    .await
    .map_err(|error| {
        map_error(inference_runtime_core::Error::internal(format!(
            "audio preparation failed: {error}"
        )))
    })?
    .map_err(map_error)?;
    Ok((prepared, response_format))
}

async fn field_text(field: Field<'_>, name: &str) -> Result<String, HTTPError> {
    field
        .text()
        .await
        .map_err(|error| invalid_request(format!("unable to read transcription {name}: {error}")))
}
