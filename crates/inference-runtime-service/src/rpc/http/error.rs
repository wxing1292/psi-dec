use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use inference_runtime_core::Error;
use serde_json::json;

pub struct HTTPError {
    error: Error,
}

impl IntoResponse for HTTPError {
    fn into_response(self) -> Response {
        let (status, code, openai_type) = classify(&self.error);
        let body = body(&self.error, code, openai_type);
        (status, Json(body)).into_response()
    }
}

pub fn map_error(error: Error) -> HTTPError {
    HTTPError { error }
}

pub fn invalid_request(message: impl Into<String>) -> HTTPError {
    map_error(Error::invalid_argument(message))
}

pub fn openai_error_body(error: &Error) -> serde_json::Value {
    let (_, code, openai_type) = classify(error);
    body(error, code, openai_type)
}

fn body(error: &Error, code: &'static str, openai_type: &'static str) -> serde_json::Value {
    json!({
        "error": {
            "message": error.to_string(),
            "type": openai_type,
            "param": null,
            "code": code,
        }
    })
}

fn classify(error: &Error) -> (StatusCode, &'static str, &'static str) {
    match error {
        Error::InvalidArgument(_) => (StatusCode::BAD_REQUEST, "invalid_request", "invalid_request_error"),
        Error::ResourceExhausted(_) => (StatusCode::TOO_MANY_REQUESTS, "resource_exhausted", "server_error"),
        Error::Cancelled(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "server_error"),
        Error::DeadlineExceeded(_) => (StatusCode::GATEWAY_TIMEOUT, "deadline_exceeded", "server_error"),
        Error::Aborted(_) => (StatusCode::CONFLICT, "aborted", "server_error"),
        Error::Unavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable", "server_error"),
        Error::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "server_error"),
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use inference_runtime_core::Error;

    use super::classify;

    #[test]
    fn test_domain_errors_use_canonical_http_classes() {
        for (error, expected_status, expected_code) in [
            (
                Error::invalid_argument("invalid"),
                StatusCode::BAD_REQUEST,
                "invalid_request",
            ),
            (
                Error::resource_exhausted("decode queue is full"),
                StatusCode::TOO_MANY_REQUESTS,
                "resource_exhausted",
            ),
            (
                Error::unavailable("runtime is stopped"),
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
            ),
            (
                Error::cancelled("request was cancelled"),
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
            ),
            (
                Error::deadline_exceeded("request deadline exceeded"),
                StatusCode::GATEWAY_TIMEOUT,
                "deadline_exceeded",
            ),
            (Error::aborted("request was aborted"), StatusCode::CONFLICT, "aborted"),
            (
                Error::internal("internal failure"),
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
            ),
        ] {
            let (status, code, _) = classify(&error);
            assert_eq!(status, expected_status);
            assert_eq!(code, expected_code);
        }
    }
}
