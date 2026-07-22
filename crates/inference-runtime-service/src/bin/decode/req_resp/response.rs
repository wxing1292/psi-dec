use inference_runtime_service::perf_metrics::DecodePerfMetrics;

#[derive(Clone, Debug)]
pub struct DecodeResponse {
    text: String,
    metrics: DecodePerfMetrics,
}

impl DecodeResponse {
    pub fn new(text: String, metrics: DecodePerfMetrics) -> Self {
        Self { text, metrics }
    }

    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn metrics(&self) -> &DecodePerfMetrics {
        &self.metrics
    }
}
