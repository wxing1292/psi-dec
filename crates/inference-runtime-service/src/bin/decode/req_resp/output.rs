use inference_runtime_service::perf_metrics::DecodePerfMetrics;

use crate::config::OutputConfig;
use crate::error::DecodeCliResult;
use crate::req_resp::DecodeResponse;

#[derive(Clone, Debug)]
pub struct DecodeOutput {
    text: String,
}

impl DecodeOutput {
    pub fn new(text: String) -> Self {
        Self { text }
    }
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn from_response(
        response: DecodeResponse,
        config: &OutputConfig,
        output_already_streamed: bool,
    ) -> DecodeCliResult<Self> {
        if config.output_str() && !output_already_streamed && !response.text().is_empty() {
            println!("{}", response.text());
        }
        if let Some(output_file) = config.output_file() {
            std::fs::write(output_file, response.text())
                .map_err(|err| format!("unable to write decode output to {output_file:?}: {err}"))?;
        }
        if config.show_stats() {
            print_stats(response.metrics());
        }
        Ok(DecodeOutput::new(response.text().to_string()))
    }
}

fn print_stats(metrics: &DecodePerfMetrics) {
    eprint!("{}", format_stats(metrics));
    eprintln!("{}", metrics.json_line());
}

fn format_stats(metrics: &DecodePerfMetrics) -> String {
    format!(
        "==========\nPrompt: {} tokens, {} tokens-per-sec\nGeneration: {} tokens, {} tokens-per-sec\n",
        metrics.input_tokens,
        format_rate(metrics.prompt_tokens_per_s()),
        metrics.sampled_tokens,
        format_rate(metrics.decode_tokens_per_s()),
    )
}

fn format_rate(rate: Option<f64>) -> String {
    match rate {
        Some(rate) => format!("{rate:.3}"),
        None => "n/a".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use inference_runtime_service::perf_metrics::DecodePerfMetrics;

    use super::*;

    #[test]
    fn stats_report_prompt_and_generation_rates() {
        let metrics = DecodePerfMetrics {
            request_id: 7,
            input_tokens: 34,
            max_sampled_tokens: 1024,
            sampled_tokens: 1024,
            chunk_count: 1024,
            elapsed: Duration::from_millis(11_389),
            ttft: Some(Duration::from_millis(634)),
            decode_elapsed: Some(Duration::from_micros(10_747_880)),
            inter_token_latencies: Vec::new(),
        };

        assert_eq!(
            format_stats(&metrics),
            "==========\nPrompt: 34 tokens, 53.628 tokens-per-sec\nGeneration: 1024 tokens, 95.275 tokens-per-sec\n"
        );
    }
}
