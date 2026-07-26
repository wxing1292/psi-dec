use inference_runtime_service::perf_metrics::DecodePerfMetrics;

pub fn format_stats(metrics: &DecodePerfMetrics) -> String {
    format!(
        "==========\nPrompt: {} tokens, {} tokens-per-sec\nGeneration: {} tokens, {} tokens-per-sec\n",
        metrics.input_tokens,
        format_rate(metrics.prompt_tokens_per_s()),
        metrics.sampled_tokens,
        format_rate(metrics.decode_tokens_per_s()),
    )
}

fn format_rate(rate: Option<f64>) -> String {
    rate.map_or_else(|| "n/a".to_string(), |rate| format!("{rate:.3}"))
}
