use clap::Parser;
use tracing_subscriber::EnvFilter;

mod args;
use args::Args;

mod config;
use config::DecodeConfig;

mod error;
use error::DecodeCliResult;

mod output;

mod client;
use client::InferenceClient;

mod stream;

mod tokenizer;
use tokenizer::load_tokenizer;

#[tokio::main]
async fn main() {
    init_tracing();

    if let Err(err) = run().await {
        eprintln!("decode failed: {err}");
        std::process::exit(1);
    }
}

async fn run() -> DecodeCliResult<()> {
    let config = DecodeConfig::from_args(Args::parse())?;
    let tokenizer = load_tokenizer(config.model())?;
    let template = config.chat_template().load(config.model())?;
    let prompt = config.input().read_prompt()?;
    InferenceClient::connect(config, template, tokenizer)
        .await?
        .decode(&prompt)
        .await?;
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
