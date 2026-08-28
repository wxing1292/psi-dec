use std::collections::BTreeMap;
use std::path::Path;

use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::tokenizer::huggingface::HFTokenizer;
use serde::Deserialize;
use tokenizers::AddedToken;
use tokenizers::SplitDelimiterBehavior;
use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::pre_tokenizers::split::SplitPattern;

const QWEN2_PRETOKENIZE_REGEX: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
    r"[^\r\n\p{L}\p{N}]?\p{L}+|",
    r"\p{N}|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|",
    r"\s*[\r\n]+|",
    r"\s+(?!\S)|",
    r"\s+",
);

#[derive(Deserialize)]
struct TokenizerConfig {
    add_prefix_space: bool,
    added_tokens_decoder: BTreeMap<u32, AddedTokenConfig>,
}

#[derive(Deserialize)]
struct AddedTokenConfig {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: bool,
}

pub fn load(model_dir: &Path) -> Result<HFTokenizer> {
    let vocab = model_dir.join("vocab.json");
    let merges = model_dir.join("merges.txt");
    let model = BPE::from_file(utf8_path(&vocab)?, utf8_path(&merges)?)
        .build()
        .map_err(|error| Error::internal(format!("unable to load Qwen3-ASR BPE files: {error}")))?;

    let config_path = model_dir.join("tokenizer_config.json");
    let config_file = std::fs::File::open(&config_path).map_err(|error| {
        Error::internal(format!(
            "unable to open Qwen3-ASR tokenizer config {config_path:?}: {error}"
        ))
    })?;
    let config: TokenizerConfig = serde_json::from_reader(config_file).map_err(|error| {
        Error::internal(format!(
            "unable to parse Qwen3-ASR tokenizer config {config_path:?}: {error}"
        ))
    })?;

    let split = Split::new(
        SplitPattern::Regex(QWEN2_PRETOKENIZE_REGEX.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .expect("the Qwen2 pre-tokenization regex must compile");
    let byte_level = ByteLevel::new(config.add_prefix_space, true, false);
    let mut tokenizer = Tokenizer::new(model);
    tokenizer
        .with_normalizer(Some(NFC))
        .expect("NFC must be a valid tokenizer normalizer");
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![split.into(), byte_level.into()])));
    tokenizer.with_decoder(Some(byte_level));

    let added_tokens = config
        .added_tokens_decoder
        .into_iter()
        .map(|(id, token)| {
            let added_token = AddedToken::from(token.content.clone(), token.special)
                .lstrip(token.lstrip)
                .normalized(token.normalized)
                .rstrip(token.rstrip)
                .single_word(token.single_word);
            (id, token.content, added_token)
        })
        .collect::<Vec<_>>();
    tokenizer
        .add_tokens(added_tokens.iter().map(|(_, _, token)| token.clone()))
        .map_err(|error| Error::internal(format!("unable to add Qwen3-ASR tokenizer tokens: {error}")))?;
    for (expected_id, content, _) in &added_tokens {
        let actual_id = tokenizer
            .token_to_id(content)
            .expect("an added Qwen3-ASR tokenizer token must be addressable");
        if actual_id != *expected_id {
            return Err(Error::internal(format!(
                "Qwen3-ASR tokenizer maps {content:?} to {actual_id}; expected {expected_id}"
            )));
        }
    }
    Ok(HFTokenizer::new(tokenizer))
}

fn utf8_path(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| Error::internal(format!("Qwen3-ASR tokenizer path must be UTF-8: {path:?}")))
}

#[cfg(test)]
#[path = "tokenizer_test.rs"]
mod tests;
