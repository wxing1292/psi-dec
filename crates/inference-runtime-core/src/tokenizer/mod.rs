use crate::Result;
use crate::runtime::Token;

pub mod huggingface;

pub trait Tokenizer: Send + Sync {
    /// Encodes prepared inference text without adding control tokens.
    fn encode(&self, text: &str) -> Result<Vec<Token>>;

    /// Decodes tokens without filtering control tokens.
    fn decode(&self, tokens: &[Token]) -> Result<String>;
}
