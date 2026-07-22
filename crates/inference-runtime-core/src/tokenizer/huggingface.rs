use std::ops::Deref;
use std::ops::DerefMut;
use std::path::Path;
use std::str::FromStr;

use crate::Error;
use crate::Result;

pub struct HFTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl HFTokenizer {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn from_file(file: impl AsRef<Path>) -> Result<Self> {
        tokenizers::Tokenizer::from_file(file)
            .map(Self::new)
            .map_err(|err| Error::internal(format!("encountered err: {err:?}")))
    }

    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self> {
        tokenizers::Tokenizer::from_bytes(bytes)
            .map(Self::new)
            .map_err(|err| Error::internal(format!("encountered err: {err:?}")))
    }

    pub fn from_string(string: &str) -> Result<Self> {
        tokenizers::Tokenizer::from_str(string)
            .map(Self::new)
            .map_err(|err| Error::internal(format!("encountered err: {err:?}")))
    }
}

impl Deref for HFTokenizer {
    type Target = tokenizers::Tokenizer;

    fn deref(&self) -> &Self::Target {
        &self.tokenizer
    }
}

impl DerefMut for HFTokenizer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tokenizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_tokenizer() {
        let source = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let serialized = source.to_string(false).unwrap();
        let tokenizer = HFTokenizer::from_string(&serialized).unwrap();
        assert_eq!(tokenizer.get_vocab_size(false), 0);
    }
}
