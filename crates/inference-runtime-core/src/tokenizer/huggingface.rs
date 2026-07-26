use std::path::Path;
use std::str::FromStr;

use crate::Error;
use crate::Result;
use crate::runtime::Token;
use crate::tokenizer::Tokenizer;

type InnerDecodeStream<'a> = tokenizers::tokenizer::DecodeStream<
    'a,
    tokenizers::tokenizer::ModelWrapper,
    tokenizers::tokenizer::NormalizerWrapper,
    tokenizers::tokenizer::PreTokenizerWrapper,
    tokenizers::tokenizer::PostProcessorWrapper,
    tokenizers::tokenizer::DecoderWrapper,
>;

pub struct HFTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

pub struct IncrementalDecoder<'a> {
    inner: InnerDecodeStream<'a>,
}

impl HFTokenizer {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn from_file(file: impl AsRef<Path>) -> Result<Self> {
        let file = file.as_ref();
        tokenizers::Tokenizer::from_file(file)
            .map(Self::new)
            .map_err(|err| Error::internal(format!("unable to load tokenizer {file:?}: {err}")))
    }

    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self> {
        tokenizers::Tokenizer::from_bytes(bytes)
            .map(Self::new)
            .map_err(|err| Error::internal(format!("unable to parse tokenizer bytes: {err}")))
    }

    pub fn from_string(string: &str) -> Result<Self> {
        tokenizers::Tokenizer::from_str(string)
            .map(Self::new)
            .map_err(|err| Error::internal(format!("unable to parse tokenizer string: {err}")))
    }

    pub fn decode_without_special_tokens(&self, tokens: &[Token]) -> Result<String> {
        self.decode_tokens(tokens, true)
    }

    pub fn token(&self, value: &str) -> Option<Token> {
        self.tokenizer.token_to_id(value).map(Token::new)
    }

    fn decode_tokens(&self, tokens: &[Token], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(
                &tokens.iter().map(|token| token.value()).collect::<Vec<_>>(),
                skip_special_tokens,
            )
            .map_err(|err| Error::internal(format!("unable to decode tokens: {err}")))
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<Token>> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().iter().copied().map(Token::new).collect())
            .map_err(|err| Error::internal(format!("unable to encode text: {err}")))
    }

    fn decode(&self, tokens: &[Token]) -> Result<String> {
        self.decode_tokens(tokens, false)
    }
}

impl<'a> IncrementalDecoder<'a> {
    pub fn new(tokenizer: &'a HFTokenizer) -> Self {
        Self::with_special_tokens(tokenizer, true)
    }

    pub fn without_special_tokens(tokenizer: &'a HFTokenizer) -> Self {
        Self::with_special_tokens(tokenizer, false)
    }

    fn with_special_tokens(tokenizer: &'a HFTokenizer, include_special_tokens: bool) -> Self {
        Self {
            inner: tokenizer.tokenizer.decode_stream(!include_special_tokens),
        }
    }

    pub fn decode(&mut self, tokens: &[Token]) -> Result<Option<String>> {
        debug_assert!(
            !tokens.is_empty(),
            "incremental decoder requires a non-empty token batch"
        );

        let mut text = String::new();
        for token in tokens {
            if let Some(chunk) = self
                .inner
                .step(token.value())
                .map_err(|err| Error::internal(format!("unable to incrementally decode token: {err}")))?
            {
                text.push_str(&chunk);
            }
        }
        Ok((!text.is_empty()).then_some(text))
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::AddedToken;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::metaspace::Metaspace;
    use tokenizers::models::bpe::BPE;
    use tokenizers::models::wordlevel::WordLevel;

    use super::*;

    #[test]
    fn test_encode_and_decode_tokens() {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("<special>".to_string(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut source = tokenizers::Tokenizer::new(model);
        source
            .add_special_tokens([AddedToken::from("<special>", true)])
            .unwrap();
        let serialized = source.to_string(false).unwrap();
        let tokenizer = HFTokenizer::from_string(&serialized).unwrap();

        assert_eq!(tokenizer.encode("hello").unwrap(), vec![Token::new(1)]);
        assert_eq!(tokenizer.token("<special>"), Some(Token::new(2)));
        assert_eq!(tokenizer.token("missing"), None);
        assert_eq!(tokenizer.decode(&[Token::new(2)]).unwrap(), "<special>");
        assert_eq!(
            tokenizer
                .decode_without_special_tokens(&[Token::new(1), Token::new(2)])
                .unwrap(),
            "hello"
        );
    }

    #[test]
    fn test_incremental_decode_buffers_incomplete_utf8() {
        let tokenizer = byte_fallback_tokenizer();
        let mut decoder = IncrementalDecoder::new(&tokenizer);

        assert_eq!(decoder.decode(&[Token::new(0)]).unwrap(), None);
        assert_eq!(decoder.decode(&[Token::new(1)]).unwrap(), Some("é".to_string()));

        let mut decoder = IncrementalDecoder::new(&tokenizer);
        assert_eq!(
            decoder.decode(&[Token::new(0), Token::new(1)]).unwrap(),
            Some("é".to_string())
        );
    }

    #[test]
    fn test_incremental_decode_preserves_metaspace_context() {
        let tokenizer = metaspace_tokenizer();
        let mut decoder = IncrementalDecoder::new(&tokenizer);

        assert_eq!(decoder.decode(&[Token::new(0)]).unwrap(), Some("This".to_string()));
        assert_eq!(decoder.decode(&[Token::new(0)]).unwrap(), Some(" This".to_string()));
    }

    #[test]
    fn test_incremental_decode_without_special_tokens() {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("<special>".to_string(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut source = tokenizers::Tokenizer::new(model);
        source
            .add_special_tokens([AddedToken::from("<special>", true)])
            .unwrap();
        let tokenizer = HFTokenizer::new(source);
        let mut decoder = IncrementalDecoder::without_special_tokens(&tokenizer);

        assert_eq!(
            decoder.decode(&[Token::new(1), Token::new(2)]).unwrap(),
            Some("hello".to_string())
        );
    }

    fn byte_fallback_tokenizer() -> HFTokenizer {
        let model = BPE::builder()
            .vocab_and_merges([("<0xC3>".to_string(), 0), ("<0xA9>".to_string(), 1)], Vec::new())
            .byte_fallback(true)
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_decoder(Some(ByteFallback::default()));
        HFTokenizer::new(tokenizer)
    }

    fn metaspace_tokenizer() -> HFTokenizer {
        let model = BPE::builder()
            .vocab_and_merges([("▁This".to_string(), 0)], Vec::new())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_decoder(Some(Metaspace::default()));
        HFTokenizer::new(tokenizer)
    }
}
