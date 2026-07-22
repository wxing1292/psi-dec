use crate::runtime::Token;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QueryTokens {
    Prefill {
        epoch: usize,
        token_index: usize,
        tokens: Vec<Token>,
        window: usize,
    },
    Decode {
        epoch: usize,
        token_index: usize,
        tokens: Vec<Token>,
        spec_tokens: Vec<Token>,
    },
}

impl QueryTokens {
    pub fn epoch(&self) -> usize {
        match self {
            Self::Prefill { epoch, .. } => *epoch,
            Self::Decode { epoch, .. } => *epoch,
        }
    }

    pub fn token_index(&self) -> usize {
        match self {
            Self::Prefill {
                token_index: query_token_index,
                ..
            } => *query_token_index,
            Self::Decode {
                token_index: query_token_index,
                ..
            } => *query_token_index,
        }
    }

    pub fn token_consumption(&self) -> usize {
        match self {
            Self::Prefill { window, .. } => *window,
            Self::Decode {
                tokens: validated_tokens,
                spec_tokens,
                ..
            } => validated_tokens.len() + spec_tokens.len(),
        }
    }

    pub fn num_spec_tokens(&self) -> usize {
        match self {
            Self::Decode { spec_tokens, .. } => spec_tokens.len(),
            Self::Prefill { .. } => 0,
        }
    }

    pub fn token_ids_by_lane(&self, lane: usize) -> Box<dyn Iterator<Item = u32> + '_> {
        match self {
            Self::Prefill { tokens, window, .. } => {
                debug_assert!(
                    *window > 0 && *window <= tokens.len(),
                    "prefill token window must be non-empty and fit token storage"
                );
                debug_assert!(
                    lane <= tokens.len() - window,
                    "prefill token lane exceeds available lookahead"
                );
                Box::new(tokens.iter().skip(lane).take(*window).map(|token| token.value()))
            },
            Self::Decode {
                tokens, spec_tokens, ..
            } => {
                Box::new(
                    tokens
                        .iter()
                        .chain(spec_tokens.iter())
                        .skip(lane)
                        .map(|token| token.value()),
                )
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(value: u32) -> Token {
        Token::new(value)
    }

    #[test]
    fn test_prefill() {
        let query_tokens = QueryTokens::Prefill {
            epoch: 7,
            token_index: 3,
            tokens: vec![token(10), token(11), token(12)],
            window: 2,
        };

        assert_eq!(query_tokens.epoch(), 7);
        assert_eq!(query_tokens.token_index(), 3);
        assert_eq!(query_tokens.token_consumption(), 2);
        assert_eq!(query_tokens.num_spec_tokens(), 0);
        assert_eq!(query_tokens.token_ids_by_lane(0).collect::<Vec<_>>(), vec![10, 11]);
        assert_eq!(query_tokens.token_ids_by_lane(1).collect::<Vec<_>>(), vec![11, 12]);
    }

    #[test]
    fn test_decode() {
        let query_tokens = QueryTokens::Decode {
            epoch: 11,
            token_index: 5,
            tokens: vec![token(10), token(11)],
            spec_tokens: vec![token(20), token(21)],
        };

        assert_eq!(query_tokens.epoch(), 11);
        assert_eq!(query_tokens.token_index(), 5);
        assert_eq!(query_tokens.token_consumption(), 4);
        assert_eq!(query_tokens.num_spec_tokens(), 2);
        assert_eq!(
            query_tokens.token_ids_by_lane(0).collect::<Vec<_>>(),
            vec![10, 11, 20, 21]
        );
        assert_eq!(query_tokens.token_ids_by_lane(1).collect::<Vec<_>>(), vec![11, 20, 21]);
        assert_eq!(query_tokens.token_ids_by_lane(2).collect::<Vec<_>>(), vec![20, 21]);
    }
}
