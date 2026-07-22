use std::cmp::min;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenConsumption {
    Skip,
    Prefill(usize),
    Decode(usize),
}

impl TokenConsumption {
    pub fn token_consumption(&self) -> usize {
        match self {
            TokenConsumption::Skip => 0,
            TokenConsumption::Prefill(consumption) => *consumption,
            TokenConsumption::Decode(consumption) => *consumption,
        }
    }
}

pub fn token_consumption<const L: usize>(
    token_budget: usize,
    num_ready_tokens: usize,
    num_queued_tokens: usize,
    num_spec_tokens: usize,
) -> TokenConsumption {
    debug_assert!(1 <= token_budget);
    debug_assert!(num_spec_tokens < L);
    if 1 == L {
        // noop
    } else {
        // 1 < L
        if num_ready_tokens == 0 {
            debug_assert!(1 <= num_queued_tokens);
        } else {
            debug_assert!(L - 1 <= num_queued_tokens);
        }
    }

    let num_cachable_tokens = num_queued_tokens.saturating_sub(L - 1);
    let cachable_tokens_end = num_ready_tokens + num_cachable_tokens;
    let decode_tokens_start = num_ready_tokens + num_queued_tokens;
    let decode_tokens_end = decode_tokens_start + num_spec_tokens;
    if decode_tokens_end == 0 {
        return TokenConsumption::Skip;
    }

    // ready token | num_queued_tokens
    if token_budget >= decode_tokens_start {
        TokenConsumption::Decode(min(token_budget, decode_tokens_end))
    } else if cachable_tokens_end == 0 {
        TokenConsumption::Skip
    } else {
        TokenConsumption::Prefill(min(token_budget, cachable_tokens_end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_consumption_0() {
        let num_ready_tokens = 1;
        let num_queued_tokens = 3;
        let num_spec_tokens = 0;

        for token_budget in 1..4 {
            assert_eq!(
                TokenConsumption::Prefill(1),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        for token_budget in 4..5 {
            assert_eq!(
                TokenConsumption::Decode(token_budget),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        let token_budget = 5;
        assert_eq!(
            TokenConsumption::Decode(4),
            token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
        );
    }

    #[test]
    fn test_token_consumption_1() {
        let num_ready_tokens = 1;
        let num_queued_tokens = 3;
        let num_spec_tokens = 3;

        for token_budget in 1..4 {
            assert_eq!(
                TokenConsumption::Prefill(1),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        for token_budget in 4..8 {
            assert_eq!(
                TokenConsumption::Decode(token_budget),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        let token_budget = 8;
        assert_eq!(
            TokenConsumption::Decode(7),
            token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
        );
    }

    #[test]
    fn test_token_consumption_2() {
        let num_ready_tokens = 0;
        let num_queued_tokens = 0;
        let num_spec_tokens = 0;

        for token_budget in 1..2 {
            assert_eq!(
                TokenConsumption::Skip,
                token_consumption::<1>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        let token_budget = 2;
        assert_eq!(
            TokenConsumption::Skip,
            token_consumption::<1>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
        );
    }

    #[test]
    fn test_token_consumption_3() {
        let num_ready_tokens = 0;
        let num_queued_tokens = 1;
        let num_spec_tokens = 0;

        for token_budget in 1..2 {
            assert_eq!(
                TokenConsumption::Decode(token_budget),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        let token_budget = 2;
        assert_eq!(
            TokenConsumption::Decode(1),
            token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
        );
    }

    #[test]
    fn test_token_consumption_4() {
        let num_ready_tokens = 0;
        let num_queued_tokens = 1;
        let num_spec_tokens = 3;

        for token_budget in 1..5 {
            assert_eq!(
                TokenConsumption::Decode(token_budget),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        let token_budget = 5;
        assert_eq!(
            TokenConsumption::Decode(4),
            token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
        );
    }

    #[test]
    fn test_token_consumption_5() {
        let num_ready_tokens = 0;
        let num_queued_tokens = 2;
        let num_spec_tokens = 0;

        for token_budget in 1..2 {
            assert_eq!(
                TokenConsumption::Skip,
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        for token_budget in 2..3 {
            assert_eq!(
                TokenConsumption::Decode(token_budget),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        let token_budget = 3;
        assert_eq!(
            TokenConsumption::Decode(2),
            token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
        );
    }

    #[test]
    fn test_token_consumption_6() {
        let num_ready_tokens = 0;
        let num_queued_tokens = 2;
        let num_spec_tokens = 3;

        for token_budget in 1..2 {
            assert_eq!(
                TokenConsumption::Skip,
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        for token_budget in 2..6 {
            assert_eq!(
                TokenConsumption::Decode(token_budget),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        let token_budget = 6;
        assert_eq!(
            TokenConsumption::Decode(5),
            token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
        );
    }

    #[test]
    fn test_token_consumption_7() {
        let num_ready_tokens = 0;
        let num_queued_tokens = 3;
        let num_spec_tokens = 0;

        for token_budget in 1..3 {
            assert_eq!(
                TokenConsumption::Skip,
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        for token_budget in 3..4 {
            assert_eq!(
                TokenConsumption::Decode(token_budget),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        let token_budget = 4;
        assert_eq!(
            TokenConsumption::Decode(3),
            token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
        );
    }

    #[test]
    fn test_token_consumption_8() {
        let num_ready_tokens = 0;
        let num_queued_tokens = 3;
        let num_spec_tokens = 3;

        for token_budget in 1..3 {
            assert_eq!(
                TokenConsumption::Skip,
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        for token_budget in 3..7 {
            assert_eq!(
                TokenConsumption::Decode(token_budget),
                token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
            );
        }

        let token_budget = 7;
        assert_eq!(
            TokenConsumption::Decode(6),
            token_consumption::<4>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens)
        );
    }
}
