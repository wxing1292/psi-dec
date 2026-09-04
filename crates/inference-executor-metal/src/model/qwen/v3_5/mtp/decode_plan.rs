//! Pure per-module Qwen3.5 MTP Decode token planning.
//!
//! Let `K` be the number of MTP modules. Let `V` be the number of committed
//! continuation tokens after the first Main input token. Let `m` be a
//! zero-based module index. Let `C` be the cache-local index of the first Main
//! input token. The continuation is
//! `main_input_tokens.skip(1) + validated_tokens`.
//!
//! ```text
//! input_index(m) = C - max(m - V, 0)
//!
//! input_tokens(m) =
//!     continuation_tokens.skip(m)
//!     + [sampled_token]
//!     + draft_tokens.take(m)
//!
//! num_input_rows(m) = max(V, m) + 1
//! ```
//!
//! If a known input replaces the cached proposal prefix, `repair_tail` disables
//! reuse from old x1. Then the start is C - m and every continuation token is input.
//! All modules still end at C + V + 1. Main does not replay any old token.
//!
//! Replacement [w, a, b, c], C = 3:
//! ```text
//! Main @3: w, a, b, c
//! MTP0 @3: a, b, c, y
//! MTP1 @2: a, b, c, y, z1
//! MTP2 @1: a, b, c, y, z1, z2
//! ```
//!
//! The examples use `K = 3` and `C = 3`. The previous proposal is
//! `[x1, x2, x3]`. Main verifies `[w, x1, x2, x3]` and samples `y`.
//!
//! Full reject, `V = 0`:
//!
//! ```text
//! +--------+-------------+--------------+
//! | Module | Input index | Input tokens |
//! +--------+-------------+--------------+
//! | MTP0   | 3           | y            |
//! | MTP1   | 2           | y, z1        |
//! | MTP2   | 1           | y, z1, z2    |
//! +--------+-------------+--------------+
//! ```
//!
//! Accept `x1`, `V = 1`:
//!
//! ```text
//! +--------+-------------+----------------+
//! | Module | Input index | Input tokens   |
//! +--------+-------------+----------------+
//! | MTP0   | 3           | x1, y          |
//! | MTP1   | 3           | y, z1          |
//! | MTP2   | 2           | y, z1, z2      |
//! +--------+-------------+----------------+
//! ```
//!
//! Accept `x1`, `x2`, and `x3`, `V = 3`:
//!
//! ```text
//! +--------+-------------+----------------+
//! | Module | Input index | Input tokens   |
//! +--------+-------------+----------------+
//! | MTP0   | 3           | x1, x2, x3, y  |
//! | MTP1   | 3           | x2, x3, y, z1  |
//! | MTP2   | 3           | x3, y, z1, z2  |
//! +--------+-------------+----------------+
//! ```

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35MTPDecodeTokenSource {
    Continuation { token_offset: usize },
    Sampled,
    Draft { step_index: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MTPDecodePlan {
    num_spec_tokens: usize,
    num_continuation_tokens: usize,
    repair_tail: bool,
}

impl Qwen35MTPDecodePlan {
    pub fn new(num_spec_tokens: usize, num_continuation_tokens: usize, repair_tail: bool) -> Self {
        debug_assert!(num_spec_tokens > 0);
        Self {
            num_spec_tokens,
            num_continuation_tokens,
            repair_tail,
        }
    }

    pub fn module(self, module_index: usize) -> Qwen35MTPDecodeModulePlan {
        debug_assert!(module_index < self.num_spec_tokens);
        Qwen35MTPDecodeModulePlan {
            num_continuation_tokens: self.num_continuation_tokens,
            module_index,
            repair_tail: self.repair_tail,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MTPDecodeModulePlan {
    num_continuation_tokens: usize,
    module_index: usize,
    repair_tail: bool,
}

impl Qwen35MTPDecodeModulePlan {
    pub fn num_input_rows(self) -> usize {
        self.num_continuation_tokens + 1 + self.module_index - self.num_reused_tokens()
    }

    pub fn num_reused_tokens(self) -> usize {
        if self.repair_tail {
            0
        } else {
            self.num_continuation_tokens.min(self.module_index)
        }
    }

    pub fn token_index(self, pending_token_index: u32) -> u32 {
        let rewind = self.module_index - self.num_reused_tokens();
        pending_token_index
            .checked_sub(rewind as u32)
            .expect("qwen3.5 MTP Decode requires initialized hidden-state cache history")
    }

    pub fn token_source(self, input_row_offset: usize) -> Qwen35MTPDecodeTokenSource {
        debug_assert!(input_row_offset < self.num_input_rows());
        let num_continuation_rows = self.num_continuation_tokens - self.num_reused_tokens();
        if input_row_offset < num_continuation_rows {
            return Qwen35MTPDecodeTokenSource::Continuation {
                token_offset: self.num_reused_tokens() + input_row_offset,
            };
        }
        if input_row_offset == num_continuation_rows {
            return Qwen35MTPDecodeTokenSource::Sampled;
        }
        Qwen35MTPDecodeTokenSource::Draft {
            step_index: input_row_offset - num_continuation_rows - 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replacement_recomputes_from_x1_and_preserves_the_common_end() {
        for num_continuation_tokens in 0..8 {
            for module_index in 0..3 {
                let module = Qwen35MTPDecodePlan::new(3, num_continuation_tokens, true).module(module_index);
                assert_eq!(module.num_reused_tokens(), 0);
                assert_eq!(module.token_index(3), 3 - module_index as u32);
                assert_eq!(module.num_input_rows(), num_continuation_tokens + module_index + 1);
                assert_eq!(
                    module.token_index(3) + module.num_input_rows() as u32,
                    4 + num_continuation_tokens as u32
                );
                for token_offset in 0..num_continuation_tokens {
                    assert_eq!(
                        module.token_source(token_offset),
                        Qwen35MTPDecodeTokenSource::Continuation { token_offset }
                    );
                }
                assert_eq!(
                    module.token_source(num_continuation_tokens),
                    Qwen35MTPDecodeTokenSource::Sampled
                );
                for step_index in 0..module_index {
                    assert_eq!(
                        module.token_source(num_continuation_tokens + 1 + step_index),
                        Qwen35MTPDecodeTokenSource::Draft { step_index }
                    );
                }
            }
        }
    }

    #[test]
    fn test_all_reject_input_indices_and_tokens() {
        let plan = Qwen35MTPDecodePlan::new(3, 0, false);
        assert_module(plan.module(0), 3, 3, &[Qwen35MTPDecodeTokenSource::Sampled]);
        assert_module(
            plan.module(1),
            3,
            2,
            &[
                Qwen35MTPDecodeTokenSource::Sampled,
                Qwen35MTPDecodeTokenSource::Draft { step_index: 0 },
            ],
        );
        assert_module(
            plan.module(2),
            3,
            1,
            &[
                Qwen35MTPDecodeTokenSource::Sampled,
                Qwen35MTPDecodeTokenSource::Draft { step_index: 0 },
                Qwen35MTPDecodeTokenSource::Draft { step_index: 1 },
            ],
        );
    }

    #[test]
    fn test_partial_accept_input_indices_and_tokens() {
        let plan = Qwen35MTPDecodePlan::new(3, 1, false);
        assert_module(
            plan.module(0),
            3,
            3,
            &[
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 0 },
                Qwen35MTPDecodeTokenSource::Sampled,
            ],
        );
        assert_module(
            plan.module(1),
            3,
            3,
            &[
                Qwen35MTPDecodeTokenSource::Sampled,
                Qwen35MTPDecodeTokenSource::Draft { step_index: 0 },
            ],
        );
        assert_module(
            plan.module(2),
            3,
            2,
            &[
                Qwen35MTPDecodeTokenSource::Sampled,
                Qwen35MTPDecodeTokenSource::Draft { step_index: 0 },
                Qwen35MTPDecodeTokenSource::Draft { step_index: 1 },
            ],
        );
    }

    #[test]
    fn test_full_accept_input_indices_and_tokens() {
        let plan = Qwen35MTPDecodePlan::new(3, 3, false);
        assert_module(
            plan.module(0),
            3,
            3,
            &[
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 0 },
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 1 },
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 2 },
                Qwen35MTPDecodeTokenSource::Sampled,
            ],
        );
        assert_module(
            plan.module(1),
            3,
            3,
            &[
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 1 },
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 2 },
                Qwen35MTPDecodeTokenSource::Sampled,
                Qwen35MTPDecodeTokenSource::Draft { step_index: 0 },
            ],
        );
        assert_module(
            plan.module(2),
            3,
            3,
            &[
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 2 },
                Qwen35MTPDecodeTokenSource::Sampled,
                Qwen35MTPDecodeTokenSource::Draft { step_index: 0 },
                Qwen35MTPDecodeTokenSource::Draft { step_index: 1 },
            ],
        );
    }

    #[test]
    fn test_prompt_tail_extends_the_continuation_beyond_the_proposal_width() {
        let plan = Qwen35MTPDecodePlan::new(3, 5, false);
        assert_module(
            plan.module(2),
            7,
            7,
            &[
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 2 },
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 3 },
                Qwen35MTPDecodeTokenSource::Continuation { token_offset: 4 },
                Qwen35MTPDecodeTokenSource::Sampled,
                Qwen35MTPDecodeTokenSource::Draft { step_index: 0 },
                Qwen35MTPDecodeTokenSource::Draft { step_index: 1 },
            ],
        );
    }

    #[test]
    fn test_all_continuation_lengths_finish_at_one_cache_local_index() {
        let num_spec_tokens = 5;
        let pending_token_index = 11;
        for num_continuation_tokens in 0..=8 {
            let plan = Qwen35MTPDecodePlan::new(num_spec_tokens, num_continuation_tokens, false);
            for module_index in 0..num_spec_tokens {
                let module = plan.module(module_index);
                assert_eq!(module.num_input_rows(), num_continuation_tokens.max(module_index) + 1);
                assert_eq!(
                    module.token_index(pending_token_index),
                    pending_token_index - module_index.saturating_sub(num_continuation_tokens) as u32
                );
                assert_eq!(
                    module.token_index(pending_token_index) + module.num_input_rows() as u32,
                    pending_token_index + num_continuation_tokens as u32 + 1
                );
            }
        }
    }

    fn assert_module(
        module: Qwen35MTPDecodeModulePlan,
        pending_token_index: u32,
        expected_token_index: u32,
        expected_tokens: &[Qwen35MTPDecodeTokenSource],
    ) {
        assert_eq!(module.token_index(pending_token_index), expected_token_index);
        assert_eq!(module.num_input_rows(), expected_tokens.len());
        for (row_offset, &expected) in expected_tokens.iter().enumerate() {
            assert_eq!(module.token_source(row_offset), expected);
        }
    }
}
