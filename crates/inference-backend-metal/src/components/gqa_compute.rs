use crate::metal::Dtype;

/// A backend-selected GQA compute path and its complete SDPA geometry.
///
/// Production callers provide workload facts to [`GQACompute`]. They do not
/// select a path, kernel family, or tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GQAComputePath {
    SingleQueryToken {
        kv_token_tile_size: u32,
        num_threads_per_threadblock: u32,
        q_head_tile_size: u32,
    },
    TiledQueryTokens {
        q_token_tile_size: u32,
        kv_token_tile_size: u32,
        q_head_tile_size: u32,
    },
}

/// Fixed workload facts used to select a GQA compute path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQAComputeConfig {
    pub io_dtype: Dtype,
    pub page_bytes: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
}

impl GQAComputeConfig {
    pub fn validate(self) {
        assert!(matches!(self.io_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(self.page_bytes > 0);
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.head_dim > 0);
        let _ = self.num_tokens_per_page();
    }

    pub fn q_heads_per_kv_head(self) -> u32 {
        self.num_q_heads / self.num_kv_heads
    }

    pub fn num_tokens_per_page(self) -> u32 {
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|elements| {
                elements.checked_mul(
                    self.io_dtype
                        .item_size()
                        .try_into()
                        .expect("io_dtype size must fit u32"),
                )
            })
            .expect("GQA K/V bytes per token must fit u32");
        assert!(
            self.page_bytes.is_multiple_of(kv_bytes_per_token),
            "GQA page_bytes must be divisible by the K/V bytes per token"
        );
        self.page_bytes / kv_bytes_per_token
    }
}

/// Backend-owned GQA compute-path selector.
///
/// `SingleQueryToken` and `TiledQueryTokens` use different metadata and
/// kernels. This selector also returns the complete SDPA geometry that the
/// selected metadata builder and kernel implementation share.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQACompute {
    q_heads_per_kv_head: u32,
    supports_tiled_query_tokens: bool,
    full_q_head_group_for_tiled: bool,
    single_query_token_kv_token_tile_size: u32,
    single_query_token_num_threads_per_threadblock: u32,
    tiled_query_token_tile_size: u32,
    tiled_kv_token_tile_size: u32,
    max_q_head_tile_size: u32,
}

impl GQACompute {
    pub fn new(config: GQAComputeConfig) -> Self {
        config.validate();
        let tiled_profile = (config.head_dim, config.num_tokens_per_page());
        let supports_tiled_query_tokens = config.io_dtype == Dtype::Bfloat16
            && matches!(tiled_profile, (128, 8) | (256, 16))
            && config.q_heads_per_kv_head() <= 8;
        Self::with_tiled_profile(config, supports_tiled_query_tokens, tiled_profile == (128, 8))
    }

    /// Selects the single-query-token partial ABI required by a larger
    /// DSpark history composition. The paged-history and block-bidirectional
    /// maps must produce the same partial layout for one shared reduce.
    pub fn new_dspark_history(config: GQAComputeConfig) -> Self {
        config.validate();
        Self::with_tiled_profile(config, false, false)
    }

    fn with_tiled_profile(
        config: GQAComputeConfig,
        supports_tiled_query_tokens: bool,
        full_q_head_group_for_tiled: bool,
    ) -> Self {
        let bounded_head_dim = config.head_dim.min(256);
        let single_query_token_num_threads_per_threadblock = bounded_head_dim.max(32).next_power_of_two();
        let tiled_query_token_tile_size = 8;
        Self {
            q_heads_per_kv_head: config.q_heads_per_kv_head(),
            supports_tiled_query_tokens,
            full_q_head_group_for_tiled,
            single_query_token_kv_token_tile_size: single_query_token_num_threads_per_threadblock,
            single_query_token_num_threads_per_threadblock,
            tiled_query_token_tile_size,
            tiled_kv_token_tile_size: 16,
            max_q_head_tile_size: 256 / (tiled_query_token_tile_size / 8 * 32),
        }
    }

    pub fn tiled_query_token_tile_size(self) -> u32 {
        self.tiled_query_token_tile_size
    }

    pub fn max_query_tokens_per_partial_output(self) -> u32 {
        if self.supports_tiled_query_tokens {
            self.tiled_query_token_tile_size
        } else {
            1
        }
    }

    pub fn select(self, num_tokens: u32, num_query_token_tiles: u32) -> GQAComputePath {
        assert!(num_tokens > 0);
        assert!(num_query_token_tiles > 0 && num_query_token_tiles <= num_tokens);
        if !self.supports_tiled_query_tokens || (num_tokens as u64) < 2 * num_query_token_tiles as u64 {
            return GQAComputePath::SingleQueryToken {
                kv_token_tile_size: self.single_query_token_kv_token_tile_size,
                num_threads_per_threadblock: self.single_query_token_num_threads_per_threadblock,
                q_head_tile_size: self.q_heads_per_kv_head.min(self.max_q_head_tile_size),
            };
        }
        let desired_q_head_tile_size =
            if !self.full_q_head_group_for_tiled && (num_tokens as u64) < 4 * num_query_token_tiles as u64 {
                self.q_heads_per_kv_head.div_ceil(2)
            } else {
                self.q_heads_per_kv_head
            };
        GQAComputePath::TiledQueryTokens {
            q_token_tile_size: self.tiled_query_token_tile_size,
            kv_token_tile_size: self.tiled_kv_token_tile_size,
            q_head_tile_size: desired_q_head_tile_size.min(self.max_q_head_tile_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GQACompute;
    use super::GQAComputeConfig;
    use super::GQAComputePath;
    use crate::metal::Dtype;

    fn config(head_dim: u32, q_heads_per_kv_head: u32, num_tokens_per_page: u32) -> GQAComputeConfig {
        let num_kv_heads = 2;
        GQAComputeConfig {
            io_dtype: Dtype::Bfloat16,
            page_bytes: 2 * num_kv_heads * head_dim * num_tokens_per_page * Dtype::Bfloat16.item_size() as u32,
            num_q_heads: num_kv_heads * q_heads_per_kv_head,
            num_kv_heads,
            head_dim,
        }
    }

    #[test]
    fn test_selects_single_and_tiled_query_paths() {
        let compute = GQACompute::new(config(256, 6, 16));
        assert_eq!(compute.max_query_tokens_per_partial_output(), 8);
        assert_eq!(
            compute.select(4, 4),
            GQAComputePath::SingleQueryToken {
                kv_token_tile_size: 256,
                num_threads_per_threadblock: 256,
                q_head_tile_size: 6,
            }
        );
        assert_eq!(
            compute.select(8, 4),
            GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 8,
                kv_token_tile_size: 16,
                q_head_tile_size: 3,
            }
        );
        assert_eq!(
            compute.select(16, 4),
            GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 8,
                kv_token_tile_size: 16,
                q_head_tile_size: 6,
            }
        );
    }

    #[test]
    fn test_full_query_head_group_profile() {
        let compute = GQACompute::new(config(128, 8, 8));
        assert_eq!(compute.max_query_tokens_per_partial_output(), 8);
        assert_eq!(
            compute.select(8, 4),
            GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 8,
                kv_token_tile_size: 16,
                q_head_tile_size: 8,
            }
        );
    }

    #[test]
    fn test_unsupported_profile_uses_single_query_token_path() {
        let compute = GQACompute::new(config(256, 8, 8));
        assert_eq!(compute.max_query_tokens_per_partial_output(), 1);
        assert!(matches!(compute.select(16, 4), GQAComputePath::SingleQueryToken { .. }));
    }

    #[test]
    fn test_single_query_token_partial_constraint_disables_tiled_layout() {
        let compute = GQACompute::new_dspark_history(config(128, 8, 8));
        assert_eq!(compute.max_query_tokens_per_partial_output(), 1);
        assert!(matches!(compute.select(16, 4), GQAComputePath::SingleQueryToken { .. }));
    }

    #[test]
    #[should_panic(expected = "GQA K/V bytes per token must fit u32")]
    fn test_config_rejects_kv_byte_overflow() {
        GQAComputeConfig {
            io_dtype: Dtype::Float32,
            page_bytes: 65536,
            num_q_heads: u32::MAX,
            num_kv_heads: u32::MAX,
            head_dim: 2,
        }
        .validate();
    }
}
