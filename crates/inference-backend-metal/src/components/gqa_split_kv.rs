use crate::metal::Dtype;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GQASDPAMapThreadBlockSpecialization {
    pub max_q_tokens: u32,
    pub max_q_heads: u32,
    pub kv_tokens_per_iteration: u32,
    pub required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GQASDPAKVCacheSpecialization {
    pub tokens_per_page: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GQASDPAMapKernelSpecialization {
    pub thread_block: GQASDPAMapThreadBlockSpecialization,
    pub kv_cache: GQASDPAKVCacheSpecialization,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GQASDPAReduceThreadBlockSpecialization {
    pub max_q_tokens: u32,
    pub max_q_heads: u32,
    pub required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GQASDPAReduceKernelSpecialization {
    pub thread_block: GQASDPAReduceThreadBlockSpecialization,
}

/// One compatible GQA SDPA Map and Reduce kernel specialization pair.
///
/// `map.thread_block.max_q_tokens == 1` describes the current SingleQ geometry. A larger value
/// describes a current TiledQ geometry. The planner does not expose a
/// SingleQ/TiledQ execution enum.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GQASDPAExecutionSpecialization {
    pub map: GQASDPAMapKernelSpecialization,
    pub reduce: GQASDPAReduceKernelSpecialization,
}

impl GQASDPAExecutionSpecialization {
    pub fn single_q(
        config: GQASDPAConfig,
        kv_tokens_per_iteration: u32,
        required_threads: u32,
        max_q_heads: u32,
    ) -> Self {
        config.validate();
        let reduce_output_coordinates = 256 / config.head_dim.min(256);
        let reduce_max_q_heads = config.num_q_heads.min(reduce_output_coordinates).max(1);
        let reduce_max_q_tokens = (reduce_output_coordinates / reduce_max_q_heads).max(1);
        let execution = Self {
            map: GQASDPAMapKernelSpecialization {
                thread_block: GQASDPAMapThreadBlockSpecialization {
                    max_q_tokens: 1,
                    max_q_heads,
                    kv_tokens_per_iteration,
                    required_threads,
                },
                kv_cache: GQASDPAKVCacheSpecialization {
                    tokens_per_page: config.tokens_per_page,
                },
            },
            reduce: GQASDPAReduceKernelSpecialization {
                thread_block: GQASDPAReduceThreadBlockSpecialization {
                    max_q_tokens: reduce_max_q_tokens,
                    max_q_heads: reduce_max_q_heads,
                    required_threads: 256,
                },
            },
        };
        assert!(execution.supports(config));
        execution
    }

    pub fn tiled_q(config: GQASDPAConfig, max_q_tokens: u32, kv_tokens_per_iteration: u32, max_q_heads: u32) -> Self {
        config.validate();
        let required_threads = max_q_tokens / 8 * max_q_heads * 32;
        let execution = Self {
            map: GQASDPAMapKernelSpecialization {
                thread_block: GQASDPAMapThreadBlockSpecialization {
                    max_q_tokens,
                    max_q_heads,
                    kv_tokens_per_iteration,
                    required_threads,
                },
                kv_cache: GQASDPAKVCacheSpecialization {
                    tokens_per_page: config.tokens_per_page,
                },
            },
            reduce: GQASDPAReduceKernelSpecialization {
                thread_block: GQASDPAReduceThreadBlockSpecialization {
                    max_q_tokens,
                    max_q_heads: 1,
                    required_threads,
                },
            },
        };
        assert!(execution.supports(config));
        execution
    }

    pub fn supports(self, config: GQASDPAConfig) -> bool {
        let map = self.map.thread_block;
        let reduce = self.reduce.thread_block;
        if self.map.kv_cache.tokens_per_page != config.tokens_per_page
            || map.max_q_tokens == 0
            || map.max_q_heads == 0
            || map.max_q_heads > config.q_heads_per_kv_head()
            || map.kv_tokens_per_iteration == 0
            || map.required_threads == 0
            || reduce.max_q_tokens == 0
            || reduce.max_q_heads == 0
            || reduce.required_threads == 0
        {
            return false;
        }

        if map.max_q_tokens == 1 {
            let reduce_output_coordinates = 256 / config.head_dim.min(256);
            let expected_reduce_max_q_heads = config.num_q_heads.min(reduce_output_coordinates).max(1);
            let expected_reduce_max_q_tokens = (reduce_output_coordinates / expected_reduce_max_q_heads).max(1);
            return map.max_q_heads <= 8
                && map.kv_tokens_per_iteration <= 1024
                && map.required_threads.is_power_of_two()
                && map.required_threads <= 256
                && reduce.max_q_tokens == expected_reduce_max_q_tokens
                && reduce.max_q_heads == expected_reduce_max_q_heads
                && reduce.required_threads == 256;
        }

        let profile = (config.head_dim, config.tokens_per_page);
        config.io_dtype == Dtype::Bfloat16
            && matches!(profile, (128, 8) | (256, 8 | 16))
            && config.q_heads_per_kv_head() <= 8
            && matches!(map.max_q_tokens, 8 | 16)
            && matches!(map.kv_tokens_per_iteration, 8 | 16)
            && map.required_threads == map.max_q_tokens / 8 * map.max_q_heads * 32
            && map.required_threads <= 256
            && reduce.max_q_tokens == map.max_q_tokens
            && reduce.max_q_heads == 1
            && reduce.required_threads == map.required_threads
    }
}

/// Static GQA SDPA workload facts used for capability filtering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQASDPAConfig {
    pub io_dtype: Dtype,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub tokens_per_page: u32,
}

impl GQASDPAConfig {
    pub fn validate(self) {
        assert!(matches!(self.io_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.head_dim > 0);
        assert!(self.tokens_per_page > 0);
    }

    pub fn q_heads_per_kv_head(self) -> u32 {
        self.num_q_heads / self.num_kv_heads
    }
}

/// Backend-owned registry of legal GQA SDPA kernel specializations.
///
/// The registry performs only static capability filtering. The executor
/// planner owns dynamic workload selection and complete-plan comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GQASDPASpecializationRegistry {
    config: GQASDPAConfig,
    executions: Box<[GQASDPAExecutionSpecialization]>,
}

impl GQASDPASpecializationRegistry {
    pub fn new(config: GQASDPAConfig) -> Self {
        config.validate();
        let mut executions = vec![single_q_execution(config)];
        if supports_tiled_q(config) {
            let full_q_heads = config.q_heads_per_kv_head().min(max_tiled_q_heads(8));
            if (config.head_dim, config.tokens_per_page) != (128, 8) {
                let half_q_heads = config.q_heads_per_kv_head().div_ceil(2).min(max_tiled_q_heads(8));
                if half_q_heads != full_q_heads {
                    executions.push(tiled_q_execution(config, half_q_heads));
                }
            }
            executions.push(tiled_q_execution(config, full_q_heads));
        }
        Self::from_execution_specializations(config, executions)
    }

    /// Creates a registry that contains only the current SingleQ execution.
    /// Composite operators can use this registry when all of their Map kernels
    /// must share the SingleQ partial-state ABI and one Reduce kernel.
    pub fn new_single_q_only(config: GQASDPAConfig) -> Self {
        config.validate();
        Self::from_execution_specializations(config, vec![single_q_execution(config)])
    }

    /// Creates a registry from explicit backend specializations.
    ///
    /// Backend benchmarks can use this constructor to force one legal
    /// specialization. Production model plans use [`Self::new`] or
    /// [`Self::new_single_q_only`].
    pub fn from_execution_specializations(
        config: GQASDPAConfig,
        executions: Vec<GQASDPAExecutionSpecialization>,
    ) -> Self {
        config.validate();
        assert!(!executions.is_empty(), "GQA SDPA registry requires an execution");
        assert!(
            executions.iter().all(|execution| execution.supports(config)),
            "GQA SDPA registry contains an unsupported execution specialization"
        );
        Self {
            config,
            executions: executions.into_boxed_slice(),
        }
    }

    pub fn config(&self) -> GQASDPAConfig {
        self.config
    }

    pub fn execution_specializations(&self) -> &[GQASDPAExecutionSpecialization] {
        &self.executions
    }

    pub fn max_q_tokens_per_map_task(&self) -> u32 {
        self.executions
            .iter()
            .map(|execution| execution.map.thread_block.max_q_tokens)
            .max()
            .expect("GQA SDPA registry requires an execution")
    }
}

fn supports_tiled_q(config: GQASDPAConfig) -> bool {
    let profile = (config.head_dim, config.tokens_per_page);
    config.io_dtype == Dtype::Bfloat16
        && matches!(profile, (128, 8) | (256, 8 | 16))
        && config.q_heads_per_kv_head() <= 8
}

fn single_q_execution(config: GQASDPAConfig) -> GQASDPAExecutionSpecialization {
    let required_threads = config.head_dim.clamp(32, 256).next_power_of_two();
    let max_q_heads = config.q_heads_per_kv_head().min(8);
    GQASDPAExecutionSpecialization::single_q(config, required_threads, required_threads, max_q_heads)
}

fn tiled_q_execution(config: GQASDPAConfig, max_q_heads: u32) -> GQASDPAExecutionSpecialization {
    let max_q_tokens = 8;
    GQASDPAExecutionSpecialization::tiled_q(config, max_q_tokens, 16, max_q_heads)
}

fn max_tiled_q_heads(max_q_tokens: u32) -> u32 {
    256 / (max_q_tokens / 8 * 32)
}

#[cfg(test)]
mod tests {
    use super::GQASDPAConfig;
    use super::GQASDPASpecializationRegistry;
    use crate::metal::Dtype;

    fn config(head_dim: u32, q_heads_per_kv_head: u32, tokens_per_page: u32) -> GQASDPAConfig {
        let num_kv_heads = 2;
        GQASDPAConfig {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: num_kv_heads * q_heads_per_kv_head,
            num_kv_heads,
            head_dim,
            tokens_per_page,
        }
    }

    #[test]
    fn test_registry_filters_static_execution_capabilities() {
        let d256_page16 = GQASDPASpecializationRegistry::new(config(256, 6, 16));
        let d256_page8 = GQASDPASpecializationRegistry::new(config(256, 6, 8));
        let d128_page8 = GQASDPASpecializationRegistry::new(config(128, 8, 8));
        let unsupported = GQASDPASpecializationRegistry::new(config(256, 8, 4));
        let dspark = GQASDPASpecializationRegistry::new_single_q_only(config(128, 8, 8));

        assert_eq!(d256_page16.execution_specializations().len(), 3);
        assert_eq!(d256_page8.execution_specializations().len(), 3);
        assert_eq!(d128_page8.execution_specializations().len(), 2);
        assert_eq!(unsupported.execution_specializations().len(), 1);
        assert_eq!(dspark.execution_specializations().len(), 1);
        for registry in [d256_page16, d256_page8, d128_page8, unsupported, dspark] {
            assert!(
                registry
                    .execution_specializations()
                    .iter()
                    .all(|execution| execution.supports(registry.config()))
            );
        }
    }

    #[test]
    fn test_registry_uses_specialization_geometry_without_a_path_enum() {
        let registry = GQASDPASpecializationRegistry::new(config(256, 6, 8));
        let executions = registry.execution_specializations();

        assert_eq!(executions[0].map.thread_block.max_q_tokens, 1);
        assert_eq!(executions[0].map.thread_block.max_q_heads, 6);
        assert_eq!(executions[0].map.thread_block.kv_tokens_per_iteration, 256);
        assert_eq!(executions[0].map.thread_block.required_threads, 256);
        assert_eq!(executions[1].map.thread_block.max_q_tokens, 8);
        assert_eq!(executions[1].map.thread_block.max_q_heads, 3);
        assert_eq!(executions[2].map.thread_block.max_q_heads, 6);
        assert_eq!(registry.max_q_tokens_per_map_task(), 8);
    }
}
