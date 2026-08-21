use std::cmp::Reverse;
use std::ops::Range;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_executor_core::attn::GQAReplayShape;

use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;

const D256_PAGE8_MIN_SELECTION_SCORE: u64 = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestShape {
    pub num_history_tokens: u32,
    pub num_q_tokens: u32,
}

impl RequestShape {
    pub fn from_batch(token_indices: &[u32], cu_tokens: &[u32]) -> Vec<Self> {
        assert_eq!(cu_tokens.len(), token_indices.len() + 1);
        assert_eq!(cu_tokens.first().copied(), Some(0));
        token_indices
            .iter()
            .zip(cu_tokens.windows(2))
            .map(|(&num_history_tokens, cu)| {
                assert!(cu[0] <= cu[1], "GQA batch cu_tokens must be nondecreasing");
                let num_q_tokens = cu[1] - cu[0];
                assert!(num_q_tokens > 0, "GQA requests must contain Q tokens");
                num_history_tokens
                    .checked_add(num_q_tokens)
                    .expect("GQA request context length must fit u32");
                Self {
                    num_history_tokens,
                    num_q_tokens,
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectorLimits {
    pub max_map_task_templates: u32,
    pub partial_state_group_capacity: usize,
    pub max_active_partial_state_groups: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QTokenRange {
    pub request_index: u32,
    pub flat_q_token_indices: Range<u32>,
    pub max_visible_kv_tokens: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MapTaskTemplate {
    pub q_token_range_index: u32,
    pub request_local_kv_token_indices: Range<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectionMetrics {
    pub num_scheduled_qk_token_pairs: u64,
    pub num_active_qk_token_pairs: u64,
    pub num_map_threadblocks_per_kv_head: u64,
    pub num_map_simdgroup_waves_per_kv_head: u64,
    pub num_active_partial_state_groups: u64,
    pub num_active_partial_states: u64,
    pub num_reserved_partial_state_groups: u64,
    pub num_replay_reserved_partial_state_groups: u64,
    pub max_kv_iterations_per_map_task: u32,
    pub num_logical_qk_token_pairs: u64,
}

impl SelectionMetrics {
    fn selection_overhead_score(self) -> u64 {
        self.num_map_threadblocks_per_kv_head
            .checked_add(self.num_map_simdgroup_waves_per_kv_head)
            .and_then(|score| score.checked_add(self.num_active_partial_state_groups))
            .and_then(|score| score.checked_add(self.num_replay_reserved_partial_state_groups))
            .expect("GQA SDPA selection overhead score must fit u64")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Selection {
    variant: backend_sdpa::ExecutionVariant,
    q_token_ranges: Vec<QTokenRange>,
    map_task_templates: Vec<MapTaskTemplate>,
    cu_partial_outputs_by_q_token_range: Vec<u32>,
    replay_shape: GQAReplayShape,
    metrics: SelectionMetrics,
}

impl Selection {
    pub fn variant(&self) -> backend_sdpa::ExecutionVariant {
        self.variant
    }

    pub fn q_token_ranges(&self) -> &[QTokenRange] {
        &self.q_token_ranges
    }

    pub fn map_task_templates(&self) -> &[MapTaskTemplate] {
        &self.map_task_templates
    }

    pub fn cu_partial_outputs_by_q_token_range(&self) -> &[u32] {
        &self.cu_partial_outputs_by_q_token_range
    }

    pub fn replay_shape(&self) -> GQAReplayShape {
        self.replay_shape
    }

    pub fn metrics(&self) -> SelectionMetrics {
        self.metrics
    }
}

#[derive(Clone, Debug)]
pub struct Selector {
    registry: backend_sdpa::Registry,
    limits: SelectorLimits,
}

impl Selector {
    pub fn new(registry: backend_sdpa::Registry, max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "GQA SDPA selector requires tokens");
        let max_map_task_templates = max_tokens
            .try_into()
            .expect("GQA SDPA Map task-template capacity must fit u32");
        let partial_state_group_capacity = max_tokens
            .checked_mul(registry.max_q_tokens_per_map_task() as usize)
            .expect("GQA SDPA partial-state-group capacity must fit usize");
        Self {
            registry,
            limits: SelectorLimits {
                max_map_task_templates,
                partial_state_group_capacity,
                max_active_partial_state_groups: max_map_task_templates,
            },
        }
    }

    pub fn limits(&self) -> SelectorLimits {
        self.limits
    }

    pub fn select(
        &self,
        request_shapes: &[RequestShape],
        policy: &GQAReplayBucketPolicy,
        num_total_tokens: u32,
    ) -> Selection {
        self.validate_policy(policy);
        assert!(!request_shapes.is_empty(), "GQA SDPA selector requires requests");
        let num_tokens = request_shapes
            .iter()
            .try_fold(0u32, |total, shape| total.checked_add(shape.num_q_tokens))
            .expect("GQA SDPA token count must fit u32");
        assert!(num_tokens > 0);
        assert!(num_tokens <= self.limits.max_map_task_templates);
        assert!(
            num_tokens <= num_total_tokens,
            "GQA active token count must not exceed the total token count"
        );
        assert!(
            num_total_tokens <= policy.max_tokens(),
            "GQA total token count must not exceed the metadata capacity"
        );

        let candidates = self
            .registry
            .variants()
            .iter()
            .map(|&variant| {
                materialize_candidate(
                    variant,
                    self.registry.config(),
                    request_shapes,
                    self.limits,
                    policy,
                    num_total_tokens,
                )
            })
            .collect::<Vec<_>>();
        select_candidate(self.registry.config(), candidates)
    }

    fn validate_policy(&self, policy: &GQAReplayBucketPolicy) {
        assert_eq!(
            policy.max_tokens(),
            self.limits.max_map_task_templates,
            "GQA replay bucket policy must match the SDPA selection capacity"
        );
    }
}

#[derive(Clone)]
struct QTokenRangeWork {
    q_token_range: QTokenRange,
    num_kv_iterations: u32,
    num_map_tasks: u32,
}

fn materialize_candidate(
    variant: backend_sdpa::ExecutionVariant,
    config: backend_sdpa::Config,
    request_shapes: &[RequestShape],
    limits: SelectorLimits,
    policy: &GQAReplayBucketPolicy,
    num_total_tokens: u32,
) -> Selection {
    assert!(variant.supports(config));
    let map = variant.map.thread_block;
    let mut ranges = build_q_token_ranges(request_shapes, map.max_q_tokens, map.kv_tokens_per_iteration);
    allocate_map_tasks(&mut ranges, limits);

    let mut q_token_ranges = Vec::with_capacity(ranges.len());
    let mut map_task_templates = Vec::new();
    let mut cu_partial_outputs_by_q_token_range = Vec::with_capacity(ranges.len() + 1);
    cu_partial_outputs_by_q_token_range.push(0);
    let mut max_kv_iterations_per_map_task = 0;
    for (q_token_range_index, range) in ranges.iter().enumerate() {
        q_token_ranges.push(range.q_token_range.clone());
        for map_task_index in 0..range.num_map_tasks {
            let kv_iteration_begin =
                range.num_kv_iterations as u64 * u64::from(map_task_index) / u64::from(range.num_map_tasks);
            let kv_iteration_end =
                range.num_kv_iterations as u64 * u64::from(map_task_index + 1) / u64::from(range.num_map_tasks);
            let num_kv_iterations = (kv_iteration_end - kv_iteration_begin)
                .try_into()
                .expect("GQA SDPA KV-iteration count must fit u32");
            max_kv_iterations_per_map_task = max_kv_iterations_per_map_task.max(num_kv_iterations);
            let kv_token_begin = kv_iteration_begin
                .checked_mul(u64::from(map.kv_tokens_per_iteration))
                .and_then(|value| value.try_into().ok())
                .expect("GQA SDPA KV-token begin must fit u32");
            let kv_token_end = range.q_token_range.max_visible_kv_tokens.min(
                kv_iteration_end
                    .checked_mul(u64::from(map.kv_tokens_per_iteration))
                    .and_then(|value| value.try_into().ok())
                    .expect("GQA SDPA KV-token end must fit u32"),
            );
            map_task_templates.push(MapTaskTemplate {
                q_token_range_index: q_token_range_index
                    .try_into()
                    .expect("GQA SDPA Q-token-range index must fit u32"),
                request_local_kv_token_indices: kv_token_begin..kv_token_end,
            });
        }
        cu_partial_outputs_by_q_token_range.push(
            map_task_templates
                .len()
                .try_into()
                .expect("GQA SDPA partial-output count must fit u32"),
        );
    }

    let num_q_token_ranges = q_token_ranges
        .len()
        .try_into()
        .expect("GQA SDPA Q-token-range count must fit u32");
    let num_map_task_templates = map_task_templates
        .len()
        .try_into()
        .expect("GQA SDPA Map task-template count must fit u32");
    assert!(num_map_task_templates <= limits.max_map_task_templates);
    let num_tokens = request_shapes.iter().map(|shape| shape.num_q_tokens).sum();
    let (num_total_tokens, num_total_q_token_ranges, num_total_map_task_templates) =
        policy.capacities(num_total_tokens, num_q_token_ranges, num_map_task_templates);
    let replay_shape = GQAReplayShape::new(
        num_tokens,
        num_total_tokens,
        num_q_token_ranges,
        num_total_q_token_ranges,
        num_map_task_templates,
        num_total_map_task_templates,
        map.max_q_tokens > 1 || num_map_task_templates > num_tokens,
    );

    let num_active_partial_state_groups = ranges
        .iter()
        .map(|range| {
            u64::from(range.q_token_range.flat_q_token_indices.end - range.q_token_range.flat_q_token_indices.start)
                * u64::from(range.num_map_tasks)
        })
        .sum::<u64>();
    let num_reserved_partial_state_groups = u64::from(num_map_task_templates) * u64::from(map.max_q_tokens);
    let num_replay_reserved_partial_state_groups =
        u64::from(num_total_map_task_templates) * u64::from(map.max_q_tokens);
    assert!(
        num_reserved_partial_state_groups as usize <= limits.partial_state_group_capacity,
        "GQA SDPA selection exceeds its partial-state-group capacity"
    );
    let num_scheduled_qk_token_pairs = ranges
        .iter()
        .map(|range| {
            u64::from(range.num_kv_iterations) * u64::from(map.kv_tokens_per_iteration) * u64::from(map.max_q_tokens)
        })
        .sum();
    let num_active_qk_token_pairs = ranges
        .iter()
        .map(|range| {
            u64::from(range.num_kv_iterations)
                * u64::from(map.kv_tokens_per_iteration)
                * u64::from(
                    range.q_token_range.flat_q_token_indices.end - range.q_token_range.flat_q_token_indices.start,
                )
        })
        .sum();
    let num_q_head_ranges_per_kv_head = config.q_heads_per_kv_head().div_ceil(map.max_q_heads);
    let num_map_threadblocks_per_kv_head = u64::from(num_map_task_templates) * u64::from(num_q_head_ranges_per_kv_head);
    let num_map_simdgroup_waves_per_kv_head = num_map_threadblocks_per_kv_head * u64::from(map.required_threads / 32);
    let num_logical_qk_token_pairs = request_shapes
        .iter()
        .map(|shape| {
            let num_q_tokens = u64::from(shape.num_q_tokens);
            num_q_tokens
                .checked_mul(u64::from(shape.num_history_tokens) + 1)
                .and_then(|pairs| {
                    num_q_tokens
                        .checked_mul(num_q_tokens.saturating_sub(1))
                        .and_then(|causal_pairs| pairs.checked_add(causal_pairs / 2))
                })
                .expect("GQA logical query-key pair count must fit u64")
        })
        .sum();
    let metrics = SelectionMetrics {
        num_scheduled_qk_token_pairs,
        num_active_qk_token_pairs,
        num_map_threadblocks_per_kv_head,
        num_map_simdgroup_waves_per_kv_head,
        num_active_partial_state_groups,
        num_active_partial_states: num_active_partial_state_groups
            .checked_mul(u64::from(config.num_q_heads))
            .expect("GQA active partial-state count must fit u64"),
        num_reserved_partial_state_groups,
        num_replay_reserved_partial_state_groups,
        max_kv_iterations_per_map_task,
        num_logical_qk_token_pairs,
    };

    Selection {
        variant,
        q_token_ranges,
        map_task_templates,
        cu_partial_outputs_by_q_token_range,
        replay_shape,
        metrics,
    }
}

fn build_q_token_ranges(
    request_shapes: &[RequestShape],
    max_q_tokens: u32,
    kv_tokens_per_iteration: u32,
) -> Vec<QTokenRangeWork> {
    let mut ranges = Vec::new();
    let mut flat_request_begin = 0u32;
    for (request_index, shape) in request_shapes.iter().enumerate() {
        let flat_request_end = flat_request_begin
            .checked_add(shape.num_q_tokens)
            .expect("GQA flat Q-token end must fit u32");
        let mut flat_q_token_begin = flat_request_begin;
        while flat_q_token_begin < flat_request_end {
            let flat_q_token_end = flat_q_token_begin.saturating_add(max_q_tokens).min(flat_request_end);
            let max_visible_kv_tokens = shape
                .num_history_tokens
                .checked_add(flat_q_token_end - flat_request_begin)
                .expect("GQA visible KV-token count must fit u32");
            ranges.push(QTokenRangeWork {
                q_token_range: QTokenRange {
                    request_index: request_index.try_into().expect("GQA request index must fit u32"),
                    flat_q_token_indices: flat_q_token_begin..flat_q_token_end,
                    max_visible_kv_tokens,
                },
                num_kv_iterations: max_visible_kv_tokens.div_ceil(kv_tokens_per_iteration),
                num_map_tasks: 1,
            });
            flat_q_token_begin = flat_q_token_end;
        }
        flat_request_begin = flat_request_end;
    }
    assert!(!ranges.is_empty());
    ranges
}

fn allocate_map_tasks(ranges: &mut [QTokenRangeWork], limits: SelectorLimits) {
    let mut num_map_tasks = ranges.len();
    let mut num_active_partial_state_groups = ranges
        .iter()
        .map(|range| {
            (range.q_token_range.flat_q_token_indices.end - range.q_token_range.flat_q_token_indices.start) as usize
        })
        .sum::<usize>();
    assert!(num_map_tasks <= limits.max_map_task_templates as usize);
    assert!(num_active_partial_state_groups <= limits.max_active_partial_state_groups as usize);

    while num_map_tasks < limits.max_map_task_templates as usize
        && num_active_partial_state_groups < limits.max_active_partial_state_groups as usize
    {
        let candidate = ranges
            .iter()
            .enumerate()
            .filter(|(_, range)| {
                let num_active_q_tokens = (range.q_token_range.flat_q_token_indices.end
                    - range.q_token_range.flat_q_token_indices.start)
                    as usize;
                num_active_partial_state_groups + num_active_q_tokens <= limits.max_active_partial_state_groups as usize
                    && range.num_map_tasks < range.num_kv_iterations
            })
            .map(|(index, range)| (index, range.num_kv_iterations.div_ceil(range.num_map_tasks)))
            .max_by_key(|&(index, iterations_per_task)| (iterations_per_task, Reverse(index)));
        let Some((index, _)) = candidate else {
            break;
        };
        num_active_partial_state_groups += (ranges[index].q_token_range.flat_q_token_indices.end
            - ranges[index].q_token_range.flat_q_token_indices.start)
            as usize;
        ranges[index].num_map_tasks += 1;
        num_map_tasks += 1;
    }
}

fn select_candidate(config: backend_sdpa::Config, mut candidates: Vec<Selection>) -> Selection {
    assert!(!candidates.is_empty());
    if candidates.len() == 1 {
        return candidates.pop().unwrap();
    }

    let single_q_index = candidates
        .iter()
        .position(|selection| selection.variant.map.thread_block.max_q_tokens == 1);
    let tiled_indices = candidates
        .iter()
        .enumerate()
        .filter(|(_, selection)| selection.variant.map.thread_block.max_q_tokens > 1)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let Some(&first_tiled_index) = tiled_indices.first() else {
        return candidates.remove(single_q_index.expect("GQA SDPA registry requires an execution"));
    };
    let Some(single_q_index) = single_q_index else {
        return candidates.remove(first_tiled_index);
    };

    let num_tokens = candidates[first_tiled_index].replay_shape.num_tokens;
    let num_q_token_ranges = candidates[first_tiled_index].replay_shape.num_q_token_tiles;
    if u64::from(num_tokens) < 2 * u64::from(num_q_token_ranges) {
        return candidates.remove(single_q_index);
    }

    let full_q_heads = tiled_indices
        .iter()
        .map(|&index| candidates[index].variant.map.thread_block.max_q_heads)
        .max()
        .unwrap();
    let desired_q_heads = if (config.head_dim, config.tokens_per_page) != (128, 8)
        && u64::from(num_tokens) < 4 * u64::from(num_q_token_ranges)
    {
        config.q_heads_per_kv_head().div_ceil(2).min(full_q_heads)
    } else {
        full_q_heads
    };
    let tiled_q_index = tiled_indices
        .iter()
        .copied()
        .find(|&index| candidates[index].variant.map.thread_block.max_q_heads == desired_q_heads)
        .unwrap_or(first_tiled_index);

    if (config.head_dim, config.tokens_per_page) == (256, 8)
        && !prefer_d256_page8_tiled_q(&candidates[single_q_index], &candidates[tiled_q_index])
    {
        return candidates.remove(single_q_index);
    }
    candidates.remove(tiled_q_index)
}

fn prefer_d256_page8_tiled_q(single_q: &Selection, tiled_q: &Selection) -> bool {
    let selection_score = single_q
        .metrics
        .num_logical_qk_token_pairs
        .checked_add(
            single_q
                .metrics
                .selection_overhead_score()
                .min(tiled_q.metrics.selection_overhead_score()),
        )
        .expect("GQA SDPA selection score must fit u64");
    let tiled_q_has_sufficient_map_utilization = tiled_q
        .metrics
        .num_active_qk_token_pairs
        .checked_mul(2)
        .is_some_and(|active_pairs| active_pairs >= tiled_q.metrics.num_scheduled_qk_token_pairs);
    tiled_q_has_sufficient_map_utilization && selection_score >= D256_PAGE8_MIN_SELECTION_SCORE
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
    use inference_backend_metal::metal::Dtype;

    use super::RequestShape;
    use super::Selection;
    use super::Selector;
    use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;

    fn selector() -> Selector {
        selector_for(backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            tokens_per_page: 8,
        })
    }

    fn selector_for(config: backend_sdpa::Config) -> Selector {
        Selector::new(backend_sdpa::Registry::new(config), 128)
    }

    fn selection(token_indices: &[u32], request_tokens: &[u32]) -> Selection {
        let mut cu_tokens = vec![0];
        for &num_tokens in request_tokens {
            cu_tokens.push(cu_tokens.last().copied().unwrap() + num_tokens);
        }
        let request_shapes = RequestShape::from_batch(token_indices, &cu_tokens);
        let num_total_tokens = cu_tokens.last().copied().unwrap();
        selector().select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), num_total_tokens)
    }

    fn is_single_q(selection: &Selection) -> bool {
        selection.variant().map.thread_block.max_q_tokens == 1
    }

    #[test]
    fn test_selector_preserves_measured_d256_page8_crossovers() {
        for (tokens, context) in [(1, 65536), (2, 65536), (4, 128), (8, 128), (25, 32)] {
            assert!(is_single_q(&selection(&[context], &[tokens])));
        }
        for (tokens, context) in [(4, 512), (8, 512), (16, 128), (25, 128), (64, 0)] {
            assert!(!is_single_q(&selection(&[context], &[tokens])));
        }
    }

    #[test]
    fn test_selector_preserves_non_page8_profile_selection() {
        let request_shapes = RequestShape::from_batch(&[0; 4], &[0, 2, 4, 6, 8]);
        let d256_page16 = selector_for(backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            tokens_per_page: 16,
        })
        .select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 8);
        assert_eq!(d256_page16.variant().map.thread_block.max_q_tokens, 8);
        assert_eq!(d256_page16.variant().map.thread_block.max_q_heads, 3);

        let d128_page8 = selector_for(backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            tokens_per_page: 8,
        })
        .select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 8);
        assert_eq!(d128_page8.variant().map.thread_block.max_q_tokens, 8);
        assert_eq!(d128_page8.variant().map.thread_block.max_q_heads, 8);

        let unsupported = selector_for(backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 32,
            num_kv_heads: 4,
            head_dim: 256,
            tokens_per_page: 4,
        })
        .select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 8);
        assert_eq!(unsupported.variant().map.thread_block.max_q_tokens, 1);
    }

    #[test]
    fn test_selector_prices_request_local_tail_ranges() {
        assert!(is_single_q(&selection(&[65536; 8], &[1; 8])));
        assert!(is_single_q(&selection(&[1024, 65536], &[64, 1])));
        assert!(is_single_q(&selection(&[65536, 1024], &[1, 8])));
        assert!(!is_single_q(&selection(&[65536, 65536], &[8, 1])));
    }

    #[test]
    fn test_selection_materializes_current_tiled_q_capacity_and_metrics() {
        let tail_selection = selection(&[1024, 65536], &[64, 1]);
        assert!(is_single_q(&tail_selection));

        let selection = selection(&[65536], &[25]);
        assert!(!is_single_q(&selection));
        assert_eq!(selection.q_token_ranges().len(), 4);
        assert_eq!(selection.map_task_templates().len(), 23);
        assert_eq!(selection.replay_shape().num_total_sdpa_map_task_templates, 24);
        assert_eq!(selection.metrics().num_active_partial_state_groups, 128);
        assert_eq!(selection.metrics().num_active_partial_states, 128 * 24);
        assert_eq!(selection.metrics().num_reserved_partial_state_groups, 184);
        assert_eq!(selection.metrics().num_replay_reserved_partial_state_groups, 192);
    }

    #[test]
    fn test_selection_map_tasks_cover_each_visible_kv_range_once() {
        let selection = selection(&[1024], &[25]);
        for (range_index, range) in selection.q_token_ranges().iter().enumerate() {
            let offsets = selection.cu_partial_outputs_by_q_token_range();
            let templates =
                &selection.map_task_templates()[offsets[range_index] as usize..offsets[range_index + 1] as usize];
            assert_eq!(templates.first().unwrap().request_local_kv_token_indices.start, 0);
            assert_eq!(
                templates.last().unwrap().request_local_kv_token_indices.end,
                range.max_visible_kv_tokens
            );
            for pair in templates.windows(2) {
                assert_eq!(
                    pair[0].request_local_kv_token_indices.end,
                    pair[1].request_local_kv_token_indices.start
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "GQA active token count must not exceed the total token count")]
    fn test_total_token_count_rejects_active_overflow() {
        let request_shapes = RequestShape::from_batch(&[0], &[0, 5]);
        selector().select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 4);
    }

    #[test]
    #[should_panic(expected = "GQA total token count must not exceed the metadata capacity")]
    fn test_total_token_count_rejects_metadata_overflow() {
        let request_shapes = RequestShape::from_batch(&[0], &[0, 5]);
        selector().select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 129);
    }
}
