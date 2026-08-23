use half::bf16;
use inference_executor_core::mlp::moe::reference::moe_routing_from_bf16_probs_reference;

use super::*;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayProgram;
use crate::metal::Stream;
use crate::test_support::ReplayTestCache;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.moe.routing.num_active_tokens");
const ACTIVE_SEQUENCE: [u32; 8] = [1, 8, 3, 7, 2, 6, 4, 5];

#[test]
fn test_replay_matches_reference_across_active_counts_and_normalization_modes() {
    let mut cache = ReplayTestCache::new();
    for norm_topk_prob in [false, true] {
        let fixture = RoutingFixture::new(norm_topk_prob);
        let key = (fixture.shape.num_total_tokens, norm_topk_prob);
        let (_, cache_hit) = cache.record(key, || fixture.replay());
        assert!(!cache_hit);
        for (case_index, num_active_tokens) in ACTIVE_SEQUENCE.into_iter().enumerate() {
            let probs = fixture.write_probs(0x5A17_920B_u32.wrapping_add(case_index as u32));
            let (replay, cache_hit) = cache.record(key, || unreachable!());
            assert!(cache_hit);
            fixture.submit(replay, num_active_tokens);
            fixture.assert_active_output(&probs, num_active_tokens);
        }
    }
}

struct RoutingFixture {
    stream: Stream,
    config: Config,
    shape: Shape,
    compute: Compute,
    router_probs: Buffer,
    expert_indices: Buffer,
    expert_probs: Buffer,
}

impl RoutingFixture {
    fn new(norm_topk_prob: bool) -> Self {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            num_experts: 8,
            num_experts_per_token: 3,
            norm_topk_prob,
        };
        let shape = Shape { num_total_tokens: 8 };
        Self {
            router_probs: Buffer::new_zeroed(&device, config.router_probs_bytes(shape)),
            expert_indices: Buffer::new_zeroed(&device, config.expert_indices_bytes(shape)),
            expert_probs: Buffer::new_zeroed(&device, config.expert_probs_bytes(shape)),
            compute: Compute::new(&device, config),
            stream,
            config,
            shape,
        }
    }

    fn replay(&self) -> ReplayProgram {
        let mut builder = self.stream.create_replay_program();
        builder.record(self.compute.invoke(
            self.shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            Buffers {
                router_probs: &self.router_probs,
                expert_indices: &self.expert_indices,
                expert_probs: &self.expert_probs,
            },
        ));
        builder.build()
    }

    fn write_probs(&self, seed: u32) -> Vec<f32> {
        let values = bf16_values(&generated_probs(
            self.shape.num_total_tokens as usize,
            self.config.num_experts as usize,
            seed,
        ));
        self.router_probs.write_typed(0, &bf16_bits(&values));
        values
    }

    fn submit(&self, replay: &ReplayProgram, num_active_tokens: u32) {
        let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens);
        self.stream.submit_replay_with_arguments(replay, &arguments).wait();
    }

    fn assert_active_output(&self, probs: &[f32], num_active_tokens: u32) {
        let num_active_tokens = num_active_tokens as usize;
        let num_experts = self.config.num_experts as usize;
        let num_active_routes = num_active_tokens * self.config.num_experts_per_token as usize;
        let expected = moe_routing_from_bf16_probs_reference(
            &probs[..num_active_tokens * num_experts],
            num_active_tokens,
            num_experts,
            self.config.num_experts_per_token as usize,
            self.config.norm_topk_prob,
        );
        let actual_indices = self.expert_indices.read_typed::<u32>(0, num_active_routes);
        let actual_probs = self.expert_probs.read_typed::<f32>(0, num_active_routes);
        assert_eq!(actual_indices, expected.expert_indices);
        assert_close(&actual_probs, &expected.expert_probs, 1.0e-3);
    }
}

fn generated_probs(num_tokens: usize, num_experts: usize, mut state: u32) -> Vec<f32> {
    let mut probs = Vec::with_capacity(num_tokens * num_experts);
    for _ in 0..num_tokens {
        let mut row = Vec::with_capacity(num_experts);
        let mut sum = 0.0_f32;
        for _ in 0..num_experts {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = ((state >> 8) as f32 / 16_777_216.0) + 0.01;
            row.push(value);
            sum += value;
        }
        probs.extend(row.into_iter().map(|value| value / sum));
    }
    probs
}

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect()
}

fn bf16_values(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
