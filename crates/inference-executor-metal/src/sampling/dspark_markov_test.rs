use std::rc::Rc;

use half::bf16;
use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::dspark::DSparkConfidenceReferenceConfig;
use inference_executor_core::sampling::dspark::DSparkConfidenceReferenceWeights;
use inference_executor_core::sampling::dspark::DSparkMarkovReferenceConfig;
use inference_executor_core::sampling::dspark::DSparkMarkovReferenceWeights;
use inference_executor_core::sampling::dspark::dspark_confidence_reference;
use inference_executor_core::sampling::dspark::dspark_markov_reference;

use super::DSparkConfidenceInput;
use super::DSparkConfidenceWeights;
use super::DSparkMarkovConfidenceConfig;
use super::DSparkMarkovInput;
use super::DSparkMarkovReplayShape;
use super::DSparkMarkovSampling;
use super::DSparkMarkovSamplingConfig;
use super::DSparkMarkovWeights;
use super::SpecProbsStore;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::ReplayRecorder;
use crate::replay::Replay;
use crate::replay::ReplayComponent;
use crate::sampling::sampling_params::SamplingParamsStore;

const BLOCK_SIZE: usize = 3;
const MAX_REQUESTS: usize = 8;
const VOCAB_SIZE: usize = 64;
const RANK: usize = 64;
const HIDDEN_DIM: usize = 32;
const ACTIVE_SEQUENCE: [usize; 8] = [1, 8, 3, 7, 2, 6, 4, 5];
const REQUEST_SLOTS: [u32; 8] = [7, 0, 5, 2, 6, 1, 4, 3];
const ANCHOR_TOKEN_IDS: [u32; 8] = [1, 21, 45, 13, 33, 9, 51, 17];
const ANCHOR_POSITIONS: [u32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

#[test]
fn test_replay_matches_reference_across_active_counts_and_request_slots() {
    let runtime = MetalRuntime::system_default();
    let replay_runtime = MetalReplayRuntime::new(runtime.stream());
    let mut replay = Replay::new("DSpark Markov component test", MarkovFixture::new(runtime.device()));

    for num_active_requests in ACTIVE_SEQUENCE {
        let req_slots = &REQUEST_SLOTS[..num_active_requests];
        let shape = replay.component().prepare(num_active_requests);
        assert_eq!(shape.num_total_requests, num_active_requests as u32);
        assert_eq!(shape.num_active_requests, num_active_requests as u32);
        assert!(shape.sampling.num_total_sampling_inputs >= num_active_requests as u32);

        let (key, cache_hit) = replay.record(&replay_runtime, &shape);
        assert!(!cache_hit);
        assert_eq!(replay.record(&replay_runtime, &shape), (key, true));
        let mut arguments = ReplayArguments::new();
        replay.component().add_replay_arguments(shape, &mut arguments);
        replay_runtime
            .submit_replay_with_arguments(replay.replay(&key), &arguments)
            .wait();
        replay.component_mut().assert_output_matches_reference(shape, req_slots);
    }
}

struct MarkovFixture {
    markov: DSparkMarkovSampling,
    distribution_store: SpecProbsStore,
    base_logits: Buffer,
    hidden: Buffer,
    w1_weight: Buffer,
    w1_scales: Buffer,
    w1_biases: Buffer,
    w2_weight: Buffer,
    w2_scales: Buffer,
    w2_biases: Buffer,
    confidence_weight: Buffer,
    confidence_bias: Buffer,
    w1_weight_values: Vec<u8>,
    w2_weight_values: Vec<u8>,
    confidence_weight_values: Vec<f32>,
    confidence_bias_value: f32,
    base_logits_values: Vec<f32>,
    hidden_values: Vec<f32>,
    sampler_config: SamplerConfig,
    sampling_params: Rc<SamplingParamsStore>,
    sample_positions: Buffer,
}

impl MarkovFixture {
    fn new(device: &inference_backend_metal::metal::Device) -> Self {
        let sampling = TopKSamplingBounds {
            max_sampling_inputs: MAX_REQUESTS as u32,
            vocab_size: VOCAB_SIZE as u32,
            top_k: 4,
        };
        let sampling_params = Rc::new(SamplingParamsStore::new(
            device,
            TopKSamplingBounds {
                max_sampling_inputs: 2 * MAX_REQUESTS as u32,
                ..sampling
            },
            MAX_REQUESTS as u32,
        ));
        let markov = DSparkMarkovSampling::new(
            device,
            DSparkMarkovSamplingConfig {
                block_size: BLOCK_SIZE,
                vocab_size: VOCAB_SIZE as u32,
                rank: RANK as u32,
                w1_group_size: RANK as u32,
                w1_bits: 8,
                w2_group_size: RANK as u32,
                w2_bits: 8,
                io_dtype: Dtype::Bfloat16,
                scale_bias_dtype: Dtype::Bfloat16,
                confidence: DSparkMarkovConfidenceConfig {
                    hidden_dim: HIDDEN_DIM as u32,
                },
                sampling,
            },
            Rc::clone(&sampling_params),
        );
        let mut w1_weight_values = vec![0_u8; VOCAB_SIZE * RANK];
        let mut w2_weight_values = vec![0_u8; VOCAB_SIZE * RANK];
        for token_id in 0..RANK - 1 {
            w1_weight_values[token_id * RANK + token_id] = 1;
            w2_weight_values[(token_id + 1) * RANK + token_id] = 2;
        }
        let unit_affine = vec![bf16::from_f32(1.0).to_bits(); VOCAB_SIZE];
        let zero_affine = vec![bf16::ZERO.to_bits(); VOCAB_SIZE];
        let confidence_weight_values = (0..HIDDEN_DIM + RANK)
            .map(|index| bf16::from_f32((index as f32 - 16.0) * 0.002).to_f32())
            .collect::<Vec<_>>();
        let confidence_bias_value = bf16::from_f32(-0.25).to_f32();
        let base_logits_values = vec![0.0; BLOCK_SIZE * MAX_REQUESTS * VOCAB_SIZE];
        let hidden_values = (0..BLOCK_SIZE * MAX_REQUESTS * HIDDEN_DIM)
            .map(|index| bf16::from_f32((index % HIDDEN_DIM) as f32 * 0.001).to_f32())
            .collect::<Vec<_>>();
        Self {
            distribution_store: SpecProbsStore::new(
                device,
                BLOCK_SIZE,
                MAX_REQUESTS,
                MAX_REQUESTS * (BLOCK_SIZE + 1),
                4,
            ),
            base_logits: Buffer::from_slice(device, &bf16_bits(&base_logits_values)),
            hidden: Buffer::from_slice(device, &bf16_bits(&hidden_values)),
            w1_weight: Buffer::from_slice(device, &w1_weight_values),
            w1_scales: Buffer::from_slice(device, &unit_affine),
            w1_biases: Buffer::from_slice(device, &zero_affine),
            w2_weight: Buffer::from_slice(device, &w2_weight_values),
            w2_scales: Buffer::from_slice(device, &unit_affine),
            w2_biases: Buffer::from_slice(device, &zero_affine),
            confidence_weight: Buffer::from_slice(device, &bf16_bits(&confidence_weight_values)),
            confidence_bias: Buffer::from_slice(device, &[bf16::from_f32(confidence_bias_value).to_bits()]),
            markov,
            w1_weight_values,
            w2_weight_values,
            confidence_weight_values,
            confidence_bias_value,
            base_logits_values,
            hidden_values,
            sampler_config: SamplerConfig {
                temperature: 0.7,
                top_k: 4,
                top_p: 0.9,
                seed: 42,
            },
            sampling_params,
            sample_positions: Buffer::new_zeroed_elements(device, MAX_REQUESTS, Dtype::Uint32),
        }
    }

    fn prepare(&self, num_active_requests: usize) -> DSparkMarkovReplayShape {
        let req_slots = &REQUEST_SLOTS[..num_active_requests];
        let configs = vec![self.sampler_config; num_active_requests];
        self.sampling_params.set(req_slots, &configs);
        self.markov.anchor_token_ids().write_typed(
            0,
            &ANCHOR_TOKEN_IDS[..num_active_requests]
                .iter()
                .map(|&token_id| token_id as i32)
                .collect::<Vec<_>>(),
        );
        self.sample_positions.write_typed(
            0,
            &ANCHOR_POSITIONS[..num_active_requests]
                .iter()
                .map(|&position| position + 1)
                .collect::<Vec<_>>(),
        );
        self.markov
            .prepare_static(req_slots, &configs, &self.distribution_store)
    }

    fn add_replay_arguments(&self, shape: DSparkMarkovReplayShape, arguments: &mut ReplayArguments) {
        self.markov.add_replay_arguments(shape, arguments);
    }

    fn weights(&self) -> DSparkMarkovWeights<'_> {
        DSparkMarkovWeights {
            w1_weight: &self.w1_weight,
            w1_scales: &self.w1_scales,
            w1_biases: &self.w1_biases,
            w2_weight: &self.w2_weight,
            w2_scales: &self.w2_scales,
            w2_biases: &self.w2_biases,
        }
    }

    fn confidence(&self) -> DSparkConfidenceInput<'_> {
        DSparkConfidenceInput {
            hidden: &self.hidden,
            weights: DSparkConfidenceWeights {
                weight: &self.confidence_weight,
                bias: &self.confidence_bias,
            },
        }
    }

    fn assert_output_matches_reference(&mut self, shape: DSparkMarkovReplayShape, req_slots: &[u32]) {
        let num_requests = req_slots.len();
        let reference_config = DSparkMarkovReferenceConfig {
            block_size: BLOCK_SIZE,
            vocab_size: VOCAB_SIZE,
            rank: RANK,
            w1_group_size: RANK,
            w1_bits: 8,
            w2_group_size: RANK,
            w2_bits: 8,
        };
        let reference_weights = DSparkMarkovReferenceWeights {
            w1_weight: &self.w1_weight_values,
            w1_scales: &vec![1.0; VOCAB_SIZE],
            w1_biases: &vec![0.0; VOCAB_SIZE],
            w2_weight: &self.w2_weight_values,
            w2_scales: &vec![1.0; VOCAB_SIZE],
            w2_biases: &vec![0.0; VOCAB_SIZE],
        };
        let num_base_values = BLOCK_SIZE * num_requests * VOCAB_SIZE;
        let reference = dspark_markov_reference(
            reference_config,
            reference_weights,
            &ANCHOR_TOKEN_IDS[..num_requests],
            &ANCHOR_POSITIONS[..num_requests],
            &vec![self.sampler_config; num_requests],
            &self.base_logits_values[..num_base_values],
            4,
        );
        let reference_confidences = dspark_confidence_reference(
            reference_config,
            reference_weights,
            DSparkConfidenceReferenceConfig { hidden_dim: HIDDEN_DIM },
            DSparkConfidenceReferenceWeights {
                weight: &self.confidence_weight_values,
                bias: self.confidence_bias_value,
            },
            &ANCHOR_TOKEN_IDS[..num_requests],
            &reference,
            &self.hidden_values[..BLOCK_SIZE * num_requests * HIDDEN_DIM],
        );
        let proposal = self.markov.read_proposal(req_slots, &mut self.distribution_store);
        let expected_token_ids = reference
            .samples_by_request
            .iter()
            .map(|steps| steps.iter().map(|step| step.sampled_token).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let expected_token_probs = reference
            .samples_by_request
            .iter()
            .map(|steps| steps.iter().map(|step| step.sampled_prob).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(proposal.token_ids, expected_token_ids);
        assert_close_nested(&proposal.token_probs, &expected_token_probs, 1.0e-5);
        assert_close_nested(&proposal.confidences, &reference_confidences, 1.0e-5);

        for (&req_slot, reference_samples) in req_slots.iter().zip(&reference.samples_by_request[..req_slots.len()]) {
            for (step_index, reference_sample) in reference_samples[..BLOCK_SIZE].iter().enumerate() {
                let distribution_index = self.distribution_store.draft_distribution_index(req_slot, step_index);
                let slot_begin = distribution_index as usize * 4;
                assert_eq!(
                    self.distribution_store
                        .draft_token_ids()
                        .read_typed::<i32>(slot_begin, 4),
                    reference_sample.prob_token_ids
                );
                assert_close(
                    &self.distribution_store.draft_probs().read_typed::<f32>(slot_begin, 4),
                    &reference_sample.prob_values,
                    1.0e-5,
                );
            }
        }
        assert_eq!(shape.num_active_requests as usize, num_requests);
    }
}

impl ReplayComponent for MarkovFixture {
    type Key = (u32, u32, u32);
    type Input<'a> = DSparkMarkovReplayShape;

    fn replay_key(&self, shape: &Self::Input<'_>) -> Self::Key {
        (
            shape.num_total_requests,
            shape.sampling.num_total_sampling_inputs,
            shape.sampling.top_k,
        )
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, shape: &Self::Input<'a>) {
        self.markov.record(
            recorder,
            DSparkMarkovInput {
                shape: *shape,
                base_logits: &self.base_logits,
                sample_positions: &self.sample_positions,
                distribution_store: &self.distribution_store,
                weights: self.weights(),
                confidence: self.confidence(),
            },
        );
    }
}

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect()
}

fn assert_close_nested(actual: &[Vec<f32>], expected: &[Vec<f32>], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_close(actual, expected, tolerance);
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "DSpark Markov reference mismatch at {index}: actual={actual} expected={expected} diff={diff}"
        );
    }
}
