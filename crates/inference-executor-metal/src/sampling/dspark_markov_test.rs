use half::bf16;
use inference_backend_metal::MetalRuntime;
use inference_backend_metal::components::DSparkConfidenceConfig;
use inference_backend_metal::components::DSparkMarkovTopKMapConfig;
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
use super::DSparkMarkovSampling;
use super::DSparkMarkovSamplingConfig;
use super::DSparkMarkovWeights;
use super::SpecProbsStore;
use crate::def::replay_op::MetalReplayRuntime;

#[test]
fn test_markov_sampling_uses_each_sampled_token_for_the_next_step() {
    const BLOCK_SIZE: usize = 3;
    const MAX_REQUESTS: usize = 6;
    const NUM_REQUESTS: usize = 3;
    const VOCAB_SIZE: usize = 64;
    const RANK: usize = 64;
    const HIDDEN_DIM: usize = 32;
    const REQ_SLOTS: [u32; NUM_REQUESTS] = [0, 2, 5];
    const ANCHOR_TOKEN_IDS: [u32; NUM_REQUESTS] = [1, 21, 45];
    const ANCHOR_POSITIONS: [u32; NUM_REQUESTS] = [10, 20, 30];

    let runtime = MetalRuntime::system_default();
    let device = runtime.device();
    let bounds = TopKSamplingBounds {
        max_sampling_inputs: MAX_REQUESTS as u32,
        vocab_size: VOCAB_SIZE as u32,
        top_k: 4,
    };
    let map_config = DSparkMarkovTopKMapConfig {
        vocab_size: VOCAB_SIZE as u32,
        rank: RANK as u32,
        w1_group_size: RANK as u32,
        w1_bits: 8,
        w2_group_size: RANK as u32,
        w2_bits: 8,
        io_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
        confidence: DSparkConfidenceConfig {
            hidden_dim: HIDDEN_DIM as u32,
        },
    };

    let mut w1_weight = vec![0u8; VOCAB_SIZE * RANK];
    let mut w2_weight = vec![0u8; VOCAB_SIZE * RANK];
    for token_id in 0..RANK - 1 {
        w1_weight[token_id * RANK + token_id] = 1;
        w2_weight[(token_id + 1) * RANK + token_id] = 2;
    }
    let unit_affine = vec![bf16::from_f32(1.0).to_bits(); VOCAB_SIZE];
    let zero_affine = vec![bf16::ZERO.to_bits(); VOCAB_SIZE];
    let sampler_config = SamplerConfig {
        temperature: 0.7,
        top_k: 4,
        top_p: 0.9,
        seed: 42,
    };
    let base_logits_values = vec![0.0; BLOCK_SIZE * NUM_REQUESTS * VOCAB_SIZE];
    let reference = dspark_markov_reference(
        DSparkMarkovReferenceConfig {
            block_size: BLOCK_SIZE,
            vocab_size: VOCAB_SIZE,
            rank: RANK,
            w1_group_size: RANK,
            w1_bits: 8,
            w2_group_size: RANK,
            w2_bits: 8,
        },
        DSparkMarkovReferenceWeights {
            w1_weight: &w1_weight,
            w1_scales: &vec![1.0; VOCAB_SIZE],
            w1_biases: &vec![0.0; VOCAB_SIZE],
            w2_weight: &w2_weight,
            w2_scales: &vec![1.0; VOCAB_SIZE],
            w2_biases: &vec![0.0; VOCAB_SIZE],
        },
        &ANCHOR_TOKEN_IDS,
        &ANCHOR_POSITIONS,
        &[sampler_config; NUM_REQUESTS],
        &base_logits_values,
        4,
    );
    let hidden_values = (0..BLOCK_SIZE * NUM_REQUESTS * HIDDEN_DIM)
        .map(|index| (index % HIDDEN_DIM) as f32 * 0.001)
        .collect::<Vec<_>>();
    let confidence_weight = (0..HIDDEN_DIM + RANK)
        .map(|index| (index as f32 - 16.0) * 0.002)
        .collect::<Vec<_>>();
    let confidence_bias = -0.25;
    let reference_confidences = dspark_confidence_reference(
        DSparkMarkovReferenceConfig {
            block_size: BLOCK_SIZE,
            vocab_size: VOCAB_SIZE,
            rank: RANK,
            w1_group_size: RANK,
            w1_bits: 8,
            w2_group_size: RANK,
            w2_bits: 8,
        },
        DSparkMarkovReferenceWeights {
            w1_weight: &w1_weight,
            w1_scales: &vec![1.0; VOCAB_SIZE],
            w1_biases: &vec![0.0; VOCAB_SIZE],
            w2_weight: &w2_weight,
            w2_scales: &vec![1.0; VOCAB_SIZE],
            w2_biases: &vec![0.0; VOCAB_SIZE],
        },
        DSparkConfidenceReferenceConfig { hidden_dim: HIDDEN_DIM },
        DSparkConfidenceReferenceWeights {
            weight: &confidence_weight,
            bias: confidence_bias,
        },
        &ANCHOR_TOKEN_IDS,
        &reference,
        &hidden_values,
    );
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
            sampling: bounds,
        },
    );
    let w1_weight_buffer = Buffer::from_slice(device, &w1_weight);
    let w1_scales = Buffer::from_slice(device, &unit_affine);
    let w1_biases = Buffer::from_slice(device, &zero_affine);
    let w2_weight_buffer = Buffer::from_slice(device, &w2_weight);
    let w2_scales = Buffer::from_slice(device, &unit_affine);
    let w2_biases = Buffer::from_slice(device, &zero_affine);
    let weights = DSparkMarkovWeights {
        w1_weight: &w1_weight_buffer,
        w1_scales: &w1_scales,
        w1_biases: &w1_biases,
        w2_weight: &w2_weight_buffer,
        w2_scales: &w2_scales,
        w2_biases: &w2_biases,
    };
    let confidence_weight_buffer = Buffer::from_slice(
        device,
        &confidence_weight
            .iter()
            .map(|&value| bf16::from_f32(value).to_bits())
            .collect::<Vec<_>>(),
    );
    let confidence_bias_buffer = Buffer::from_slice(device, &[bf16::from_f32(confidence_bias).to_bits()]);
    let mut distribution_store =
        SpecProbsStore::new(device, BLOCK_SIZE, MAX_REQUESTS, MAX_REQUESTS * (BLOCK_SIZE + 1), 4);
    let shape = markov.prepare(
        &REQ_SLOTS,
        &ANCHOR_TOKEN_IDS,
        &ANCHOR_POSITIONS,
        &[sampler_config; NUM_REQUESTS],
        &distribution_store,
    );
    assert_eq!(shape.sampling.num_active_sampling_inputs, NUM_REQUESTS as u32);
    assert_eq!(shape.sampling.num_total_sampling_inputs, 4);
    let mut base_logits_storage = vec![bf16::ZERO.to_bits(); BLOCK_SIZE * MAX_REQUESTS * VOCAB_SIZE];
    for (output, &value) in base_logits_storage.iter_mut().zip(&base_logits_values) {
        *output = bf16::from_f32(value).to_bits();
    }
    let base_logits = Buffer::from_slice(device, &base_logits_storage);
    let mut hidden_storage = vec![bf16::ZERO.to_bits(); BLOCK_SIZE * MAX_REQUESTS * HIDDEN_DIM];
    for (output, &value) in hidden_storage.iter_mut().zip(&hidden_values) {
        *output = bf16::from_f32(value).to_bits();
    }
    let hidden = Buffer::from_slice(device, &hidden_storage);
    let replay_runtime = MetalReplayRuntime::new(runtime.stream());
    let mut recorder = replay_runtime.create_recorder();
    markov.record(
        &mut recorder,
        DSparkMarkovInput {
            shape,
            base_logits: &base_logits,
            distribution_store: &distribution_store,
            weights,
            confidence: DSparkConfidenceInput {
                hidden: &hidden,
                weights: DSparkConfidenceWeights {
                    weight: &confidence_weight_buffer,
                    bias: &confidence_bias_buffer,
                },
            },
        },
    );
    let replay = recorder.build();
    let mut arguments = ReplayArguments::new();
    markov.add_replay_arguments(shape, &mut arguments);
    replay_runtime.submit_replay_with_arguments(&replay, &arguments).wait();

    let proposal = markov.read_proposal(&REQ_SLOTS, &mut distribution_store);
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

    let distribution_slots = BLOCK_SIZE * MAX_REQUESTS * 4;
    let draft_token_ids = distribution_store
        .draft_token_ids()
        .read_typed::<i32>(0, distribution_slots);
    let draft_probs = distribution_store
        .draft_probs()
        .read_typed::<f32>(0, distribution_slots);
    for (request_index, &req_slot) in REQ_SLOTS.iter().enumerate() {
        for step_index in 0..BLOCK_SIZE {
            let distribution_index = req_slot as usize * BLOCK_SIZE + step_index;
            let slot_begin = distribution_index * 4;
            let slot_end = slot_begin + 4;
            assert_eq!(
                &draft_token_ids[slot_begin..slot_end],
                reference.samples_by_request[request_index][step_index].prob_token_ids
            );
            assert_close(
                &draft_probs[slot_begin..slot_end],
                &reference.samples_by_request[request_index][step_index].prob_values,
                1.0e-5,
            );
        }
    }
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
