use half::bf16;
use inference_backend_metal::MetalRuntime;
use inference_backend_metal::components::QuantizedEmbeddingKernel;
use inference_backend_metal::components::RowwiseAddConfig;
use inference_backend_metal::components::RowwiseAddKernel;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::TopKSamplingBounds;

use super::AffineQuantizedMatmul;
use super::AffineQuantizedMatmulConfig;
use super::DSparkMarkovSampling;
use super::DSparkMarkovWeights;
use super::QuantizedEmbeddingConfig;
use super::SpecProbsStore;
use super::TopKSampling;
use super::TopKSamplingOutputBuffers;
use super::replay_bucket_capacity;
use crate::def::replay_op::MetalReplayRuntime;

#[test]
fn test_sampling_bucket_caps_at_non_power_of_two_request_capacity() {
    assert_eq!(replay_bucket_capacity(3, 6), 4);
    assert_eq!(replay_bucket_capacity(5, 6), 6);
}

#[test]
fn test_markov_sampling_uses_each_sampled_token_for_the_next_step() {
    const BLOCK_SIZE: usize = 3;
    const NUM_REQUESTS: usize = 2;
    const VOCAB_SIZE: usize = 64;
    const RANK: usize = 32;

    let runtime = MetalRuntime::system_default();
    let device = runtime.device();
    let bounds = TopKSamplingBounds {
        max_sampling_inputs: NUM_REQUESTS as u32,
        vocab_size: VOCAB_SIZE as u32,
        top_k: 1,
    };
    let w1_config = QuantizedEmbeddingConfig {
        vocab_size: VOCAB_SIZE as u32,
        hidden_dim: RANK as u32,
        group_size: RANK as u32,
        bits: 8,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
    };
    let w2_config =
        AffineQuantizedMatmulConfig::same_dtype(VOCAB_SIZE as i32, RANK as i32, RANK as i32, 8, Dtype::Bfloat16);

    let mut w1_weight = vec![0u8; VOCAB_SIZE * RANK];
    let mut w2_weight = vec![0u8; VOCAB_SIZE * RANK];
    for token_id in 0..RANK - 1 {
        w1_weight[token_id * RANK + token_id] = 1;
        w2_weight[(token_id + 1) * RANK + token_id] = 16;
    }
    let unit_affine = vec![bf16::from_f32(1.0).to_bits(); VOCAB_SIZE];
    let zero_affine = vec![bf16::ZERO.to_bits(); VOCAB_SIZE];
    let weights = DSparkMarkovWeights {
        w1_weight: Buffer::from_slice(device, &w1_weight),
        w1_scales: Buffer::from_slice(device, &unit_affine),
        w1_biases: Buffer::from_slice(device, &zero_affine),
        w2_weight: Buffer::from_slice(device, &w2_weight),
        w2_scales: Buffer::from_slice(device, &unit_affine),
        w2_biases: Buffer::from_slice(device, &zero_affine),
    };
    let markov = DSparkMarkovSampling {
        block_size: BLOCK_SIZE,
        max_requests: NUM_REQUESTS,
        w1_config,
        weights,
        w1: QuantizedEmbeddingKernel::new(device, w1_config),
        w2: AffineQuantizedMatmul::new(device, w2_config),
        add_bias: RowwiseAddKernel::new(
            device,
            RowwiseAddConfig {
                row_width: VOCAB_SIZE as u32,
                dtype: Dtype::Bfloat16,
            },
        ),
        anchor_token_ids: Buffer::new_zeroed_elements(device, NUM_REQUESTS, Dtype::Int32),
        latent: Buffer::new_zeroed_elements(device, NUM_REQUESTS * RANK, Dtype::Bfloat16),
        bias_logits: Buffer::new_zeroed_elements(device, NUM_REQUESTS * VOCAB_SIZE, Dtype::Bfloat16),
        corrected_logits: Buffer::new_zeroed_elements(device, NUM_REQUESTS * VOCAB_SIZE, Dtype::Bfloat16),
        step_samplers: (0..BLOCK_SIZE).map(|_| TopKSampling::new(device, bounds)).collect(),
        step_outputs: (0..BLOCK_SIZE)
            .map(|_| TopKSamplingOutputBuffers::new(device, bounds))
            .collect(),
        step_distribution_indices: (0..BLOCK_SIZE)
            .map(|_| Buffer::new_zeroed_elements(device, NUM_REQUESTS, Dtype::Uint32))
            .collect(),
    };
    let mut distribution_store = SpecProbsStore::new(device, BLOCK_SIZE, NUM_REQUESTS, 1);
    let sampler_config = SamplerConfig {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        seed: 42,
    };
    let shape = markov.prepare(
        &[0, 1],
        &[1, 5],
        &[10, 20],
        &[sampler_config; NUM_REQUESTS],
        &distribution_store,
    );
    let base_logits = Buffer::new_zeroed_elements(device, BLOCK_SIZE * NUM_REQUESTS * VOCAB_SIZE, Dtype::Bfloat16);
    let replay_runtime = MetalReplayRuntime::new(runtime.stream());
    let mut recorder = replay_runtime.create_recorder();
    markov.record(&mut recorder, shape, &base_logits, &distribution_store);
    let replay = recorder.build();
    let mut arguments = ReplayArguments::new();
    markov.add_replay_arguments(shape, &mut arguments);
    replay_runtime.submit_replay_with_arguments(&replay, &arguments).wait();

    let proposal = markov.read_proposal(&[0, 1], &mut distribution_store);
    assert_eq!(proposal.token_ids, vec![vec![2, 3, 4], vec![6, 7, 8]]);
    assert_eq!(proposal.token_probs, vec![vec![1.0; BLOCK_SIZE]; NUM_REQUESTS]);
}
