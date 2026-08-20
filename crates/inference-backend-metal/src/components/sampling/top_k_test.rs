use half::bf16;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::reference::sparse_sample_row_reference;
use inference_executor_core::sampling::reference::sparse_sample_row_with_domain_reference;

use crate::components::TopKMapBuffers;
use crate::components::TopKMapKernels;
use crate::components::TopKReduceKernels;
use crate::components::TopKSampleAndWriteDistributionBuffers;
use crate::components::TopKSampleBuffers;
use crate::components::TopKSampleShape;
use crate::components::TopKSamplingOperation;
use crate::components::TopKWriteDistributionBuffers;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::ReplayArguments;
use crate::metal::Stream;

fn sampling_runtime_params(device: &Device, rows: u32, temperature: f32, top_p: f32, seed: u32, top_k: u32) -> Buffer {
    let params = Buffer::new_zeroed(device, rows as usize * 6 * size_of::<u32>());
    let config = SamplerConfig {
        temperature,
        top_k: top_k as usize,
        top_p,
        seed,
    };
    for row in 0..rows as usize {
        write_sampling_runtime_params(&params, row, &config, row as u32, SamplingDomain::Target);
    }
    params
}

fn write_sampling_runtime_params(
    params: &Buffer,
    row: usize,
    config: &SamplerConfig,
    sample_position: u32,
    domain: SamplingDomain,
) {
    let offset = row * 6;
    params.write_typed(offset, &[config.temperature, config.top_p]);
    params.write_typed(
        offset + 2,
        &[config.seed(), sample_position, config.top_k as u32, domain as u32],
    );
}

#[test]
fn test_map_kernel_specialization_selection() {
    let thread_block = super::TopKMapThreadBlockSpecialization::current();
    assert_eq!(thread_block.max_vocab_tokens, 256);
    assert_eq!(thread_block.required_threads, 256);
    assert_eq!(super::standard_partial_candidate_layout().vocab_partition_size(), 256);
    let reduction_shape = TopKSampleShape {
        num_total_sampling_inputs: 1,
        vocab_size: 256,
        top_k: 20,
    };
    let bitonic_shape = TopKSampleShape {
        top_k: 64,
        ..reduction_shape
    };
    assert_eq!(
        super::TopKMapKernelSpecialization::select(
            reduction_shape,
            crate::metal::Dtype::Float32,
            TopKSamplingOperation::Sample,
        )
        .algorithm,
        super::TopKMapAlgorithm::Reduction
    );
    assert_eq!(
        super::TopKMapKernelSpecialization::select(
            bitonic_shape,
            crate::metal::Dtype::Float32,
            TopKSamplingOperation::Sample,
        )
        .algorithm,
        super::TopKMapAlgorithm::Bitonic
    );
    assert_eq!(
        super::TopKMapKernelSpecialization::select(
            reduction_shape,
            crate::metal::Dtype::Float32,
            TopKSamplingOperation::WriteDistribution,
        )
        .algorithm,
        super::TopKMapAlgorithm::Bitonic
    );
    assert_eq!(
        super::TopKMapKernelSpecialization::select(
            reduction_shape,
            crate::metal::Dtype::Float32,
            TopKSamplingOperation::SampleAndWriteDistribution,
        )
        .algorithm,
        super::TopKMapAlgorithm::Bitonic
    );
}

fn partial_candidate_count(shape: TopKSampleShape) -> usize {
    super::partial_candidate_count(shape, super::standard_partial_candidate_layout())
}

#[test]
fn test_sample_mixed_runtime_params_and_bucket_reuse() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = TopKSampleShape {
        num_total_sampling_inputs: 3,
        vocab_size: 17,
        top_k: 5,
    };
    let logits_values = generated_logits(
        shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
        0x19A2_7C4D,
    );
    let logits = Buffer::from_slice(&device, &logits_values);
    let num_tile_candidates = partial_candidate_count(shape);
    let num_tile_candidates_per_row = num_tile_candidates / shape.num_total_sampling_inputs as usize;
    let tile_token_canary = i32::MIN;
    let tile_logit_canary = -777.0_f32;
    let output_token_canary = -99_i32;
    let output_prob_canary = -99.0_f32;
    let tile_token_ids = Buffer::from_slice(&device, &vec![tile_token_canary; num_tile_candidates]);
    let tile_logits = Buffer::from_slice(&device, &vec![tile_logit_canary; num_tile_candidates]);
    let token_ids = Buffer::from_slice(&device, &[output_token_canary; 3]);
    let token_probs = Buffer::from_slice(&device, &[output_prob_canary; 3]);
    let topk = TopKMapKernels::new(&device);
    let logits_dtype = crate::metal::Dtype::Float32;
    let sample = TopKReduceKernels::new(&device);
    let runtime_params = Buffer::new_zeroed(&device, 3 * 6 * size_of::<u32>());
    let configs = [
        SamplerConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            seed: 7,
        },
        SamplerConfig {
            temperature: 0.9,
            top_k: 5,
            top_p: 0.82,
            seed: 19,
        },
        SamplerConfig {
            temperature: 0.7,
            top_k: 3,
            top_p: 0.9,
            seed: 31,
        },
    ];
    let sample_positions = [11, 29, 41];
    let domains = [SamplingDomain::Target, SamplingDomain::Draft, SamplingDomain::Target];
    for row in 0..3 {
        write_sampling_runtime_params(&runtime_params, row, &configs[row], sample_positions[row], domains[row]);
    }

    let mut builder = stream.create_replay_program();
    builder.record(topk.invoke_replay(
        shape,
        logits_dtype,
        TopKSamplingOperation::Sample,
        TopKMapBuffers {
            logits: &logits,
            logits_offset_bytes: 0,
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
        },
    ));
    builder.record_with_barrier_before(sample.invoke_sample(
        shape,
        TopKSampleBuffers {
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
            token_ids: &token_ids,
            token_probs: &token_probs,
            runtime_params: &runtime_params,
        },
    ));
    let replay = builder.build();

    let submit = |num_active_sampling_inputs| {
        let mut arguments = ReplayArguments::new();
        topk.add_replay_arguments(shape, num_active_sampling_inputs, &mut arguments);
        sample.add_replay_arguments(shape, num_active_sampling_inputs, &mut arguments);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();
    };
    let assert_active_outputs = |num_active_sampling_inputs: usize| {
        let actual_tokens = token_ids.read_typed::<i32>(0, 3);
        let actual_probs = token_probs.read_typed::<f32>(0, 3);
        for row in 0..num_active_sampling_inputs {
            let expected = sparse_sample_row_with_domain_reference(
                &configs[row],
                &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
                configs[row].top_k,
                sample_positions[row],
                domains[row],
            );
            assert_eq!(actual_tokens[row], expected.sampled_token as i32, "row={row}");
            assert_close(&actual_probs[row..row + 1], &[expected.sampled_prob], 1.0e-5);
        }
    };

    submit(2);
    assert_active_outputs(2);
    assert_eq!(token_ids.read_typed::<i32>(2, 1), vec![output_token_canary]);
    assert_eq!(token_probs.read_typed::<f32>(2, 1), vec![output_prob_canary]);
    let inactive_tile_start = 2 * num_tile_candidates_per_row;
    assert_eq!(
        tile_token_ids.read_typed::<i32>(inactive_tile_start, num_tile_candidates_per_row),
        vec![tile_token_canary; num_tile_candidates_per_row]
    );
    assert_eq!(
        tile_logits.read_typed::<f32>(inactive_tile_start, num_tile_candidates_per_row),
        vec![tile_logit_canary; num_tile_candidates_per_row]
    );

    submit(3);
    assert_active_outputs(3);
    let full_tokens = token_ids.read_typed::<i32>(0, 3);
    let full_probs = token_probs.read_typed::<f32>(0, 3);
    let full_tile_tokens = tile_token_ids.read_typed::<i32>(0, num_tile_candidates);
    let full_tile_logits = tile_logits.read_typed::<f32>(0, num_tile_candidates);

    logits.write_typed(2 * shape.vocab_size as usize, &[f32::NAN; 17]);
    submit(2);
    assert_active_outputs(2);
    assert_eq!(token_ids.read_typed::<i32>(2, 1), full_tokens[2..]);
    assert_eq!(token_probs.read_typed::<f32>(2, 1), full_probs[2..]);
    assert_eq!(
        tile_token_ids.read_typed::<i32>(inactive_tile_start, num_tile_candidates_per_row),
        full_tile_tokens[inactive_tile_start..]
    );
    assert_eq!(
        tile_logits.read_typed::<f32>(inactive_tile_start, num_tile_candidates_per_row),
        full_tile_logits[inactive_tile_start..]
    );
}

#[test]
fn test_row_offset() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = TopKSampleShape {
        num_total_sampling_inputs: 1,
        vocab_size: 4,
        top_k: 1,
    };
    let logits = Buffer::from_slice(
        &device,
        &[
            0.0_f32, 9.0, 0.0, 0.0, // ignored row
            0.0, 0.0, 11.0, 0.0, // active row
        ],
    );
    let tile_token_ids = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<i32>());
    let tile_logits = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<f32>());
    let topk = TopKMapKernels::new(&device);
    let logits_dtype = crate::metal::Dtype::Float32;
    let mut builder = stream.create_replay_program();
    builder.record(topk.invoke_replay(
        shape,
        logits_dtype,
        TopKSamplingOperation::Sample,
        TopKMapBuffers {
            logits: &logits,
            logits_offset_bytes: shape.vocab_size as usize * size_of::<f32>(),
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
        },
    ));
    let replay = builder.build();
    let mut arguments = ReplayArguments::new();
    topk.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    stream.submit_replay_with_arguments(&replay, &arguments).wait();

    assert_eq!(tile_token_ids.read_typed::<i32>(0, 1), vec![2]);
}

#[test]
fn test_fused_distribution() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = TopKSampleShape {
        num_total_sampling_inputs: 2,
        vocab_size: 8,
        top_k: 4,
    };
    let logits_values = vec![
        0.2f32, 1.0, 1.8, 2.5, 2.1, 0.3, -0.2, 1.4, //
        2.2, 0.1, 1.7, 0.9, 2.8, 1.2, -0.4, 2.0,
    ];
    let logits = Buffer::from_slice(&device, &logits_values);
    let tile_token_ids = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<i32>());
    let tile_logits = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<f32>());
    let sampled_token_ids = Buffer::new_zeroed(&device, 2 * size_of::<i32>());
    let sampled_token_probs = Buffer::new_zeroed(&device, 2 * size_of::<f32>());
    let distribution_token_ids = Buffer::new_zeroed(&device, 2 * shape.top_k as usize * size_of::<i32>());
    let distribution_probs = Buffer::new_zeroed(&device, 2 * shape.top_k as usize * size_of::<f32>());
    let output_distribution_indices = Buffer::from_slice(&device, &[1_u32, 0]);
    let runtime_params = Buffer::new_zeroed(&device, 2 * 6 * size_of::<u32>());
    let configs = [
        SamplerConfig {
            temperature: 0.8,
            top_k: 1,
            top_p: 1.0,
            seed: 7,
        },
        SamplerConfig {
            temperature: 0.9,
            top_k: 4,
            top_p: 0.8,
            seed: 19,
        },
    ];
    write_sampling_runtime_params(&runtime_params, 0, &configs[0], 11, SamplingDomain::Target);
    write_sampling_runtime_params(&runtime_params, 1, &configs[1], 29, SamplingDomain::Draft);

    let topk = TopKMapKernels::new(&device);
    let logits_dtype = crate::metal::Dtype::Float32;
    let sample_and_write_distribution = TopKReduceKernels::new(&device);
    let mut builder = stream.create_replay_program();
    builder.record(topk.invoke_replay(
        shape,
        logits_dtype,
        TopKSamplingOperation::SampleAndWriteDistribution,
        TopKMapBuffers {
            logits: &logits,
            logits_offset_bytes: 0,
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
        },
    ));
    builder.record_with_barrier_before(sample_and_write_distribution.invoke_sample_and_write_distribution(
        shape,
        TopKSampleAndWriteDistributionBuffers {
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
            sampled_token_ids: &sampled_token_ids,
            sampled_token_probs: &sampled_token_probs,
            distribution_token_ids: &distribution_token_ids,
            distribution_probs: &distribution_probs,
            runtime_params: &runtime_params,
            output_distribution_indices: &output_distribution_indices,
            max_k: shape.top_k,
            num_output_distributions: 2,
        },
    ));
    let program = builder.build();
    let mut arguments = ReplayArguments::new();
    topk.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    sample_and_write_distribution.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    stream.submit_replay_with_arguments(&program, &arguments).wait();

    let expected = configs
        .iter()
        .enumerate()
        .map(|(row, config)| {
            sparse_sample_row_with_domain_reference(
                config,
                &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
                config.top_k,
                [11, 29][row],
                [SamplingDomain::Target, SamplingDomain::Draft][row],
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sampled_token_ids.read_typed::<i32>(0, 2),
        expected.iter().map(|row| row.sampled_token as i32).collect::<Vec<_>>()
    );
    assert_close(
        &sampled_token_probs.read_typed::<f32>(0, 2),
        &expected.iter().map(|row| row.sampled_prob).collect::<Vec<_>>(),
        1.0e-5,
    );
    let distribution_ids = distribution_token_ids.read_typed::<i32>(0, 8);
    let distribution_probs = distribution_probs.read_typed::<f32>(0, 8);
    assert_eq!(&distribution_ids[0..4], expected[1].prob_token_ids.as_slice());
    assert_close(&distribution_probs[0..4], &expected[1].prob_values, 1.0e-5);
    assert_eq!(distribution_ids[4], expected[0].prob_token_ids[0]);
    assert_close(&distribution_probs[4..5], &expected[0].prob_values[0..1], 1.0e-5);
}

#[test]
fn test_distribution_slots() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = TopKSampleShape {
        num_total_sampling_inputs: 3,
        vocab_size: 17,
        top_k: 5,
    };
    let random_seed = 0xE6B4_2A17;
    let logits_values = generated_logits(
        shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
        random_seed,
    );
    let logits = Buffer::from_slice(&device, &logits_values);
    let tile_token_ids = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<i32>());
    let tile_logits = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<f32>());
    let output_row_offset = 1;
    let max_k = 8;
    let num_output_distributions = shape.num_total_sampling_inputs + output_row_offset;
    let output_distribution_indices = Buffer::from_slice(
        &device,
        &(0..shape.num_total_sampling_inputs)
            .map(|row| row + output_row_offset)
            .collect::<Vec<_>>(),
    );
    let distribution_token_ids = Buffer::new_zeroed(
        &device,
        num_output_distributions as usize * max_k as usize * size_of::<i32>(),
    );
    let distribution_probs = Buffer::new_zeroed(
        &device,
        num_output_distributions as usize * max_k as usize * size_of::<f32>(),
    );
    let topk = TopKMapKernels::new(&device);
    let logits_dtype = crate::metal::Dtype::Float32;
    let write_distribution = TopKReduceKernels::new(&device);
    let temperature = 0.9;
    let top_p = 0.82;
    let runtime_params = sampling_runtime_params(
        &device,
        shape.num_total_sampling_inputs,
        temperature,
        top_p,
        1,
        shape.top_k,
    );

    let mut builder = stream.create_replay_program();
    builder.record(topk.invoke_replay(
        shape,
        logits_dtype,
        TopKSamplingOperation::WriteDistribution,
        TopKMapBuffers {
            logits: &logits,
            logits_offset_bytes: 0,
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
        },
    ));
    builder.record_with_barrier_before(write_distribution.invoke_write_distribution(
        shape,
        TopKWriteDistributionBuffers {
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
            distribution_token_ids: &distribution_token_ids,
            distribution_probs: &distribution_probs,
            runtime_params: &runtime_params,
            output_distribution_indices: &output_distribution_indices,
            max_k,
            num_output_distributions,
        },
    ));
    let program = builder.build();
    let mut arguments = ReplayArguments::new();
    topk.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    write_distribution.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    stream.submit_replay_with_arguments(&program, &arguments).wait();

    let actual_tokens = distribution_token_ids.read_typed::<i32>(0, num_output_distributions as usize * max_k as usize);
    let actual_probs = distribution_probs.read_typed::<f32>(0, num_output_distributions as usize * max_k as usize);
    let config = SamplerConfig {
        temperature,
        top_k: shape.top_k as usize,
        top_p,
        seed: 1,
    };
    for row in 0..shape.num_total_sampling_inputs as usize {
        let expected = sparse_sample_row_reference(
            &config,
            &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
            shape.top_k as usize,
            row as u32,
        );
        let start = (row + output_row_offset as usize) * max_k as usize;
        assert_eq!(
            &actual_tokens[start..start + shape.top_k as usize],
            expected.prob_token_ids.as_slice()
        );
        assert_close(
            &actual_probs[start..start + shape.top_k as usize],
            &expected.prob_values,
            1.0e-5,
        );
    }
}

#[test]
fn test_bf16_reduction() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = TopKSampleShape {
        num_total_sampling_inputs: 2,
        vocab_size: 257,
        top_k: 20,
    };
    let random_seed = 0x4F91_C3E8;
    let sample_seed = 0xB72D_5A60;
    let logits_values = generated_logits(
        shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
        random_seed,
    );
    let logits = bf16_buffer(&device, &logits_values);
    let tile_token_ids = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<i32>());
    let tile_logits = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<f32>());
    let token_ids = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<i32>());
    let token_probs = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<f32>());
    let topk = TopKMapKernels::new(&device);
    let logits_dtype = crate::metal::Dtype::Bfloat16;
    let sample = TopKReduceKernels::new(&device);
    let temperature = 0.7;
    let top_p = 0.8;
    let runtime_params = sampling_runtime_params(
        &device,
        shape.num_total_sampling_inputs,
        temperature,
        top_p,
        sample_seed,
        shape.top_k,
    );

    let mut builder = stream.create_replay_program();
    builder.record(topk.invoke_replay(
        shape,
        logits_dtype,
        TopKSamplingOperation::Sample,
        TopKMapBuffers {
            logits: &logits,
            logits_offset_bytes: 0,
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
        },
    ));
    builder.record_with_barrier_before(sample.invoke_sample(
        shape,
        TopKSampleBuffers {
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
            token_ids: &token_ids,
            token_probs: &token_probs,
            runtime_params: &runtime_params,
        },
    ));
    let program = builder.build();
    let mut arguments = ReplayArguments::new();
    topk.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    sample.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    stream.submit_replay_with_arguments(&program, &arguments).wait();

    assert_sample_matches_bf16_reference(
        shape,
        &logits_values,
        &token_ids.read_typed::<i32>(0, shape.num_total_sampling_inputs as usize),
        &token_probs.read_typed::<f32>(0, shape.num_total_sampling_inputs as usize),
        SamplerConfig {
            temperature,
            top_k: shape.top_k as usize,
            top_p,
            seed: sample_seed,
        },
    );
}

#[test]
fn test_f32_bitonic() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = TopKSampleShape {
        num_total_sampling_inputs: 2,
        vocab_size: 257,
        top_k: 64,
    };
    let random_seed = 0xD03A_8F51;
    let sample_seed = 0x2C76_EB94;
    let logits_values = generated_logits(
        shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
        random_seed,
    );
    let logits = Buffer::from_slice(&device, &logits_values);
    let tile_token_ids = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<i32>());
    let tile_logits = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<f32>());
    let token_ids = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<i32>());
    let token_probs = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<f32>());
    let topk = TopKMapKernels::new(&device);
    let logits_dtype = crate::metal::Dtype::Float32;
    let sample = TopKReduceKernels::new(&device);
    let temperature = 0.7;
    let top_p = 0.8;
    let runtime_params = sampling_runtime_params(
        &device,
        shape.num_total_sampling_inputs,
        temperature,
        top_p,
        sample_seed,
        shape.top_k,
    );

    let mut builder = stream.create_replay_program();
    builder.record(topk.invoke_replay(
        shape,
        logits_dtype,
        TopKSamplingOperation::Sample,
        TopKMapBuffers {
            logits: &logits,
            logits_offset_bytes: 0,
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
        },
    ));
    builder.record_with_barrier_before(sample.invoke_sample(
        shape,
        TopKSampleBuffers {
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
            token_ids: &token_ids,
            token_probs: &token_probs,
            runtime_params: &runtime_params,
        },
    ));
    let program = builder.build();
    let mut arguments = ReplayArguments::new();
    topk.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    sample.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    stream.submit_replay_with_arguments(&program, &arguments).wait();

    let config = SamplerConfig {
        temperature,
        top_k: shape.top_k as usize,
        top_p,
        seed: sample_seed,
    };
    let actual_tokens = token_ids.read_typed::<i32>(0, shape.num_total_sampling_inputs as usize);
    let actual_probs = token_probs.read_typed::<f32>(0, shape.num_total_sampling_inputs as usize);
    for row in 0..shape.num_total_sampling_inputs as usize {
        let expected = sparse_sample_row_reference(
            &config,
            &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
            shape.top_k as usize,
            row as u32,
        );
        assert_eq!(actual_tokens[row], expected.sampled_token as i32, "row={row}");
        assert!(
            (actual_probs[row] - expected.sampled_prob).abs() <= 1.0e-5,
            "row={row} actual_prob={} expected_prob={}",
            actual_probs[row],
            expected.sampled_prob
        );
    }
}

#[test]
fn test_bf16_bitonic_distribution() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = TopKSampleShape {
        num_total_sampling_inputs: 2,
        vocab_size: 257,
        top_k: 64,
    };
    let random_seed = 0x8A15_6D3F;
    let logits_values = generated_logits(
        shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
        random_seed,
    );
    let logits = bf16_buffer(&device, &logits_values);
    let tile_token_ids = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<i32>());
    let tile_logits = Buffer::new_zeroed(&device, partial_candidate_count(shape) * size_of::<f32>());
    let distribution_token_ids = Buffer::new_zeroed(
        &device,
        shape.num_total_sampling_inputs as usize * shape.top_k as usize * size_of::<i32>(),
    );
    let distribution_probs = Buffer::new_zeroed(
        &device,
        shape.num_total_sampling_inputs as usize * shape.top_k as usize * size_of::<f32>(),
    );
    let output_distribution_indices =
        Buffer::from_slice(&device, &(0..shape.num_total_sampling_inputs).collect::<Vec<_>>());
    let topk = TopKMapKernels::new(&device);
    let logits_dtype = crate::metal::Dtype::Bfloat16;
    let write_distribution = TopKReduceKernels::new(&device);
    let temperature = 0.7;
    let top_p = 0.8;
    let runtime_params = sampling_runtime_params(
        &device,
        shape.num_total_sampling_inputs,
        temperature,
        top_p,
        1,
        shape.top_k,
    );

    let mut builder = stream.create_replay_program();
    builder.record(topk.invoke_replay(
        shape,
        logits_dtype,
        TopKSamplingOperation::WriteDistribution,
        TopKMapBuffers {
            logits: &logits,
            logits_offset_bytes: 0,
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
        },
    ));
    builder.record_with_barrier_before(write_distribution.invoke_write_distribution(
        shape,
        TopKWriteDistributionBuffers {
            tile_token_ids: &tile_token_ids,
            tile_logits: &tile_logits,
            distribution_token_ids: &distribution_token_ids,
            distribution_probs: &distribution_probs,
            runtime_params: &runtime_params,
            output_distribution_indices: &output_distribution_indices,
            max_k: shape.top_k,
            num_output_distributions: shape.num_total_sampling_inputs,
        },
    ));
    let program = builder.build();
    let mut arguments = ReplayArguments::new();
    topk.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    write_distribution.add_replay_arguments(shape, shape.num_total_sampling_inputs, &mut arguments);
    stream.submit_replay_with_arguments(&program, &arguments).wait();

    let actual_tokens =
        distribution_token_ids.read_typed::<i32>(0, shape.num_total_sampling_inputs as usize * shape.top_k as usize);
    let actual_probs =
        distribution_probs.read_typed::<f32>(0, shape.num_total_sampling_inputs as usize * shape.top_k as usize);
    let config = SamplerConfig {
        temperature,
        top_k: shape.top_k as usize,
        top_p,
        seed: 1,
    };
    for row in 0..shape.num_total_sampling_inputs as usize {
        let expected = sparse_sample_row_reference(
            &config,
            &bf16_rounded_logits(
                &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
            ),
            shape.top_k as usize,
            row as u32,
        );
        let start = row * shape.top_k as usize;
        assert_eq!(
            &actual_tokens[start..start + shape.top_k as usize],
            expected.prob_token_ids.as_slice(),
            "row={row}"
        );
        assert_close(
            &actual_probs[start..start + shape.top_k as usize],
            &expected.prob_values,
            1.0e-5,
        );
    }
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
    Buffer::from_slice(device, &bits)
}

fn bf16_rounded_logits(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
}

fn assert_sample_matches_bf16_reference(
    shape: TopKSampleShape,
    logits_values: &[f32],
    actual_tokens: &[i32],
    actual_probs: &[f32],
    config: SamplerConfig,
) {
    for row in 0..shape.num_total_sampling_inputs as usize {
        let expected = sparse_sample_row_reference(
            &config,
            &bf16_rounded_logits(
                &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
            ),
            shape.top_k as usize,
            row as u32,
        );
        assert_eq!(actual_tokens[row], expected.sampled_token as i32, "row={row}");
        assert!(
            (actual_probs[row] - expected.sampled_prob).abs() <= 1.0e-5,
            "row={row} actual_prob={} expected_prob={}",
            actual_probs[row],
            expected.sampled_prob
        );
    }
}

fn generated_logits(count: usize, random_seed: u32) -> Vec<f32> {
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 4_194_304.0) - 2.0
        })
        .collect()
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
