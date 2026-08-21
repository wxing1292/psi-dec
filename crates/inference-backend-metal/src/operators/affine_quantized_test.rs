use half::bf16;
use half::f16;
use inference_executor_core::replay::ReplayBucketPolicy;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::Stream;

const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.affine.num_active_rows");

fn adaptive_config(n: i32, k: i32, dtype: Dtype) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig::same_dtype(n, k, 64, 4, dtype)
}

#[test]
#[should_panic(expected = "expert affine quantized kernels do not yet support mixed dtypes")]
fn test_expert_config_rejects_unimplemented_mixed_dtype_template() {
    ExpertAffineQuantizedConfig {
        num_experts: 2,
        matmul: AffineQuantizedMatmulConfig {
            n: 32,
            k: 32,
            group_size: 32,
            bits: 4,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Float32,
            scale_bias_dtype: Dtype::Bfloat16,
        },
    }
    .validate();
}

#[test]
fn test_adaptive_large_vocabulary_qmm_crossover() {
    assert_eq!(
        Selector::key(adaptive_config(151_936, 2048, Dtype::Bfloat16), 4),
        AffineQuantizedMatmulKernelKind::QmvBn8Bk32
    );
    assert_eq!(
        Selector::key(adaptive_config(151_936, 2048, Dtype::Bfloat16), 5),
        AffineQuantizedMatmulKernelKind::QmmBm16Bn32
    );
    assert_eq!(
        Selector::key(adaptive_config(151_936, 5120, Dtype::Bfloat16), 5),
        AffineQuantizedMatmulKernelKind::QmvBn8Bk32
    );
    assert_eq!(
        Selector::key(adaptive_config(151_936, 5120, Dtype::Bfloat16), 6),
        AffineQuantizedMatmulKernelKind::QmmBm16Bn32
    );
}

#[test]
fn test_adaptive_qmm_tile_crossover() {
    assert_eq!(
        Selector::key(adaptive_config(151_936, 5120, Dtype::Bfloat16), 16),
        AffineQuantizedMatmulKernelKind::QmmBm16Bn32
    );
    assert_eq!(
        Selector::key(adaptive_config(151_936, 5120, Dtype::Bfloat16), 17),
        AffineQuantizedMatmulKernelKind::QmmBm32Bn32
    );
}

#[test]
fn test_adaptive_dense_projection_crossover() {
    let large_projection = adaptive_config(34_816, 5120, Dtype::Bfloat16);
    assert_eq!(
        Selector::key(large_projection, 5),
        AffineQuantizedMatmulKernelKind::QmvBn8Bk32
    );
    assert_eq!(
        Selector::key(large_projection, 6),
        AffineQuantizedMatmulKernelKind::QmmBm8Bn32
    );
    assert_eq!(
        Selector::key(large_projection, 8),
        AffineQuantizedMatmulKernelKind::QmmBm8Bn32
    );
    assert_eq!(
        Selector::key(large_projection, 9),
        AffineQuantizedMatmulKernelKind::QmmBm16Bn32
    );

    let common_projection = adaptive_config(1024, 2048, Dtype::Bfloat16);
    assert_eq!(
        Selector::key(common_projection, 8),
        AffineQuantizedMatmulKernelKind::QmvBn8Bk32
    );
    assert_eq!(
        Selector::key(common_projection, 18),
        AffineQuantizedMatmulKernelKind::QmmBm32Bn32
    );
}

#[test]
fn test_adaptive_topology_boundaries_follow_selector() {
    let cases = [
        (adaptive_config(151_936, 2048, Dtype::Bfloat16), &[5, 17][..]),
        (adaptive_config(151_936, 5120, Dtype::Bfloat16), &[6, 17][..]),
        (adaptive_config(34_816, 5120, Dtype::Bfloat16), &[6, 9, 17][..]),
        (adaptive_config(4096, 4096, Dtype::Bfloat16), &[12, 17][..]),
        (adaptive_config(1024, 2048, Dtype::Bfloat16), &[18][..]),
    ];

    for (config, expected) in cases {
        assert_eq!(&*adaptive_topology_boundaries(config), expected, "config={config:?}");

        let boundaries = adaptive_topology_boundaries(config);
        let policy = ReplayBucketPolicy::with_topology_boundaries(64, &boundaries);
        for num_active_rows in 1..=64 {
            let num_total_rows = policy.capacity(num_active_rows);
            assert_eq!(
                Selector::key(config, num_active_rows as i32),
                Selector::key(config, num_total_rows as i32),
                "config={config:?} num_active_rows={num_active_rows} num_total_rows={num_total_rows}"
            );
        }
    }
}

#[test]
fn test_bucketed_qmv_variants_match_exact_and_preserve_tail() {
    let same_qmv = AffineQuantizedMatmulConfig::same_dtype(4, 32, 32, 8, Dtype::Float32);
    let same_qmv_fast = AffineQuantizedMatmulConfig::same_dtype(8, 512, 64, 8, Dtype::Float32);
    let same_qmv_quad = AffineQuantizedMatmulConfig::same_dtype(9, 64, 64, 8, Dtype::Float32);
    let mixed_qmv = AffineQuantizedMatmulConfig {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };
    let mixed_qmv_fast = AffineQuantizedMatmulConfig {
        n: 8,
        k: 512,
        group_size: 64,
        bits: 8,
        input_dtype: Dtype::Float16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Float32,
    };

    for (config, kind) in [
        (same_qmv, AffineQuantizedMatmulKernelKind::QmvBn8Bk32),
        (same_qmv_fast, AffineQuantizedMatmulKernelKind::QmvBn8Bk32),
        (same_qmv_quad, AffineQuantizedMatmulKernelKind::QmvQuadBn64),
        (mixed_qmv, AffineQuantizedMatmulKernelKind::QmvBn8Bk32),
        (mixed_qmv_fast, AffineQuantizedMatmulKernelKind::QmvBn8Bk32),
    ] {
        assert_bucketed_parity_and_canary(config, kind, 4, 3);
    }
}

#[test]
fn test_bucketed_qmm_variants_match_exact_and_preserve_tail() {
    let same = AffineQuantizedMatmulConfig::same_dtype(32, 32, 32, 8, Dtype::Float32);
    let same_unaligned = AffineQuantizedMatmulConfig::same_dtype(1, 32, 32, 8, Dtype::Float32);
    let same_q4_bf16 = AffineQuantizedMatmulConfig::same_dtype(32, 64, 64, 4, Dtype::Bfloat16);
    let mixed_unaligned = AffineQuantizedMatmulConfig {
        n: 3,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };

    for (config, kind, num_total_rows, num_active_rows) in [
        (same, AffineQuantizedMatmulKernelKind::QmmBm8Bn32, 16, 5),
        (same, AffineQuantizedMatmulKernelKind::QmmBm16Bn32, 32, 9),
        (same, AffineQuantizedMatmulKernelKind::QmmBm32Bn32, 64, 17),
        (same_q4_bf16, AffineQuantizedMatmulKernelKind::QmmBm16Bn32, 32, 9),
        (same_unaligned, AffineQuantizedMatmulKernelKind::QmmBm32Bn32, 64, 17),
        (mixed_unaligned, AffineQuantizedMatmulKernelKind::QmmBm8Bn32, 16, 5),
        (mixed_unaligned, AffineQuantizedMatmulKernelKind::QmmBm16Bn32, 32, 9),
        (mixed_unaligned, AffineQuantizedMatmulKernelKind::QmmBm32Bn32, 64, 17),
    ] {
        assert_bucketed_parity_and_canary(config, kind, num_total_rows, num_active_rows);
    }
}

fn assert_bucketed_parity_and_canary(
    config: AffineQuantizedMatmulConfig,
    kind: AffineQuantizedMatmulKernelKind,
    num_total_rows: i32,
    num_active_rows: i32,
) {
    assert!(matches!(config.bits, 4 | 8));
    assert!(num_active_rows < num_total_rows);
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let input_source = fixture_values(num_total_rows as usize * config.k as usize, 0.00390625);
    let input_values = round_values_to_dtype(&input_source, config.input_dtype);
    let num_weight_values = config.n as usize * config.k as usize;
    let weight_values = if config.bits == 4 {
        fixture_q4_values(num_weight_values)
    } else {
        fixture_weight_bytes(num_weight_values)
    };
    let num_affine_values = config.n as usize * (config.k / config.group_size) as usize;
    let scales = round_values_to_dtype(&fixture_values(num_affine_values, 0.001953125), config.scale_bias_dtype);
    let biases = round_values_to_dtype(
        &fixture_values(num_affine_values, -0.0009765625),
        config.scale_bias_dtype,
    );
    let input = buffer_from_f32(&device, &input_values, config.input_dtype);
    let weight = if config.bits == 4 {
        Buffer::from_slice(&device, &pack_q4(&weight_values))
    } else {
        Buffer::from_slice(&device, &weight_values)
    };
    let scales_buffer = buffer_from_f32(&device, &scales, config.scale_bias_dtype);
    let biases_buffer = buffer_from_f32(&device, &biases, config.scale_bias_dtype);
    let sentinel = round_values_to_dtype(&[-123.0], config.output_dtype)[0];
    let bucketed_output = buffer_from_f32(
        &device,
        &vec![sentinel; num_total_rows as usize * config.n as usize],
        config.output_dtype,
    );
    let exact_output = Buffer::new_zeroed(&device, config.output_bytes(num_active_rows));
    let kernel = AffineQuantizedMatmulKernel::new(&device, config, kind);

    let mut exact_builder = stream.create_replay_program();
    exact_builder.record(kernel.invoke(
        num_active_rows,
        &exact_output,
        0,
        &input,
        0,
        &weight,
        0,
        &scales_buffer,
        0,
        &biases_buffer,
        0,
    ));
    let exact_replay = exact_builder.build();
    stream.submit_replay(&exact_replay).wait();

    let mut bucketed_builder = stream.create_replay_program();
    bucketed_builder.record(kernel.invoke_bucketed(
        num_total_rows as u32,
        NUM_ACTIVE_ROWS,
        &bucketed_output,
        0,
        &input,
        0,
        &weight,
        0,
        &scales_buffer,
        0,
        &biases_buffer,
        0,
    ));
    let bucketed_replay = bucketed_builder.build();
    stream
        .submit_replay_with_arguments(
            &bucketed_replay,
            &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, num_active_rows as u32),
        )
        .wait();

    let num_active_values = num_active_rows as usize * config.n as usize;
    let num_total_values = num_total_rows as usize * config.n as usize;
    let tolerance = output_tolerance(config.output_dtype);
    let exact = read_f32(&exact_output, num_active_values, config.output_dtype);
    let bucketed = read_f32(&bucketed_output, num_active_values, config.output_dtype);
    assert_close_case(&bucketed, &exact, tolerance, config, kind);
    let bucketed_values = read_f32(&bucketed_output, num_total_values, config.output_dtype);
    assert_eq!(
        &bucketed_values[num_active_values..],
        vec![sentinel; num_total_values - num_active_values],
        "bucketed affine wrote inactive output rows: config={config:?} kind={kind:?}"
    );

    stream
        .submit_replay_with_arguments(
            &bucketed_replay,
            &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, num_total_rows as u32),
        )
        .wait();
    let expected = round_values_to_dtype(
        &cpu_affine_reference(config, num_total_rows, &input_values, &weight_values, &scales, &biases),
        config.output_dtype,
    );
    let actual = read_f32(&bucketed_output, num_total_values, config.output_dtype);
    assert_close_case(&actual, &expected, tolerance, config, kind);

    stream
        .submit_replay_with_arguments(
            &bucketed_replay,
            &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, num_active_rows as u32),
        )
        .wait();
    let shrunk = read_f32(&bucketed_output, num_total_values, config.output_dtype);
    assert_close_case(&shrunk[..num_active_values], &exact, tolerance, config, kind);
    assert_eq!(
        &shrunk[num_active_values..],
        &actual[num_active_values..],
        "bucketed affine rewrote rows after the active prefix: config={config:?} kind={kind:?}"
    );
}

fn output_tolerance(dtype: Dtype) -> f32 {
    match dtype {
        Dtype::Float32 => 1.0e-3,
        Dtype::Float16 => 0.02,
        Dtype::Bfloat16 => 0.125,
        _ => unreachable!(),
    }
}

fn execute_matmul(stream: &Stream, invocation: AffineQuantizedMatmulInvocation<'_>) {
    let mut builder = stream.create_replay_program();
    builder.record(invocation);
    let replay = builder.build();
    stream.submit_replay(&replay).wait();
}

#[test]
fn test_adaptive_matmul_supports_all_float_dtype_combinations() {
    const DTYPES: [Dtype; 3] = [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16];

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let max_m = 31;
    let input_source = fixture_values(max_m * 32, 0.00390625);
    let weight = fixture_weight_bytes(8 * 32);
    let scales_source = fixture_values(8, 0.001953125);
    let biases_source = fixture_values(8, -0.0009765625);
    let weight_buffer = Buffer::from_slice(&device, &weight);

    for input_dtype in DTYPES {
        for scale_bias_dtype in DTYPES {
            for output_dtype in DTYPES {
                let config = AffineQuantizedMatmulConfig {
                    n: 8,
                    k: 32,
                    group_size: 32,
                    bits: 8,
                    input_dtype,
                    output_dtype,
                    scale_bias_dtype,
                };
                let input_values = round_values_to_dtype(&input_source, input_dtype);
                let scales = round_values_to_dtype(&scales_source, scale_bias_dtype);
                let biases = round_values_to_dtype(&biases_source, scale_bias_dtype);
                let input = buffer_from_f32(&device, &input_values, input_dtype);
                let scales_buffer = buffer_from_f32(&device, &scales, scale_bias_dtype);
                let biases_buffer = buffer_from_f32(&device, &biases, scale_bias_dtype);
                let matmul = AffineQuantizedMatmul::new(&device, config);

                let cases = [
                    (matmul.registry.get(Selector::qmv_key(config)), 2),
                    (matmul.registry.get(AffineQuantizedMatmulKernelKind::QmmBm8Bn32), 7),
                    (matmul.registry.get(AffineQuantizedMatmulKernelKind::QmmBm16Bn32), 15),
                    (matmul.registry.get(AffineQuantizedMatmulKernelKind::QmmBm32Bn32), 31),
                ];
                for (kernel, m) in cases {
                    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
                    execute_matmul(
                        &stream,
                        kernel.invoke(
                            m,
                            &output,
                            0,
                            &input,
                            0,
                            &weight_buffer,
                            0,
                            &scales_buffer,
                            0,
                            &biases_buffer,
                            0,
                        ),
                    );

                    let actual = read_f32(&output, m as usize * config.n as usize, output_dtype);
                    let expected = round_values_to_dtype(
                        &cpu_affine_reference(config, m, &input_values, &weight, &scales, &biases),
                        output_dtype,
                    );
                    let tolerance = match output_dtype {
                        Dtype::Float32 => 1.0e-3,
                        Dtype::Float16 => 0.02,
                        Dtype::Bfloat16 => 0.125,
                        _ => unreachable!(),
                    };
                    assert_close_case(&actual, &expected, tolerance, config, kernel.kind());
                }
            }
        }
    }
}

#[test]
fn test_qmv_fast_supports_all_float_dtype_combinations() {
    const DTYPES: [Dtype; 3] = [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16];

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 2;
    let input_source = fixture_values(m as usize * 512, 0.00390625);
    let weight = fixture_weight_bytes(8 * 512);
    let scales_source = fixture_values(8 * (512 / 64), 0.001953125);
    let biases_source = fixture_values(8 * (512 / 64), -0.0009765625);
    let weight_buffer = Buffer::from_slice(&device, &weight);

    for input_dtype in DTYPES {
        for scale_bias_dtype in DTYPES {
            for output_dtype in DTYPES {
                let config = AffineQuantizedMatmulConfig {
                    n: 8,
                    k: 512,
                    group_size: 64,
                    bits: 8,
                    input_dtype,
                    output_dtype,
                    scale_bias_dtype,
                };
                let input_values = round_values_to_dtype(&input_source, input_dtype);
                let scales = round_values_to_dtype(&scales_source, scale_bias_dtype);
                let biases = round_values_to_dtype(&biases_source, scale_bias_dtype);
                let input = buffer_from_f32(&device, &input_values, input_dtype);
                let scales_buffer = buffer_from_f32(&device, &scales, scale_bias_dtype);
                let biases_buffer = buffer_from_f32(&device, &biases, scale_bias_dtype);
                let output = Buffer::new_zeroed(&device, config.output_bytes(m));
                let kernel =
                    AffineQuantizedMatmulKernel::new(&device, config, AffineQuantizedMatmulKernelKind::QmvBn8Bk32);
                execute_matmul(
                    &stream,
                    kernel.invoke(
                        m,
                        &output,
                        0,
                        &input,
                        0,
                        &weight_buffer,
                        0,
                        &scales_buffer,
                        0,
                        &biases_buffer,
                        0,
                    ),
                );

                let actual = read_f32(&output, m as usize * config.n as usize, output_dtype);
                let expected = round_values_to_dtype(
                    &cpu_affine_reference(config, m, &input_values, &weight, &scales, &biases),
                    output_dtype,
                );
                let tolerance = match output_dtype {
                    Dtype::Float32 => 1.0e-3,
                    Dtype::Float16 => 0.02,
                    Dtype::Bfloat16 => 0.125,
                    _ => unreachable!(),
                };
                assert_close_case(&actual, &expected, tolerance, config, kernel.kind());
            }
        }
    }
}

#[test]
fn test_qmv_reference() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 2;
    let config = AffineQuantizedMatmulConfig {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };
    let input_f32 = fixture_values(m as usize * config.k as usize, 0.03125);
    let input_bf16 = input_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales = fixture_values(config.n as usize, 0.015625);
    let biases = fixture_values(config.n as usize, -0.0078125);
    let input = Buffer::from_slice(&device, &input_bf16);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales);
    let biases_buffer = Buffer::from_slice(&device, &biases);

    execute_matmul(
        &stream,
        AffineQuantizedMatmulKernel::new(&device, config, Selector::key(config, m)).invoke(
            m,
            &output,
            0,
            &input,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output.read_typed::<f32>(0, m as usize * config.n as usize);
    let expected = cpu_affine_reference(
        config,
        m,
        &input_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &weight,
        &scales,
        &biases,
    );
    assert_close(&actual, &expected, 1.0e-4);
}

#[test]
fn test_qmv_fast_reference() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 2;
    let config = AffineQuantizedMatmulConfig {
        n: 8,
        k: 512,
        group_size: 64,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };
    let input_f32 = fixture_values(m as usize * config.k as usize, 0.00390625);
    let input_bf16 = input_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales = fixture_values(config.n as usize * (config.k / config.group_size) as usize, 0.001953125);
    let biases = fixture_values(
        config.n as usize * (config.k / config.group_size) as usize,
        -0.0009765625,
    );
    let input = Buffer::from_slice(&device, &input_bf16);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales);
    let biases_buffer = Buffer::from_slice(&device, &biases);

    execute_matmul(
        &stream,
        AffineQuantizedMatmulKernel::new(&device, config, Selector::key(config, m)).invoke(
            m,
            &output,
            0,
            &input,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output.read_typed::<f32>(0, m as usize * config.n as usize);
    let expected = cpu_affine_reference(
        config,
        m,
        &input_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &weight,
        &scales,
        &biases,
    );
    assert_close(&actual, &expected, 1.0e-3);
}

#[test]
fn test_qmm_reference() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 18;
    let config = AffineQuantizedMatmulConfig {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };
    let input_f32 = fixture_values(m as usize * config.k as usize, 0.03125);
    let input_bf16 = input_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales = fixture_values(config.n as usize, 0.015625);
    let biases = fixture_values(config.n as usize, -0.0078125);
    let input = Buffer::from_slice(&device, &input_bf16);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales);
    let biases_buffer = Buffer::from_slice(&device, &biases);

    execute_matmul(
        &stream,
        AffineQuantizedMatmulKernel::new(&device, config, Selector::key(config, m)).invoke(
            m,
            &output,
            0,
            &input,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output.read_typed::<f32>(0, m as usize * config.n as usize);
    let expected = cpu_affine_reference(
        config,
        m,
        &input_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &weight,
        &scales,
        &biases,
    );
    assert_close(&actual, &expected, 1.0e-4);
}

#[test]
fn test_qmm_bm8_bn32_q4_bf16_reference() {
    assert_qmm_bm8_bm16_bn32_q4_bf16_reference(8);
}

#[test]
fn test_qmm_bm16_bn32_q4_bf16_reference() {
    assert_qmm_bm8_bm16_bn32_q4_bf16_reference(16);
}

fn assert_qmm_bm8_bm16_bn32_q4_bf16_reference(bm: usize) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 7;
    let config = AffineQuantizedMatmulConfig {
        n: 32,
        k: 64,
        group_size: 64,
        bits: 4,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    };
    let input_f32 = fixture_values(m as usize * config.k as usize, 0.03125);
    let input_bf16 = input_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let weight_values = fixture_q4_values(config.n as usize * config.k as usize);
    let weight = pack_q4(&weight_values);
    let scales_f32 = fixture_values(config.n as usize, 0.015625);
    let biases_f32 = fixture_values(config.n as usize, -0.0078125);
    let scales_bf16 = scales_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let biases_bf16 = biases_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let input = Buffer::from_slice(&device, &input_bf16);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
    let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

    let kernel = match bm {
        8 => AffineQuantizedMatmulKernel::new(&device, config, AffineQuantizedMatmulKernelKind::QmmBm8Bn32),
        16 => AffineQuantizedMatmulKernel::new(&device, config, AffineQuantizedMatmulKernelKind::QmmBm16Bn32),
        _ => panic!("QMM BM=8/16 BN=32 reference requires BM=8 or BM=16"),
    };
    execute_matmul(
        &stream,
        kernel.invoke(
            m,
            &output,
            0,
            &input,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output
        .read_typed::<u16>(0, m as usize * config.n as usize)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = cpu_affine_reference(
        config,
        m,
        &input_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &weight_values,
        &scales_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &biases_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_close(&actual, &expected, 0.125);
}

#[test]
fn test_qmv_bf16() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 1;
    let config = AffineQuantizedMatmulConfig {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Float32,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    };
    let input = fixture_values(m as usize * config.k as usize, 0.03125);
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales_f32 = fixture_values(config.n as usize, 0.015625);
    let biases_f32 = fixture_values(config.n as usize, -0.0078125);
    let scales_bf16 = scales_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let biases_bf16 = biases_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let input_buffer = Buffer::from_slice(&device, &input);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
    let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

    execute_matmul(
        &stream,
        AffineQuantizedMatmulKernel::new(&device, config, Selector::key(config, m)).invoke(
            m,
            &output,
            0,
            &input_buffer,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output
        .read_typed::<u16>(0, m as usize * config.n as usize)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = cpu_affine_reference(
        config,
        m,
        &input,
        &weight,
        &scales_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &biases_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_close(&actual, &expected, 1.0e-4);
}

#[test]
fn test_qmv_fast_bf16() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 2;
    let config = AffineQuantizedMatmulConfig {
        n: 8,
        k: 512,
        group_size: 64,
        bits: 8,
        input_dtype: Dtype::Float32,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    };
    let input = fixture_values(m as usize * config.k as usize, 0.00390625);
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales_f32 = fixture_values(config.n as usize * (config.k / config.group_size) as usize, 0.001953125);
    let biases_f32 = fixture_values(
        config.n as usize * (config.k / config.group_size) as usize,
        -0.0009765625,
    );
    let scales_bf16 = scales_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let biases_bf16 = biases_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let input_buffer = Buffer::from_slice(&device, &input);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
    let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

    execute_matmul(
        &stream,
        AffineQuantizedMatmulKernel::new(&device, config, Selector::key(config, m)).invoke(
            m,
            &output,
            0,
            &input_buffer,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output
        .read_typed::<u16>(0, m as usize * config.n as usize)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = cpu_affine_reference(
        config,
        m,
        &input,
        &weight,
        &scales_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &biases_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_close(&actual, &expected, 1.0e-3);
}

#[test]
fn test_qmm_bf16() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 18;
    let config = AffineQuantizedMatmulConfig {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Float32,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    };
    let input = fixture_values(m as usize * config.k as usize, 0.03125);
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales_f32 = fixture_values(config.n as usize, 0.015625);
    let biases_f32 = fixture_values(config.n as usize, -0.0078125);
    let scales_bf16 = scales_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let biases_bf16 = biases_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let input_buffer = Buffer::from_slice(&device, &input);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
    let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

    execute_matmul(
        &stream,
        AffineQuantizedMatmulKernel::new(&device, config, Selector::key(config, m)).invoke(
            m,
            &output,
            0,
            &input_buffer,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output
        .read_typed::<u16>(0, m as usize * config.n as usize)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = cpu_affine_reference(
        config,
        m,
        &input,
        &weight,
        &scales_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &biases_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_close(&actual, &expected, 1.0e-4);
}

fn cpu_affine_reference(
    config: AffineQuantizedMatmulConfig,
    m: i32,
    input: &[f32],
    weight: &[u8],
    scales: &[f32],
    biases: &[f32],
) -> Vec<f32> {
    let m = m as usize;
    let n = config.n as usize;
    let k = config.k as usize;
    let mut output = vec![0.0_f32; m * n];
    for row in 0..m {
        let input_row = &input[row * k..(row + 1) * k];
        for col in 0..n {
            let weight_row = &weight[col * k..(col + 1) * k];
            let mut value = 0.0_f32;
            for group in 0..(k / config.group_size as usize) {
                let group_start = group * config.group_size as usize;
                let group_end = group_start + config.group_size as usize;
                let input_group = &input_row[group_start..group_end];
                let weight_group = &weight_row[group_start..group_end];
                let input_sum = input_group.iter().copied().sum::<f32>();
                let dot = input_group
                    .iter()
                    .zip(weight_group)
                    .map(|(x, w)| *x * f32::from(*w))
                    .sum::<f32>();
                let affine_index = col * (k / config.group_size as usize) + group;
                value += scales[affine_index] * dot + input_sum * biases[affine_index];
            }
            output[row * n + col] = value;
        }
    }
    output
}

fn fixture_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|index| ((index % 17) as f32 - 8.0) * scale).collect()
}

fn round_values_to_dtype(values: &[f32], dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => values.to_vec(),
        Dtype::Float16 => values.iter().map(|&value| f16::from_f32(value).to_f32()).collect(),
        Dtype::Bfloat16 => values.iter().map(|&value| bf16::from_f32(value).to_f32()).collect(),
        _ => panic!("affine dtype test requires f32, f16, or bf16"),
    }
}

fn buffer_from_f32(device: &Device, values: &[f32], dtype: Dtype) -> Buffer {
    match dtype {
        Dtype::Float32 => Buffer::from_slice(device, values),
        Dtype::Float16 => {
            Buffer::from_slice(
                device,
                &values
                    .iter()
                    .map(|&value| f16::from_f32(value).to_bits())
                    .collect::<Vec<_>>(),
            )
        },
        Dtype::Bfloat16 => {
            Buffer::from_slice(
                device,
                &values
                    .iter()
                    .map(|&value| bf16::from_f32(value).to_bits())
                    .collect::<Vec<_>>(),
            )
        },
        _ => panic!("affine dtype test requires f32, f16, or bf16"),
    }
}

fn read_f32(buffer: &Buffer, len: usize, dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => buffer.read_typed::<f32>(0, len),
        Dtype::Float16 => {
            buffer
                .read_typed::<u16>(0, len)
                .into_iter()
                .map(|bits| f16::from_bits(bits).to_f32())
                .collect()
        },
        Dtype::Bfloat16 => {
            buffer
                .read_typed::<u16>(0, len)
                .into_iter()
                .map(|bits| bf16::from_bits(bits).to_f32())
                .collect()
        },
        _ => panic!("affine dtype test requires f32, f16, or bf16"),
    }
}

fn fixture_weight_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|index| ((index * 7 + 3) % 251) as u8).collect()
}

fn fixture_q4_values(len: usize) -> Vec<u8> {
    (0..len).map(|index| ((index * 7 + 3) % 16) as u8).collect()
}

fn pack_q4(values: &[u8]) -> Vec<u32> {
    assert!(values.len().is_multiple_of(8));
    values
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u32, |word, (index, &value)| word | u32::from(value) << (index * 4))
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "mixed affine mismatch at {index}: actual={actual} expected={expected} diff={diff}"
        );
    }
}

fn assert_close_case(
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
    config: AffineQuantizedMatmulConfig,
    kind: AffineQuantizedMatmulKernelKind,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "affine dtype combination mismatch: config={config:?} kind={kind:?} index={index} actual={actual} \
             expected={expected} diff={diff} tolerance={tolerance}"
        );
    }
}
