use std::mem::size_of;

use half::bf16;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::dense::reference::QuantizedDenseMLPReferenceWeights;
use inference_executor_core::mlp::dense::reference::quantized_dense_mlp_reference;

use super::*;
use crate::metal::Buffer;
use crate::metal::ReplayArguments;
use crate::metal::ReplayProgram;
use crate::metal::Stream;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.dense_mlp.num_active_tokens");
const HIDDEN_POISON: u16 = 0x7fc1;
const GATE_UP_CANARY: u16 = 0x3555;
const SWIGLU_CANARY: u16 = 0x3aaa;
const OUTPUT_CANARY: u16 = 0x3c00;

#[test]
fn test_swiglu_constants_have_explicit_thread_block_scope() {
    let constants = SwiGLUKernelConstants::new(Dtype::Bfloat16);
    assert_eq!(constants.io_dtype, Dtype::Bfloat16);
    assert_eq!(constants.thread_block.required_threads, 256);
}

#[test]
fn test_fixed() {
    let config = Config {
        hidden_dim: 64,
        intermediate_dim: 64,
        group_size: 32,
        bits: 4,
        dtype: Dtype::Bfloat16,
    };
    let shape = Shape { num_total_tokens: 4 };
    let (device, compute) = create_dense_mlp_compute(config);
    let stream = Stream::new(&device);
    let gate_up_config = config.gate_up_config();
    let down_config = config.down_config();
    let hidden_values = hidden_fixture(shape.num_total_tokens as usize, config.hidden_dim as usize);
    let hidden_state = bf16_buffer(&device, &hidden_values);
    let gate_up_weight_values = quantized_weight_values(gate_up_config.weight_bytes());
    let gate_up_weight = Buffer::from_slice(&device, &gate_up_weight_values);
    let gate_up_scale_values = affine_param_fixture(gate_up_config.scale_or_bias_bytes() / size_of::<u16>());
    let gate_up_scales = bf16_buffer(&device, &gate_up_scale_values);
    let gate_up_bias_values = zero_fixture(gate_up_config.scale_or_bias_bytes() / size_of::<u16>());
    let gate_up_biases = bf16_buffer(&device, &gate_up_bias_values);
    let down_weight_values = quantized_weight_values(down_config.weight_bytes());
    let down_weight = Buffer::from_slice(&device, &down_weight_values);
    let down_scale_values = affine_param_fixture(down_config.scale_or_bias_bytes() / size_of::<u16>());
    let down_scales = bf16_buffer(&device, &down_scale_values);
    let down_bias_values = zero_fixture(down_config.scale_or_bias_bytes() / size_of::<u16>());
    let down_biases = bf16_buffer(&device, &down_bias_values);
    let weights = Weights {
        gate_up_weight: &gate_up_weight,
        gate_up_scales: &gate_up_scales,
        gate_up_biases: &gate_up_biases,
        down_weight: &down_weight,
        down_scales: &down_scales,
        down_biases: &down_biases,
    };

    let replay_output = Buffer::new_zeroed(&device, config.output_bytes(shape));
    let replay_gate_up = Buffer::new_zeroed(&device, config.gate_up_output_bytes(shape));
    let replay_swiglu = Buffer::new_zeroed(&device, config.swiglu_bytes(shape));
    let mut builder = stream.create_replay_program();
    builder.record(compute.invoke(
        shape,
        Buffers {
            hidden_state: &hidden_state,
            next_hidden_state: &replay_output,
        },
        Scratch {
            gate_up: &replay_gate_up,
            swiglu: &replay_swiglu,
        },
        weights,
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();

    let expected = quantized_dense_mlp_reference(
        &DenseMLPCore {
            model_layer_index: 0,
            hidden_dim: config.hidden_dim as usize,
            intermediate_dim: config.intermediate_dim as usize,
        },
        &hidden_values
            .iter()
            .map(|value| bf16::from_f32(*value).to_f32())
            .collect::<Vec<_>>(),
        shape.num_total_tokens as usize,
        config.group_size as usize,
        config.bits as usize,
        QuantizedDenseMLPReferenceWeights {
            gate_up_weight: &gate_up_weight_values,
            gate_up_scales: &bf16_values(&gate_up_scale_values),
            gate_up_biases: &bf16_values(&gate_up_bias_values),
            down_weight: &down_weight_values,
            down_scales: &bf16_values(&down_scale_values),
            down_biases: &bf16_values(&down_bias_values),
        },
    );
    let expected = expected
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let actual = replay_output
        .read_typed::<u16>(0, config.output_bytes(shape) / size_of::<u16>())
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    assert_close_rel(&actual, &expected, 2.0e-5, 8.0e-3);
}

#[test]
fn test_random() {
    let random_seed = 0x5D2A_91C7;
    let config = Config {
        hidden_dim: 64,
        intermediate_dim: 4160,
        group_size: 32,
        bits: 4,
        dtype: Dtype::Bfloat16,
    };
    let shape = Shape { num_total_tokens: 7 };
    let (device, compute) = create_dense_mlp_compute(config);
    let stream = Stream::new(&device);
    let gate_up_config = config.gate_up_config();
    let down_config = config.down_config();
    let hidden_values = generated_values(
        shape.num_total_tokens as usize * config.hidden_dim as usize,
        random_seed,
    );
    let hidden_state = bf16_buffer(&device, &hidden_values);
    let gate_up_weight_values = generated_bytes(gate_up_config.weight_bytes(), random_seed.wrapping_add(1));
    let gate_up_weight = Buffer::from_slice(&device, &gate_up_weight_values);
    let gate_up_scale_values = generated_scales(
        gate_up_config.scale_or_bias_bytes() / size_of::<u16>(),
        random_seed.wrapping_add(2),
    );
    let gate_up_scales = bf16_buffer(&device, &gate_up_scale_values);
    let gate_up_bias_values = generated_biases(
        gate_up_config.scale_or_bias_bytes() / size_of::<u16>(),
        random_seed.wrapping_add(3),
    );
    let gate_up_biases = bf16_buffer(&device, &gate_up_bias_values);
    let down_weight_values = generated_bytes(down_config.weight_bytes(), random_seed.wrapping_add(4));
    let down_weight = Buffer::from_slice(&device, &down_weight_values);
    let down_scale_values = generated_scales(
        down_config.scale_or_bias_bytes() / size_of::<u16>(),
        random_seed.wrapping_add(5),
    );
    let down_scales = bf16_buffer(&device, &down_scale_values);
    let down_bias_values = generated_biases(
        down_config.scale_or_bias_bytes() / size_of::<u16>(),
        random_seed.wrapping_add(6),
    );
    let down_biases = bf16_buffer(&device, &down_bias_values);

    let replay_output = Buffer::new_zeroed(&device, config.output_bytes(shape));
    let replay_gate_up = Buffer::new_zeroed(&device, config.gate_up_output_bytes(shape));
    let replay_swiglu = Buffer::new_zeroed(&device, config.swiglu_bytes(shape));
    let mut builder = stream.create_replay_program();
    builder.record(compute.invoke(
        shape,
        Buffers {
            hidden_state: &hidden_state,
            next_hidden_state: &replay_output,
        },
        Scratch {
            gate_up: &replay_gate_up,
            swiglu: &replay_swiglu,
        },
        Weights {
            gate_up_weight: &gate_up_weight,
            gate_up_scales: &gate_up_scales,
            gate_up_biases: &gate_up_biases,
            down_weight: &down_weight,
            down_scales: &down_scales,
            down_biases: &down_biases,
        },
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();

    let expected = quantized_dense_mlp_reference(
        &DenseMLPCore {
            model_layer_index: 0,
            hidden_dim: config.hidden_dim as usize,
            intermediate_dim: config.intermediate_dim as usize,
        },
        &bf16_values(&hidden_values),
        shape.num_total_tokens as usize,
        config.group_size as usize,
        config.bits as usize,
        QuantizedDenseMLPReferenceWeights {
            gate_up_weight: &gate_up_weight_values,
            gate_up_scales: &bf16_values(&gate_up_scale_values),
            gate_up_biases: &bf16_values(&gate_up_bias_values),
            down_weight: &down_weight_values,
            down_scales: &bf16_values(&down_scale_values),
            down_biases: &bf16_values(&down_bias_values),
        },
    );
    let expected = expected
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let actual = replay_output
        .read_typed::<u16>(0, config.output_bytes(shape) / size_of::<u16>())
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    assert_close_rel(&actual, &expected, 2.0e-5, 8.0e-3);
}

#[test]
fn test_bucketed_replay_preserves_poisoned_tails_across_topologies_and_shrink() {
    let fixture = BucketedDenseMLPFixture::new(20);
    for (num_total_tokens, num_active_tokens, seed) in [
        (4, 3, 0x1000_0001),
        (8, 7, 0x1000_0002),
        (12, 9, 0x1000_0003),
        (20, 17, 0x1000_0004),
    ] {
        fixture.reset_canaries();
        let hidden = fixture.write_hidden(num_active_tokens, seed);
        let replay = fixture.bucketed_replay(num_total_tokens);
        fixture.submit(&replay, num_active_tokens);
        fixture.assert_active_output(&hidden, num_active_tokens);
        fixture.assert_canary_tails(num_active_tokens);
    }

    let replay = fixture.bucketed_replay(20);
    fixture.reset_canaries();
    let first_hidden = fixture.write_hidden(18, 0x2000_0001);
    fixture.submit(&replay, 18);
    fixture.assert_active_output(&first_hidden, 18);
    fixture.assert_canary_tails(18);

    let full_hidden = fixture.write_hidden(20, 0x2000_0002);
    fixture.submit(&replay, 20);
    fixture.assert_active_output(&full_hidden, 20);
    let full_gate_up = fixture.read_gate_up();
    let full_swiglu = fixture.read_swiglu();
    let full_output = fixture.read_output();

    let smaller_hidden = fixture.write_hidden(17, 0x2000_0003);
    fixture.submit(&replay, 17);
    fixture.assert_active_output(&smaller_hidden, 17);
    fixture.assert_tails_equal(17, &full_gate_up, &full_swiglu, &full_output);
}

struct BucketedDenseMLPFixture {
    stream: Stream,
    config: Config,
    compute: Compute,
    num_allocated_tokens: u32,
    hidden_state: Buffer,
    next_hidden_state: Buffer,
    gate_up: Buffer,
    swiglu: Buffer,
    weights: BucketedDenseMLPWeights,
}

impl BucketedDenseMLPFixture {
    fn new(num_allocated_tokens: u32) -> Self {
        let config = bucket_test_config();
        let shape = Shape {
            num_total_tokens: num_allocated_tokens,
        };
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let compute = Compute::new(&device, config);
        let weights = BucketedDenseMLPWeights::new(&device, config);
        Self {
            hidden_state: Buffer::new_zeroed(&device, config.input_bytes(shape)),
            next_hidden_state: Buffer::new_zeroed(&device, config.output_bytes(shape)),
            gate_up: Buffer::new_zeroed(&device, config.gate_up_output_bytes(shape)),
            swiglu: Buffer::new_zeroed(&device, config.swiglu_bytes(shape)),
            stream,
            config,
            compute,
            num_allocated_tokens,
            weights,
        }
    }

    fn bucketed_replay(&self, num_total_tokens: u32) -> ReplayProgram {
        let mut builder = self.stream.create_replay_program();
        builder.record(self.compute.invoke_bucketed(
            num_total_tokens,
            NUM_ACTIVE_TOKENS,
            self.buffers(),
            self.scratch(),
            self.weights.as_borrowed(),
        ));
        builder.build()
    }

    fn buffers(&self) -> Buffers<'_> {
        Buffers {
            hidden_state: &self.hidden_state,
            next_hidden_state: &self.next_hidden_state,
        }
    }

    fn scratch(&self) -> Scratch<'_> {
        Scratch {
            gate_up: &self.gate_up,
            swiglu: &self.swiglu,
        }
    }

    fn reset_canaries(&self) {
        self.gate_up.write_typed(
            0,
            &vec![GATE_UP_CANARY; self.num_allocated_tokens as usize * self.config.intermediate_dim as usize * 2],
        );
        self.swiglu.write_typed(
            0,
            &vec![SWIGLU_CANARY; self.num_allocated_tokens as usize * self.config.intermediate_dim as usize],
        );
        self.next_hidden_state.write_typed(
            0,
            &vec![OUTPUT_CANARY; self.num_allocated_tokens as usize * self.config.hidden_dim as usize],
        );
    }

    fn write_hidden(&self, num_active_tokens: u32, seed: u32) -> Vec<f32> {
        assert!(num_active_tokens <= self.num_allocated_tokens);
        let num_active_values = num_active_tokens as usize * self.config.hidden_dim as usize;
        let active_values = bf16_values(&generated_values(num_active_values, seed));
        let mut all_bits = vec![HIDDEN_POISON; self.num_allocated_tokens as usize * self.config.hidden_dim as usize];
        for (bits, value) in all_bits.iter_mut().zip(&active_values) {
            *bits = bf16::from_f32(*value).to_bits();
        }
        self.hidden_state.write_typed(0, &all_bits);
        active_values
    }

    fn submit(&self, replay: &ReplayProgram, num_active_tokens: u32) {
        let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens);
        self.stream.submit_replay_with_arguments(replay, &arguments).wait();
    }

    fn assert_active_output(&self, hidden: &[f32], num_active_tokens: u32) {
        let num_output_values = num_active_tokens as usize * self.config.hidden_dim as usize;
        let expected = quantized_dense_mlp_reference(
            &DenseMLPCore {
                model_layer_index: 0,
                hidden_dim: self.config.hidden_dim as usize,
                intermediate_dim: self.config.intermediate_dim as usize,
            },
            hidden,
            num_active_tokens as usize,
            self.config.group_size as usize,
            self.config.bits as usize,
            self.weights.as_reference(),
        )
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
        let actual = self
            .next_hidden_state
            .read_typed::<u16>(0, num_output_values)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        assert_close_rel(&actual, &expected, 2.0e-5, 8.0e-3);
    }

    fn assert_canary_tails(&self, num_active_tokens: u32) {
        let gate_up_tail = num_active_tokens as usize * self.config.intermediate_dim as usize * 2;
        let swiglu_tail = num_active_tokens as usize * self.config.intermediate_dim as usize;
        let output_tail = num_active_tokens as usize * self.config.hidden_dim as usize;
        assert!(
            self.read_gate_up()[gate_up_tail..]
                .iter()
                .all(|&bits| bits == GATE_UP_CANARY)
        );
        assert!(
            self.read_swiglu()[swiglu_tail..]
                .iter()
                .all(|&bits| bits == SWIGLU_CANARY)
        );
        assert!(
            self.read_output()[output_tail..]
                .iter()
                .all(|&bits| bits == OUTPUT_CANARY)
        );
    }

    fn assert_tails_equal(
        &self,
        num_active_tokens: u32,
        expected_gate_up: &[u16],
        expected_swiglu: &[u16],
        expected_output: &[u16],
    ) {
        let gate_up_tail = num_active_tokens as usize * self.config.intermediate_dim as usize * 2;
        let swiglu_tail = num_active_tokens as usize * self.config.intermediate_dim as usize;
        let output_tail = num_active_tokens as usize * self.config.hidden_dim as usize;
        assert_eq!(&self.read_gate_up()[gate_up_tail..], &expected_gate_up[gate_up_tail..]);
        assert_eq!(&self.read_swiglu()[swiglu_tail..], &expected_swiglu[swiglu_tail..]);
        assert_eq!(&self.read_output()[output_tail..], &expected_output[output_tail..]);
    }

    fn read_gate_up(&self) -> Vec<u16> {
        self.gate_up.read_typed(
            0,
            self.num_allocated_tokens as usize * self.config.intermediate_dim as usize * 2,
        )
    }

    fn read_swiglu(&self) -> Vec<u16> {
        self.swiglu.read_typed(
            0,
            self.num_allocated_tokens as usize * self.config.intermediate_dim as usize,
        )
    }

    fn read_output(&self) -> Vec<u16> {
        self.next_hidden_state
            .read_typed(0, self.num_allocated_tokens as usize * self.config.hidden_dim as usize)
    }
}

struct BucketedDenseMLPWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
    gate_up_weight_values: Vec<u8>,
    gate_up_scale_values: Vec<f32>,
    gate_up_bias_values: Vec<f32>,
    down_weight_values: Vec<u8>,
    down_scale_values: Vec<f32>,
    down_bias_values: Vec<f32>,
}

impl BucketedDenseMLPWeights {
    fn new(device: &Device, config: Config) -> Self {
        let gate_up_config = config.gate_up_config();
        let down_config = config.down_config();
        let gate_up_weight_values = generated_bytes(gate_up_config.weight_bytes(), 0x3000_0001);
        let gate_up_scale_values = bf16_values(&generated_scales(
            gate_up_config.scale_or_bias_bytes() / size_of::<u16>(),
            0x3000_0002,
        ));
        let gate_up_bias_values = bf16_values(&generated_biases(
            gate_up_config.scale_or_bias_bytes() / size_of::<u16>(),
            0x3000_0003,
        ));
        let down_weight_values = generated_bytes(down_config.weight_bytes(), 0x3000_0004);
        let down_scale_values = bf16_values(&generated_scales(
            down_config.scale_or_bias_bytes() / size_of::<u16>(),
            0x3000_0005,
        ));
        let down_bias_values = bf16_values(&generated_biases(
            down_config.scale_or_bias_bytes() / size_of::<u16>(),
            0x3000_0006,
        ));
        Self {
            gate_up_weight: Buffer::from_slice(device, &gate_up_weight_values),
            gate_up_scales: bf16_buffer(device, &gate_up_scale_values),
            gate_up_biases: bf16_buffer(device, &gate_up_bias_values),
            down_weight: Buffer::from_slice(device, &down_weight_values),
            down_scales: bf16_buffer(device, &down_scale_values),
            down_biases: bf16_buffer(device, &down_bias_values),
            gate_up_weight_values,
            gate_up_scale_values,
            gate_up_bias_values,
            down_weight_values,
            down_scale_values,
            down_bias_values,
        }
    }

    fn as_borrowed(&self) -> Weights<'_> {
        Weights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }

    fn as_reference(&self) -> QuantizedDenseMLPReferenceWeights<'_> {
        QuantizedDenseMLPReferenceWeights {
            gate_up_weight: &self.gate_up_weight_values,
            gate_up_scales: &self.gate_up_scale_values,
            gate_up_biases: &self.gate_up_bias_values,
            down_weight: &self.down_weight_values,
            down_scales: &self.down_scale_values,
            down_biases: &self.down_bias_values,
        }
    }
}

fn bucket_test_config() -> Config {
    Config {
        hidden_dim: 64,
        intermediate_dim: 4160,
        group_size: 32,
        bits: 4,
        dtype: Dtype::Bfloat16,
    }
}

fn create_dense_mlp_compute(config: Config) -> (Device, Compute) {
    let device = Device::system_default();
    let compute = Compute::new(&device, config);
    (device, compute)
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    let bits = values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &bits)
}

fn hidden_fixture(num_tokens: usize, hidden_dim: usize) -> Vec<f32> {
    (0..num_tokens * hidden_dim)
        .map(|index| ((index * 13 + 5) % 31) as f32 * 0.0625 - 1.0)
        .collect()
}

fn bf16_values(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
}

fn quantized_weight_values(len: usize) -> Vec<u8> {
    (0..len).map(|index| ((index * 13 + 17) & 0xff) as u8).collect()
}

fn affine_param_fixture(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| 0.001 + ((index * 3) % 7) as f32 * 0.0001)
        .collect()
}

fn zero_fixture(len: usize) -> Vec<f32> {
    vec![0.0; len]
}

fn generated_values(count: usize, random_seed: u32) -> Vec<f32> {
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

fn generated_scales(count: usize, random_seed: u32) -> Vec<f32> {
    generated_values(count, random_seed)
        .into_iter()
        .map(|value| 0.0005 + value.abs() * 0.001)
        .collect()
}

fn generated_biases(count: usize, random_seed: u32) -> Vec<f32> {
    generated_values(count, random_seed)
        .into_iter()
        .map(|value| value * 0.0002)
        .collect()
}

fn generated_bytes(count: usize, random_seed: u32) -> Vec<u8> {
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        })
        .collect()
}

fn assert_close_rel(actual: &[f32], expected: &[f32], abs_tolerance: f32, rel_tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        let tolerance = abs_tolerance.max(expected.abs() * rel_tolerance);
        assert!(
            diff <= tolerance,
            "dense MLP output mismatch at {index}: expected={expected} actual={actual} diff={diff} \
             tolerance={tolerance}"
        );
    }
}
