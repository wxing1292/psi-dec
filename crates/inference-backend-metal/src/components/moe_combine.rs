use std::mem::size_of;

use super::assert_u32_count_domain;
use super::assert_u32_index_domain;
use super::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const MOE_COMBINE_SOURCE: &str = include_str!("metal/moe_combine.metal");

/// Combines top-k routed expert outputs without the common expert branch.
#[derive(Clone, Copy, Debug)]
pub struct MoECombineWithoutCommonShape {
    pub num_tokens: u32,
    pub num_experts_per_token: u32,
    pub hidden_dim: u32,
    pub dtype: Dtype,
}

impl MoECombineWithoutCommonShape {
    pub fn bf16(num_tokens: u32, num_experts_per_token: u32, hidden_dim: u32) -> Self {
        Self {
            num_tokens,
            num_experts_per_token,
            hidden_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_tokens > 0);
        assert!(self.num_experts_per_token > 0);
        assert!(self.hidden_dim > 0);
        assert_eq!(self.dtype, Dtype::Bfloat16);
        assert_u32_index_domain(self.num_routed_elements(), "MoE combine routed elements");
        assert_u32_count_domain(self.num_output_elements(), "MoE combine output elements");
    }

    fn num_routes(self) -> usize {
        checked_product(
            "MoE combine route count",
            &[self.num_tokens as usize, self.num_experts_per_token as usize],
        )
    }

    fn num_routed_elements(self) -> usize {
        checked_product(
            "MoE combine routed element count",
            &[self.num_routes(), self.hidden_dim as usize],
        )
    }

    fn num_output_elements(self) -> usize {
        checked_product(
            "MoE combine output element count",
            &[self.num_tokens as usize, self.hidden_dim as usize],
        )
    }

    pub fn routed_output_bytes(self) -> usize {
        checked_product(
            "MoE combine routed-output byte length",
            &[self.num_routed_elements(), self.dtype.item_size()],
        )
    }

    pub fn routed_probs_bytes(self) -> usize {
        checked_product(
            "MoE combine routed-probability byte length",
            &[self.num_routes(), size_of::<f32>()],
        )
    }

    pub fn output_bytes(self) -> usize {
        checked_product(
            "MoE combine output byte length",
            &[self.num_output_elements(), self.dtype.item_size()],
        )
    }
}

/// Combines top-k routed expert outputs with the common expert branch.
///
/// This path fuses the routed combine and common expert contribution into one
/// output write.
#[derive(Clone, Copy, Debug)]
pub struct MoECombineWithCommonShape {
    pub num_tokens: u32,
    pub num_experts_per_token: u32,
    pub hidden_dim: u32,
    pub dtype: Dtype,
}

impl MoECombineWithCommonShape {
    pub fn bf16(num_tokens: u32, num_experts_per_token: u32, hidden_dim: u32) -> Self {
        Self {
            num_tokens,
            num_experts_per_token,
            hidden_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_tokens > 0);
        assert!(self.num_experts_per_token > 0);
        assert!(self.hidden_dim > 0);
        assert_eq!(self.dtype, Dtype::Bfloat16);
        assert_u32_index_domain(self.num_routed_elements(), "MoE combine-with-common routed elements");
        assert_u32_count_domain(self.num_output_elements(), "MoE combine-with-common output elements");
    }

    fn num_routes(self) -> usize {
        checked_product(
            "MoE combine-with-common route count",
            &[self.num_tokens as usize, self.num_experts_per_token as usize],
        )
    }

    fn num_routed_elements(self) -> usize {
        checked_product(
            "MoE combine-with-common routed element count",
            &[self.num_routes(), self.hidden_dim as usize],
        )
    }

    fn num_output_elements(self) -> usize {
        checked_product(
            "MoE combine-with-common output element count",
            &[self.num_tokens as usize, self.hidden_dim as usize],
        )
    }

    pub fn routed_output_bytes(self) -> usize {
        checked_product(
            "MoE combine-with-common routed-output byte length",
            &[self.num_routed_elements(), self.dtype.item_size()],
        )
    }

    pub fn routed_probs_bytes(self) -> usize {
        checked_product(
            "MoE combine-with-common routed-probability byte length",
            &[self.num_routes(), size_of::<f32>()],
        )
    }

    pub fn output_bytes(self) -> usize {
        checked_product(
            "MoE combine-with-common output byte length",
            &[self.num_output_elements(), self.dtype.item_size()],
        )
    }
}

pub struct MoECombineWithoutCommonBuffers<'a> {
    pub routed_hidden: &'a Buffer,
    pub routed_probs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct MoECombineWithCommonBuffers<'a> {
    pub routed_hidden: &'a Buffer,
    pub routed_probs: &'a Buffer,
    pub common_hidden: &'a Buffer,
    pub common_gate_logits: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct MoECombineKernel {
    without_common: Kernel,
    with_common: Kernel,
}

impl MoECombineKernel {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            without_common: Kernel::new(device, MOE_COMBINE_SOURCE, "moe_combine_without_common"),
            with_common: Kernel::new(device, MOE_COMBINE_SOURCE, "moe_combine_with_common"),
        }
    }

    pub fn invoke_without_common<'a>(
        &'a self,
        shape: MoECombineWithoutCommonShape,
        buffers: MoECombineWithoutCommonBuffers<'a>,
    ) -> MoECombineWithoutCommonInvocation<'a> {
        MoECombineWithoutCommonInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }

    pub fn invoke_with_common<'a>(
        &'a self,
        shape: MoECombineWithCommonShape,
        buffers: MoECombineWithCommonBuffers<'a>,
    ) -> MoECombineWithCommonInvocation<'a> {
        MoECombineWithCommonInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }
}

pub struct MoECombineWithoutCommonInvocation<'a> {
    kernel: &'a MoECombineKernel,
    shape: MoECombineWithoutCommonShape,
    buffers: MoECombineWithoutCommonBuffers<'a>,
}

impl Operator for MoECombineWithoutCommonInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        debug_validate_without_common_buffers(self.shape, &self.buffers);
        builder.set_kernel(&self.kernel.without_common);
        builder.set_buffer_read(0, self.buffers.routed_hidden, 0);
        builder.set_buffer_read(1, self.buffers.routed_probs, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        builder.set_u32(3, self.shape.num_tokens);
        builder.set_u32(4, self.shape.num_experts_per_token);
        builder.set_u32(5, self.shape.hidden_dim);
        builder.dispatch_1d(self.shape.num_output_elements(), 256);
    }
}

pub struct MoECombineWithCommonInvocation<'a> {
    kernel: &'a MoECombineKernel,
    shape: MoECombineWithCommonShape,
    buffers: MoECombineWithCommonBuffers<'a>,
}

impl Operator for MoECombineWithCommonInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        debug_validate_with_common_buffers(self.shape, &self.buffers);
        builder.set_kernel(&self.kernel.with_common);
        builder.set_buffer_read(0, self.buffers.routed_hidden, 0);
        builder.set_buffer_read(1, self.buffers.routed_probs, 0);
        builder.set_buffer_read(2, self.buffers.common_hidden, 0);
        builder.set_buffer_read(3, self.buffers.common_gate_logits, 0);
        builder.set_buffer_write(4, self.buffers.output, 0);
        builder.set_u32(5, self.shape.num_tokens);
        builder.set_u32(6, self.shape.num_experts_per_token);
        builder.set_u32(7, self.shape.hidden_dim);
        builder.dispatch_1d(self.shape.num_output_elements(), 256);
    }
}

fn debug_validate_without_common_buffers(
    shape: MoECombineWithoutCommonShape,
    buffers: &MoECombineWithoutCommonBuffers<'_>,
) {
    #[cfg(debug_assertions)]
    validate_without_common_buffers(shape, buffers);
}

fn debug_validate_with_common_buffers(shape: MoECombineWithCommonShape, buffers: &MoECombineWithCommonBuffers<'_>) {
    #[cfg(debug_assertions)]
    validate_with_common_buffers(shape, buffers);
}

fn validate_without_common_buffers(shape: MoECombineWithoutCommonShape, buffers: &MoECombineWithoutCommonBuffers<'_>) {
    let routed_output_bytes = shape.routed_output_bytes();
    let routed_probs_bytes = shape.routed_probs_bytes();
    let output_bytes = shape.output_bytes();
    assert!(
        buffers.routed_hidden.len_bytes() >= routed_output_bytes,
        "MoE combine without common routed_hidden buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        routed_output_bytes,
        buffers.routed_hidden.len_bytes()
    );
    assert!(
        buffers.routed_probs.len_bytes() >= routed_probs_bytes,
        "MoE combine without common routed_probs buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        routed_probs_bytes,
        buffers.routed_probs.len_bytes()
    );
    assert!(
        buffers.output.len_bytes() >= output_bytes,
        "MoE combine without common output buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.output.len_bytes()
    );
}

fn validate_with_common_buffers(shape: MoECombineWithCommonShape, buffers: &MoECombineWithCommonBuffers<'_>) {
    let routed_output_bytes = shape.routed_output_bytes();
    let routed_probs_bytes = shape.routed_probs_bytes();
    let output_bytes = shape.output_bytes();
    let common_gate_logits_bytes = checked_product(
        "MoE combine common-gate byte length",
        &[shape.num_tokens as usize, shape.dtype.item_size()],
    );
    assert!(
        buffers.routed_hidden.len_bytes() >= routed_output_bytes,
        "MoE combine with common routed_hidden buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        routed_output_bytes,
        buffers.routed_hidden.len_bytes()
    );
    assert!(
        buffers.routed_probs.len_bytes() >= routed_probs_bytes,
        "MoE combine with common routed_probs buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        routed_probs_bytes,
        buffers.routed_probs.len_bytes()
    );
    assert!(
        buffers.common_hidden.len_bytes() >= output_bytes,
        "MoE combine with common common_hidden buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.common_hidden.len_bytes()
    );
    assert!(
        buffers.common_gate_logits.len_bytes() >= common_gate_logits_bytes,
        "MoE combine with common common_gate_logits buffer too short: shape={shape:?} required_bytes={} \
         buffer_bytes={}",
        common_gate_logits_bytes,
        buffers.common_gate_logits.len_bytes()
    );
    assert!(
        buffers.output.len_bytes() >= output_bytes,
        "MoE combine with common output buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.output.len_bytes()
    );
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use half::bf16;
    use inference_executor_core::mlp::moe::reference::moe_combine_with_common_bf16_reference;
    use inference_executor_core::mlp::moe::reference::moe_combine_without_common_bf16_reference;

    use super::*;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Stream;

    #[test]
    #[should_panic(expected = "MoE combine output elements exceeds the shader u32 count domain")]
    fn test_without_common_shape_rejects_shader_count_overflow() {
        MoECombineWithoutCommonShape::bf16(1 << 30, 1, 4).validate();
    }

    #[test]
    #[should_panic(expected = "MoE combine-with-common output elements exceeds the shader u32 count domain")]
    fn test_with_common_shape_rejects_shader_count_overflow() {
        MoECombineWithCommonShape::bf16(1 << 30, 1, 4).validate();
    }

    #[test]
    fn test_without_common_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = MoECombineWithoutCommonShape::bf16(2, 2, 3);
        let routed_hidden_values = [
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            -1.0, 0.5, 2.0, //
            0.25, -0.75, 1.5,
        ];
        let routed_probs_values = [0.25, 0.75, 0.5, 0.5];
        let routed_hidden = bf16_buffer(&device, &routed_hidden_values);
        let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
        let output = Buffer::new_zeroed(
            &device,
            shape.num_tokens as usize * shape.hidden_dim as usize * size_of::<u16>(),
        );
        let kernel = MoECombineKernel::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_without_common(
            shape,
            MoECombineWithoutCommonBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                output: &output,
            },
        ));
        let program = builder.build();
        let submitted = stream.submit_replay(&program);
        submitted.wait();

        let actual = output.read_typed::<u16>(0, 6);
        let expected = moe_combine_without_common_bf16_reference(&routed_hidden_values, &routed_probs_values, 2, 2, 3);
        assert_close_bits(&actual, &expected, 1.0e-3);
    }

    #[test]
    fn test_with_common_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = MoECombineWithCommonShape::bf16(2, 2, 3);
        let routed_hidden_values = [
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            -1.0, 0.5, 2.0, //
            0.25, -0.75, 1.5,
        ];
        let routed_probs_values = [0.25, 0.75, 0.5, 0.5];
        let common_hidden_values = [0.5, 1.0, -2.0, 1.5, -0.5, 0.25];
        let common_gate_logits_values = [-1.0, 2.0];
        let routed_hidden = bf16_buffer(&device, &routed_hidden_values);
        let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
        let common_hidden = bf16_buffer(&device, &common_hidden_values);
        let common_gate_logits = bf16_buffer(&device, &common_gate_logits_values);
        let output = Buffer::new_zeroed(
            &device,
            shape.num_tokens as usize * shape.hidden_dim as usize * size_of::<u16>(),
        );
        let kernel = MoECombineKernel::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_with_common(
            shape,
            MoECombineWithCommonBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                common_hidden: &common_hidden,
                common_gate_logits: &common_gate_logits,
                output: &output,
            },
        ));
        let program = builder.build();
        let submitted = stream.submit_replay(&program);
        submitted.wait();

        let actual = output.read_typed::<u16>(0, 6);
        let routed = moe_combine_without_common_bf16_reference(&routed_hidden_values, &routed_probs_values, 2, 2, 3);
        let expected =
            moe_combine_with_common_bf16_reference(&routed, &common_hidden_values, &common_gate_logits_values, 2, 3);
        assert_close_bits(&actual, &expected, 1.0e-3);
    }

    #[test]
    fn test_with_common_random() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = MoECombineWithCommonShape::bf16(3, 3, 5);
        let random_seed = 0xC461_8E2B;
        let routed_hidden_values = generated_values(
            shape.num_tokens as usize * shape.num_experts_per_token as usize * shape.hidden_dim as usize,
            random_seed,
        );
        let routed_probs_values = generated_probs(
            shape.num_tokens as usize,
            shape.num_experts_per_token as usize,
            random_seed.wrapping_add(1),
        );
        let common_hidden_values = generated_values(
            shape.num_tokens as usize * shape.hidden_dim as usize,
            random_seed.wrapping_add(2),
        );
        let common_gate_logits_values = generated_values(shape.num_tokens as usize, random_seed.wrapping_add(3));
        let routed_hidden = bf16_buffer(&device, &routed_hidden_values);
        let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
        let common_hidden = bf16_buffer(&device, &common_hidden_values);
        let common_gate_logits = bf16_buffer(&device, &common_gate_logits_values);
        let output = Buffer::new_zeroed(
            &device,
            shape.num_tokens as usize * shape.hidden_dim as usize * size_of::<u16>(),
        );
        let kernel = MoECombineKernel::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_with_common(
            shape,
            MoECombineWithCommonBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                common_hidden: &common_hidden,
                common_gate_logits: &common_gate_logits,
                output: &output,
            },
        ));
        let program = builder.build();
        stream.submit_replay(&program).wait();

        let routed = moe_combine_without_common_bf16_reference(
            &routed_hidden_values,
            &routed_probs_values,
            shape.num_tokens as usize,
            shape.num_experts_per_token as usize,
            shape.hidden_dim as usize,
        );
        let expected = moe_combine_with_common_bf16_reference(
            &routed,
            &common_hidden_values,
            &common_gate_logits_values,
            shape.num_tokens as usize,
            shape.hidden_dim as usize,
        );
        let actual = output.read_typed::<u16>(0, shape.num_tokens as usize * shape.hidden_dim as usize);
        assert_close_bits(&actual, &expected, 1.0e-3);
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        Buffer::from_slice(device, &bits)
    }

    fn generated_values(count: usize, random_seed: u32) -> Vec<f32> {
        let mut state = random_seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    fn generated_probs(num_tokens: usize, num_experts_per_token: usize, random_seed: u32) -> Vec<f32> {
        let mut values = generated_values(num_tokens * num_experts_per_token, random_seed)
            .into_iter()
            .map(|value| value.abs() + 0.05)
            .collect::<Vec<_>>();
        for row in values.chunks_mut(num_experts_per_token) {
            let sum = row.iter().sum::<f32>();
            for value in row {
                *value /= sum;
            }
        }
        values
    }

    fn assert_close_bits(actual: &[u16], expected: &[u16], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let actual = bf16::from_bits(actual).to_f32();
            let expected = bf16::from_bits(expected).to_f32();
            assert!(
                (actual - expected).abs() <= tolerance,
                "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
