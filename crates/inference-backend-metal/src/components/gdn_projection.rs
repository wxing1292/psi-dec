use std::mem::size_of;

use super::assert_u32_count_domain;
use super::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const GDN_PROJECTION_SPLIT_SOURCE: &str = include_str!("metal/gdn_projection_split.metal");

/// Projection-split tensor contract:
///
/// ```text
/// qkvabz:        [T, Cqkv + 2 * Hv + Hv * Dv]
/// projected_qkv: [T, Cqkv]
/// a:             [T, Hv]
/// b:             [T, Hv]
/// z:             [T, Hv, Dv]
/// ```
///
/// `T` is the flattened token axis; `Hv` and `Dv` are the value-head and
/// within-value-head axes. The caller supplies
/// `qkv_dim = Cqkv = 2 * Hqk * Dqk + Hv * Dv` and `v_dim = Hv * Dv`.
/// `C` names only this concatenated channel axis, not a head axis or a
/// convolution-kernel extent.
#[derive(Clone, Copy, Debug)]
pub struct GDNProjectionSplitShape {
    pub num_tokens: u32,
    pub qkv_dim: u32,
    pub num_v_heads: u32,
    pub v_dim: u32,
    pub input_dtype: Dtype,
}

impl GDNProjectionSplitShape {
    pub fn f32(num_tokens: u32, qkv_dim: u32, num_v_heads: u32, v_dim: u32) -> Self {
        Self {
            num_tokens,
            qkv_dim,
            num_v_heads,
            v_dim,
            input_dtype: Dtype::Float32,
        }
    }

    pub fn bf16_to_f32(num_tokens: u32, qkv_dim: u32, num_v_heads: u32, v_dim: u32) -> Self {
        Self {
            num_tokens,
            qkv_dim,
            num_v_heads,
            v_dim,
            input_dtype: Dtype::Bfloat16,
        }
    }

    pub fn num_qkvabz_values(self) -> usize {
        checked_product(
            "GDN projection element count",
            &[self.num_tokens as usize, self.qkvabz_row_stride() as usize],
        )
    }

    pub fn num_projected_qkv_values(self) -> usize {
        checked_product(
            "GDN projected-QKV element count",
            &[self.num_tokens as usize, self.qkv_dim as usize],
        )
    }

    pub fn num_gate_values(self) -> usize {
        checked_product(
            "GDN gate element count",
            &[self.num_tokens as usize, self.num_v_heads as usize],
        )
    }

    pub fn num_z_values(self) -> usize {
        checked_product("GDN Z element count", &[self.num_tokens as usize, self.v_dim as usize])
    }

    pub fn validate(self) {
        assert!(self.num_tokens > 0);
        assert!(self.qkv_dim > 0);
        assert!(self.num_v_heads > 0);
        assert!(self.v_dim > 0);
        assert!(matches!(self.input_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert_u32_count_domain(self.num_qkvabz_values(), "GDN projection elements");
    }

    fn qkvabz_row_stride(self) -> u32 {
        self.num_v_heads
            .checked_mul(2)
            .and_then(|gate_dim| gate_dim.checked_add(self.qkv_dim))
            .and_then(|stride| stride.checked_add(self.v_dim))
            .expect("GDN projection stride must fit u32")
    }
}

pub struct GDNProjectionSplitBuffers<'a> {
    pub qkvabz: &'a Buffer,
    pub projected_qkv: &'a Buffer,
    pub a: &'a Buffer,
    pub b: &'a Buffer,
    pub z: &'a Buffer,
}

pub struct GDNProjectionSplitKernel {
    f32_kernel: Kernel,
    bf16_to_f32_kernel: Kernel,
}

impl GDNProjectionSplitKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            f32_kernel: Kernel::new(device, GDN_PROJECTION_SPLIT_SOURCE, "gdn_projection_split_f32"),
            bf16_to_f32_kernel: Kernel::new(device, GDN_PROJECTION_SPLIT_SOURCE, "gdn_projection_split_bf16_to_f32"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GDNProjectionSplitShape,
        buffers: GDNProjectionSplitBuffers<'a>,
    ) -> GDNProjectionSplitInvocation<'a> {
        GDNProjectionSplitInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }

    fn record_compute(
        &self,
        builder: &CommandRecorder,
        shape: GDNProjectionSplitShape,
        buffers: &GDNProjectionSplitBuffers<'_>,
    ) {
        let kernel = match shape.input_dtype {
            Dtype::Float32 => &self.f32_kernel,
            Dtype::Bfloat16 => &self.bf16_to_f32_kernel,
            _ => panic!("GDN projection split input dtype must be f32 or bf16"),
        };
        builder.set_kernel(kernel);
        builder.set_buffer_read(0, buffers.qkvabz, 0);
        builder.set_buffer_write(1, buffers.projected_qkv, 0);
        builder.set_buffer_write(2, buffers.a, 0);
        builder.set_buffer_write(3, buffers.b, 0);
        builder.set_buffer_write(4, buffers.z, 0);
        builder.set_u32(5, shape.num_tokens);
        builder.set_u32(6, shape.qkv_dim);
        builder.set_u32(7, shape.num_v_heads);
        builder.set_u32(8, shape.v_dim);
        builder.dispatch_1d(shape.num_qkvabz_values(), 256);
    }
}

pub struct GDNProjectionSplitInvocation<'a> {
    kernel: &'a GDNProjectionSplitKernel,
    shape: GDNProjectionSplitShape,
    buffers: GDNProjectionSplitBuffers<'a>,
}

impl Operator for GDNProjectionSplitInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        validate_projection_split_buffers(self.shape, &self.buffers);
        self.kernel.record_compute(builder, self.shape, &self.buffers);
    }
}

fn validate_projection_split_buffers(shape: GDNProjectionSplitShape, buffers: &GDNProjectionSplitBuffers<'_>) {
    assert!(buffers.qkvabz.len_bytes() >= shape.num_qkvabz_values() * shape.input_dtype.item_size());
    assert!(buffers.projected_qkv.len_bytes() >= shape.num_projected_qkv_values() * size_of::<f32>());
    assert!(buffers.a.len_bytes() >= shape.num_gate_values() * size_of::<f32>());
    assert!(buffers.b.len_bytes() >= shape.num_gate_values() * size_of::<f32>());
    assert!(buffers.z.len_bytes() >= shape.num_z_values() * size_of::<f32>());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Stream;

    #[test]
    fn test_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = GDNProjectionSplitShape::f32(2, 6, 2, 4);
        let qkvabz_values = (0..28).map(|value| value as f32).collect::<Vec<_>>();
        let qkvabz = Buffer::from_slice(&device, &qkvabz_values);
        let projected_qkv = Buffer::new_zeroed_elements(&device, shape.num_projected_qkv_values(), Dtype::Float32);
        let a = Buffer::new_zeroed_elements(&device, shape.num_gate_values(), Dtype::Float32);
        let b = Buffer::new_zeroed_elements(&device, shape.num_gate_values(), Dtype::Float32);
        let z = Buffer::new_zeroed_elements(&device, shape.num_z_values(), Dtype::Float32);
        let kernel = GDNProjectionSplitKernel::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            GDNProjectionSplitBuffers {
                qkvabz: &qkvabz,
                projected_qkv: &projected_qkv,
                a: &a,
                b: &b,
                z: &z,
            },
        ));
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(
            projected_qkv.read_typed::<f32>(0, shape.num_projected_qkv_values()),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        );
        assert_eq!(
            a.read_typed::<f32>(0, shape.num_gate_values()),
            vec![6.0, 7.0, 20.0, 21.0]
        );
        assert_eq!(
            b.read_typed::<f32>(0, shape.num_gate_values()),
            vec![8.0, 9.0, 22.0, 23.0]
        );
        assert_eq!(
            z.read_typed::<f32>(0, shape.num_z_values()),
            vec![10.0, 11.0, 12.0, 13.0, 24.0, 25.0, 26.0, 27.0]
        );
    }

    #[test]
    #[should_panic(expected = "GDN projection elements exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        GDNProjectionSplitShape::f32(1 << 30, 1, 1, 1).validate();
    }
}
