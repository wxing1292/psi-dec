use std::mem::size_of;

use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/gdn_qkvabz_split.metal");

/// Projection-split tensor contract:
///
/// ```text
/// qkvabz: [T, Cqkv + 2 * Hv + Hv * Dv] (F32)
/// qkv:    [T, Cqkv]                      (F32)
/// a:      [T, Hv]                        (F32)
/// b:      [T, Hv]                        (F32)
/// z:      [T, Hv, Dv]                    (F32)
/// ```
///
/// `T` is the flattened token axis; `Hv` and `Dv` are the value-head and
/// within-value-head axes. The caller supplies
/// `qkv_dim = Cqkv = 2 * Hqk * Dqk + Hv * Dv` and `v_dim = Hv * Dv`.
/// `C` names only this concatenated channel axis, not a head axis or a
/// convolution-kernel extent.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub qkv_dim: u32,
    pub num_v_heads: u32,
    pub v_dim: u32,
}

impl Config {
    pub fn new(qkv_dim: u32, num_v_heads: u32, v_dim: u32) -> Self {
        Self {
            qkv_dim,
            num_v_heads,
            v_dim,
        }
    }

    pub fn validate(self) {
        assert!(self.qkv_dim > 0);
        assert!(self.num_v_heads > 0);
        assert!(self.v_dim > 0);
        self.qkvabz_row_stride();
    }

    pub fn num_qkvabz_values(self, shape: Shape) -> usize {
        checked_product(
            "GDN projection element count",
            &[shape.num_total_tokens as usize, self.qkvabz_row_stride() as usize],
        )
    }

    pub fn num_qkv_values(self, shape: Shape) -> usize {
        checked_product(
            "GDN QKV element count",
            &[shape.num_total_tokens as usize, self.qkv_dim as usize],
        )
    }

    pub fn num_gate_values(self, shape: Shape) -> usize {
        checked_product(
            "GDN gate element count",
            &[shape.num_total_tokens as usize, self.num_v_heads as usize],
        )
    }

    pub fn num_z_values(self, shape: Shape) -> usize {
        checked_product(
            "GDN Z element count",
            &[shape.num_total_tokens as usize, self.v_dim as usize],
        )
    }

    fn qkvabz_row_stride(self) -> u32 {
        self.num_v_heads
            .checked_mul(2)
            .and_then(|gate_dim| gate_dim.checked_add(self.qkv_dim))
            .and_then(|stride| stride.checked_add(self.v_dim))
            .expect("GDN projection stride must fit u32")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub num_total_tokens: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_tokens > 0);
        assert_u32_count_domain(config.num_qkvabz_values(self), "GDN projection elements");
    }
}

pub struct Buffers<'a> {
    pub qkvabz: &'a Buffer,
    pub qkv: &'a Buffer,
    pub a: &'a Buffer,
    pub b: &'a Buffer,
    pub z: &'a Buffer,
}

pub struct Compute {
    config: Config,
    kernel: Kernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            kernel: Kernel::new(device, SOURCE, "gdn_qkvabz_split_f32"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>) -> Invocation<'a> {
        Invocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_tokens: ReplayU32::Fixed(shape.num_total_tokens),
        }
    }

    pub fn invoke_bucketed<'a>(
        &'a self,
        shape: Shape,
        buffers: Buffers<'a>,
        num_active_tokens: ReplayU32,
    ) -> Invocation<'a> {
        Invocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_tokens,
        }
    }
}

pub struct Invocation<'a> {
    config: Config,
    kernel: &'a Kernel,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_tokens: ReplayU32,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.shape.validate(self.config);
        validate_qkvabz_split_buffers(self.config, self.shape, &self.buffers);
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.qkvabz, 0);
        recorder.set_buffer_write(1, self.buffers.qkv, 0);
        recorder.set_buffer_write(2, self.buffers.a, 0);
        recorder.set_buffer_write(3, self.buffers.b, 0);
        recorder.set_buffer_write(4, self.buffers.z, 0);
        set_replay_u32(
            recorder,
            5,
            self.num_active_tokens,
            self.shape.num_total_tokens,
            "GDN projection-split active token count",
        );
        recorder.set_u32(6, self.config.qkv_dim);
        recorder.set_u32(7, self.config.num_v_heads);
        recorder.set_u32(8, self.config.v_dim);
        recorder.dispatch_1d(self.config.num_qkvabz_values(self.shape), 256);
    }
}

fn set_replay_u32(recorder: &CommandRecorder<'_>, index: usize, value: ReplayU32, max_value: u32, name: &str) {
    match value {
        ReplayU32::Fixed(value) => {
            assert!(value > 0, "{name} must be positive");
            assert!(value <= max_value, "{name} exceeds recorded capacity");
            recorder.set_u32(index, value);
        },
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, max_value),
    }
}

fn validate_qkvabz_split_buffers(config: Config, shape: Shape, buffers: &Buffers<'_>) {
    assert!(
        buffers.qkvabz.len_bytes()
            >= checked_product("GDN qkvabz input", &[config.num_qkvabz_values(shape), size_of::<f32>()])
    );
    assert!(
        buffers.qkv.len_bytes()
            >= checked_product("GDN Q/K/V output", &[config.num_qkv_values(shape), size_of::<f32>()])
    );
    assert!(
        buffers.a.len_bytes() >= checked_product("GDN a output", &[config.num_gate_values(shape), size_of::<f32>()])
    );
    assert!(
        buffers.b.len_bytes() >= checked_product("GDN b output", &[config.num_gate_values(shape), size_of::<f32>()])
    );
    assert!(buffers.z.len_bytes() >= checked_product("GDN z output", &[config.num_z_values(shape), size_of::<f32>()]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Dtype;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::ReplayU32;
    use crate::metal::Stream;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gdn_split.num_active_tokens");

    #[test]
    fn test_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config::new(6, 2, 4);
        let shape = Shape { num_total_tokens: 2 };
        let qkvabz_values = (0..28).map(|value| value as f32).collect::<Vec<_>>();
        let qkvabz = Buffer::from_slice(&device, &qkvabz_values);
        let qkv = Buffer::new_zeroed_elements(&device, config.num_qkv_values(shape), Dtype::Float32);
        let a = Buffer::new_zeroed_elements(&device, config.num_gate_values(shape), Dtype::Float32);
        let b = Buffer::new_zeroed_elements(&device, config.num_gate_values(shape), Dtype::Float32);
        let z = Buffer::new_zeroed_elements(&device, config.num_z_values(shape), Dtype::Float32);
        let kernel = Compute::new(&device, config);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            Buffers {
                qkvabz: &qkvabz,
                qkv: &qkv,
                a: &a,
                b: &b,
                z: &z,
            },
        ));
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(
            qkv.read_typed::<f32>(0, config.num_qkv_values(shape)),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        );
        assert_eq!(
            a.read_typed::<f32>(0, config.num_gate_values(shape)),
            vec![6.0, 7.0, 20.0, 21.0]
        );
        assert_eq!(
            b.read_typed::<f32>(0, config.num_gate_values(shape)),
            vec![8.0, 9.0, 22.0, 23.0]
        );
        assert_eq!(
            z.read_typed::<f32>(0, config.num_z_values(shape)),
            vec![10.0, 11.0, 12.0, 13.0, 24.0, 25.0, 26.0, 27.0]
        );
    }

    #[test]
    fn test_bucketed_replay_preserves_inactive_output_tail() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config::new(2, 1, 2);
        let shape = Shape { num_total_tokens: 2 };
        let qkvabz = Buffer::new_zeroed_elements(&device, config.num_qkvabz_values(shape), Dtype::Float32);
        let qkv = Buffer::new_zeroed_elements(&device, config.num_qkv_values(shape), Dtype::Float32);
        let a = Buffer::new_zeroed_elements(&device, config.num_gate_values(shape), Dtype::Float32);
        let b = Buffer::new_zeroed_elements(&device, config.num_gate_values(shape), Dtype::Float32);
        let z = Buffer::new_zeroed_elements(&device, config.num_z_values(shape), Dtype::Float32);
        let kernel = Compute::new(&device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_bucketed(
            shape,
            Buffers {
                qkvabz: &qkvabz,
                qkv: &qkv,
                a: &a,
                b: &b,
                z: &z,
            },
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
        ));
        let replay = builder.build();
        let valid_rows = [[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]];
        for num_active_tokens in [1_usize, 2, 1] {
            let mut input = valid_rows.into_iter().flatten().collect::<Vec<_>>();
            if num_active_tokens == 1 {
                input[6..].fill(f32::NAN);
            }
            qkvabz.write_typed(0, &input);
            qkv.write_typed(0, &[-777.0_f32; 4]);
            a.write_typed(0, &[-777.0_f32; 2]);
            b.write_typed(0, &[-777.0_f32; 2]);
            z.write_typed(0, &[-777.0_f32; 4]);
            stream
                .submit_replay_with_arguments(
                    &replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32),
                )
                .wait();

            let active_rows = &valid_rows[..num_active_tokens];
            assert_eq!(
                qkv.read_typed::<f32>(0, num_active_tokens * 2),
                flatten_columns(active_rows, 0, 2)
            );
            assert_eq!(
                a.read_typed::<f32>(0, num_active_tokens),
                flatten_columns(active_rows, 2, 3)
            );
            assert_eq!(
                b.read_typed::<f32>(0, num_active_tokens),
                flatten_columns(active_rows, 3, 4)
            );
            assert_eq!(
                z.read_typed::<f32>(0, num_active_tokens * 2),
                flatten_columns(active_rows, 4, 6)
            );
            assert_eq!(
                qkv.read_typed::<f32>(num_active_tokens * 2, 4 - num_active_tokens * 2),
                vec![-777.0; 4 - num_active_tokens * 2]
            );
            assert_eq!(
                a.read_typed::<f32>(num_active_tokens, 2 - num_active_tokens),
                vec![-777.0; 2 - num_active_tokens]
            );
            assert_eq!(
                b.read_typed::<f32>(num_active_tokens, 2 - num_active_tokens),
                vec![-777.0; 2 - num_active_tokens]
            );
            assert_eq!(
                z.read_typed::<f32>(num_active_tokens * 2, 4 - num_active_tokens * 2),
                vec![-777.0; 4 - num_active_tokens * 2]
            );
        }
    }

    fn flatten_columns(rows: &[[f32; 6]], start: usize, end: usize) -> Vec<f32> {
        rows.iter().flat_map(|row| row[start..end].iter().copied()).collect()
    }

    #[test]
    #[should_panic(expected = "GDN projection elements exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        Shape {
            num_total_tokens: 1 << 30,
        }
        .validate(Config::new(1, 1, 1));
    }
}
