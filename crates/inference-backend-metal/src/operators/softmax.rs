use std::collections::HashSet;
use std::path::PathBuf;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::operators::mlx_headers::find_mlx_metal_header_root;
use crate::operators::mlx_headers::read_mlx_metal_header;

#[derive(Clone, Copy, Debug)]
pub struct SoftmaxShape {
    pub num_rows: u32,
    pub num_values_per_row: u32,
    pub dtype: Dtype,
}

impl SoftmaxShape {
    pub fn validate(self) {
        assert!(self.num_rows > 0);
        assert!(self.num_values_per_row > 0);
        assert!(self.num_values_per_row <= 4096);
        assert_eq!(self.dtype, Dtype::Bfloat16);
    }

    pub fn bytes(self) -> usize {
        self.validate();
        self.num_rows as usize * self.num_values_per_row as usize * self.dtype.item_size()
    }
}

pub struct SoftmaxKernel {
    shape: SoftmaxShape,
    kernel: Kernel,
}

impl SoftmaxKernel {
    pub fn new(device: &Device, shape: SoftmaxShape) -> Self {
        shape.validate();
        let kernel = Kernel::new(device, &softmax_source(), "block_softmax_bfloat16");
        Self { shape, kernel }
    }

    pub fn invoke<'a>(&'a self, output: &'a Buffer, input: &'a Buffer) -> SoftmaxInvocation<'a> {
        SoftmaxInvocation {
            kernel: self,
            shape: self.shape,
            output,
            input,
        }
    }

    pub fn invoke_with_shape<'a>(
        &'a self,
        shape: SoftmaxShape,
        output: &'a Buffer,
        input: &'a Buffer,
    ) -> SoftmaxInvocation<'a> {
        shape.validate();
        assert_eq!(shape.num_values_per_row, self.shape.num_values_per_row);
        assert_eq!(shape.dtype, self.shape.dtype);
        SoftmaxInvocation {
            kernel: self,
            shape,
            output,
            input,
        }
    }
}

pub struct SoftmaxInvocation<'a> {
    kernel: &'a SoftmaxKernel,
    shape: SoftmaxShape,
    output: &'a Buffer,
    input: &'a Buffer,
}

impl Operator for SoftmaxInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.record_compute(builder);
    }
}

impl SoftmaxInvocation<'_> {
    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.shape;
        shape.validate();
        let bytes = shape.bytes();
        assert!(bytes <= self.input.len_bytes());
        assert!(bytes <= self.output.len_bytes());

        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.input, 0);
        builder.set_buffer_write(1, self.output, 0);
        builder.set_i32(2, shape.num_values_per_row as i32);

        let n_reads = 4usize;
        let simd_size = 32usize;
        let num_threads_needed = (shape.num_values_per_row as usize).div_ceil(n_reads);
        let num_simdgroups = num_threads_needed.div_ceil(simd_size);
        let num_threads_per_threadblock = simd_size * num_simdgroups;
        builder.dispatch_1d(
            shape.num_rows as usize * num_threads_per_threadblock,
            num_threads_per_threadblock,
        );
    }
}

fn softmax_source() -> String {
    let root = mlx_metal_header_root();
    let mut included = HashSet::new();
    let mut source = String::new();
    source.push_str(
        "#include <metal_stdlib>\n#include <metal_common>\n#include <metal_simdgroup>\nusing namespace metal;\n",
    );
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/defines.h",
        &mut included,
    ));
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/utils.h",
        &mut included,
    ));
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/softmax.h",
        &mut included,
    ));
    source.push_str(
        "\ntemplate [[host_name(\"block_softmax_bfloat16\")]] [[kernel]] decltype(softmax_single_row<bfloat16_t, \
         bfloat16_t>) softmax_single_row<bfloat16_t, bfloat16_t>;\n",
    );
    source
}

fn mlx_metal_header_root() -> PathBuf {
    find_mlx_metal_header_root("softmax.h", |_| true, "softmax")
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::SoftmaxKernel;
    use super::SoftmaxShape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::Stream;

    #[test]
    fn test_reference() {
        let shape = SoftmaxShape {
            num_rows: 2,
            num_values_per_row: 4,
            dtype: Dtype::Bfloat16,
        };
        let (device, kernel) = create_softmax_kernel(shape);
        let stream = Stream::new(&device);
        let input_values = [-2.0, -1.0, 0.0, 1.0, 4.0, 2.0, 0.0, -2.0];
        let input = bf16_buffer(&device, &input_values);
        let output = Buffer::new_zeroed(&device, shape.bytes());

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(&output, &input));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let actual = read_bf16_values(&output, input_values.len());
        let expected = cpu_softmax_bf16_rows(
            &input_values,
            shape.num_rows as usize,
            shape.num_values_per_row as usize,
        );
        assert_close(&actual, &expected, 0.01);
    }

    fn create_softmax_kernel(shape: SoftmaxShape) -> (Device, SoftmaxKernel) {
        let device = Device::system_default();
        let kernel = SoftmaxKernel::new(&device, shape);
        (device, kernel)
    }

    fn cpu_softmax_bf16_rows(values: &[f32], num_rows: usize, num_values_per_row: usize) -> Vec<f32> {
        assert_eq!(values.len(), num_rows * num_values_per_row);
        let mut output = Vec::with_capacity(values.len());
        for row in values.chunks_exact(num_values_per_row) {
            let row: Vec<f32> = row.iter().map(|value| bf16::from_f32(*value).to_f32()).collect();
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|value| (*value - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            output.extend(exps.into_iter().map(|value| bf16::from_f32(value / sum).to_f32()));
        }
        output
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        Buffer::from_slice(device, &bits)
    }

    fn read_bf16_values(buffer: &Buffer, len: usize) -> Vec<f32> {
        buffer
            .read_typed::<u16>(0, len)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "value mismatch at {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
