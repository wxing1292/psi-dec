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

const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct MLXElementwiseShape {
    pub num_values: u32,
    pub dtype: Dtype,
}

impl MLXElementwiseShape {
    pub fn validate(self) {
        assert!(self.num_values > 0);
        assert_eq!(self.dtype, Dtype::Bfloat16);
    }

    pub fn bytes(self) -> usize {
        self.validate();
        self.num_values as usize * self.dtype.item_size()
    }
}

pub struct MLXSigmoidKernel {
    shape: MLXElementwiseShape,
    kernel: Kernel,
}

impl MLXSigmoidKernel {
    pub fn new(device: &Device, shape: MLXElementwiseShape) -> Self {
        shape.validate();
        let kernel = Kernel::new(device, &mlx_sigmoid_source(), "mlx_sigmoid_bf16");
        Self { shape, kernel }
    }

    pub fn invoke<'a>(&'a self, output: &'a Buffer, input: &'a Buffer) -> MLXSigmoidInvocation<'a> {
        MLXSigmoidInvocation {
            kernel: self,
            output,
            input,
        }
    }
}

pub struct MLXSigmoidInvocation<'a> {
    kernel: &'a MLXSigmoidKernel,
    output: &'a Buffer,
    input: &'a Buffer,
}

impl Operator for MLXSigmoidInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.record_compute(builder);
    }
}

impl MLXSigmoidInvocation<'_> {
    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.kernel.shape;
        shape.validate();
        assert!(self.input.len_bytes() >= shape.bytes());
        assert!(self.output.len_bytes() >= shape.bytes());

        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.input, 0);
        builder.set_buffer_write(1, self.output, 0);
        builder.set_u32(2, shape.num_values);
        builder.dispatch_1d(shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}

pub struct MLXMultiplyKernel {
    shape: MLXElementwiseShape,
    kernel: Kernel,
}

impl MLXMultiplyKernel {
    pub fn new(device: &Device, shape: MLXElementwiseShape) -> Self {
        shape.validate();
        let kernel = Kernel::new(device, &mlx_multiply_source(), "mlx_multiply_bf16");
        Self { shape, kernel }
    }

    pub fn invoke<'a>(&'a self, output: &'a Buffer, lhs: &'a Buffer, rhs: &'a Buffer) -> MLXMultiplyInvocation<'a> {
        MLXMultiplyInvocation {
            kernel: self,
            output,
            lhs,
            rhs,
        }
    }
}

pub struct MLXMultiplyInvocation<'a> {
    kernel: &'a MLXMultiplyKernel,
    output: &'a Buffer,
    lhs: &'a Buffer,
    rhs: &'a Buffer,
}

impl Operator for MLXMultiplyInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.record_compute(builder);
    }
}

impl MLXMultiplyInvocation<'_> {
    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.kernel.shape;
        shape.validate();
        assert!(self.lhs.len_bytes() >= shape.bytes());
        assert!(self.rhs.len_bytes() >= shape.bytes());
        assert!(self.output.len_bytes() >= shape.bytes());

        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.lhs, 0);
        builder.set_buffer_read(1, self.rhs, 0);
        builder.set_buffer_write(2, self.output, 0);
        builder.set_u32(3, shape.num_values);
        builder.dispatch_1d(shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}

fn mlx_sigmoid_source() -> String {
    let root = mlx_metal_header_root("unary_ops.h");
    let mut included = HashSet::new();
    let mut source = String::new();
    source.push_str("#include <metal_stdlib>\n#include <metal_common>\nusing namespace metal;\n");
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
        "mlx/backend/metal/kernels/unary_ops.h",
        &mut included,
    ));
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/unary.h",
        &mut included,
    ));
    source.push_str(
        "\ntemplate [[host_name(\"mlx_sigmoid_bf16\")]] [[kernel]] decltype(unary_v<bfloat16_t, bfloat16_t, Sigmoid>) \
         unary_v<bfloat16_t, bfloat16_t, Sigmoid>;\n",
    );
    source
}

fn mlx_multiply_source() -> String {
    let root = mlx_metal_header_root("binary_ops.h");
    let mut included = HashSet::new();
    let mut source = String::new();
    source.push_str(
        "#include <metal_stdlib>\n#include <metal_common>\n#include <metal_integer>\n#include <metal_math>\nusing \
         namespace metal;\n",
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
        "mlx/backend/metal/kernels/binary_ops.h",
        &mut included,
    ));
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/binary.h",
        &mut included,
    ));
    source.push_str(
        "\ntemplate [[host_name(\"mlx_multiply_bf16\")]] [[kernel]] decltype(binary_vv<bfloat16_t, bfloat16_t, \
         Multiply>) binary_vv<bfloat16_t, bfloat16_t, Multiply>;\n",
    );
    source
}

fn mlx_metal_header_root(required_header: &str) -> PathBuf {
    find_mlx_metal_header_root(required_header, |_| true, "elementwise wrapper")
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::MLXElementwiseShape;
    use super::MLXMultiplyKernel;
    use super::MLXSigmoidKernel;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::Stream;

    #[test]
    fn test_sigmoid() {
        let (device, kernel) = create_sigmoid_kernel(6);
        let stream = Stream::new(&device);
        let input = bf16_buffer(&device, &[-4.0, -1.0, 0.0, 0.5, 2.0, 8.0]);
        let output = Buffer::new_zeroed(&device, 6 * Dtype::Bfloat16.item_size());

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(&output, &input));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let actual = read_bf16_values(&output, 6);
        let expected: Vec<f32> = [-4.0_f32, -1.0, 0.0, 0.5, 2.0, 8.0]
            .into_iter()
            .map(|value| bf16::from_f32(1.0 / (1.0 + (-value).exp())).to_f32())
            .collect();
        assert_close(&actual, &expected, 0.004);
    }

    #[test]
    fn test_multiply() {
        let (device, kernel) = create_multiply_kernel(5);
        let stream = Stream::new(&device);
        let lhs_values = [-3.0, -1.25, 0.5, 2.0, 7.5];
        let rhs_values = [0.25, -2.0, 3.0, -4.0, 0.125];
        let lhs = bf16_buffer(&device, &lhs_values);
        let rhs = bf16_buffer(&device, &rhs_values);
        let output = Buffer::new_zeroed(&device, 5 * Dtype::Bfloat16.item_size());

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(&output, &lhs, &rhs));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let actual = read_bf16_values(&output, 5);
        let expected: Vec<f32> = lhs_values
            .into_iter()
            .zip(rhs_values)
            .map(|(lhs, rhs)| bf16::from_f32(bf16::from_f32(lhs).to_f32() * bf16::from_f32(rhs).to_f32()).to_f32())
            .collect();
        assert_close(&actual, &expected, 0.004);
    }

    fn create_sigmoid_kernel(num_values: u32) -> (Device, MLXSigmoidKernel) {
        let device = Device::system_default();
        let kernel = MLXSigmoidKernel::new(
            &device,
            MLXElementwiseShape {
                num_values,
                dtype: Dtype::Bfloat16,
            },
        );
        (device, kernel)
    }

    fn create_multiply_kernel(num_values: u32) -> (Device, MLXMultiplyKernel) {
        let device = Device::system_default();
        let kernel = MLXMultiplyKernel::new(
            &device,
            MLXElementwiseShape {
                num_values,
                dtype: Dtype::Bfloat16,
            },
        );
        (device, kernel)
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
