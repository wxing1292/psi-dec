use std::collections::HashSet;
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::operators::mlx_headers::find_mlx_metal_header_root;
use crate::operators::mlx_headers::read_mlx_metal_header;

fn checked_product(name: &str, factors: &[usize]) -> usize {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit usize"))
}

fn checked_bytes(name: &str, dimensions: &[usize], dtype: Dtype) -> usize {
    checked_product(name, dimensions)
        .checked_mul(dtype.item_size())
        .unwrap_or_else(|| panic!("{name} byte length must fit usize"))
}

#[derive(Clone, Copy, Debug)]
pub struct AffineQuantizedMatmulShape {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
    pub affine_dtype: Dtype,
}

impl AffineQuantizedMatmulShape {
    pub fn same_dtype(m: i32, n: i32, k: i32, group_size: i32, bits: i32, dtype: Dtype) -> Self {
        Self {
            m,
            n,
            k,
            group_size,
            bits,
            input_dtype: dtype,
            output_dtype: dtype,
            affine_dtype: dtype,
        }
    }

    pub fn validate(self) {
        assert!(self.m > 0);
        assert!(self.n > 0);
        assert!(self.k > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.k % self.group_size, 0);
        assert!(matches!(
            self.input_dtype,
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ));
        assert!(matches!(
            self.output_dtype,
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ));
        assert!(matches!(
            self.affine_dtype,
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ));
    }

    pub fn output_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "affine matmul output",
            &[self.m as usize, self.n as usize],
            self.output_dtype,
        )
    }

    pub fn input_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "affine matmul input",
            &[self.m as usize, self.k as usize],
            self.input_dtype,
        )
    }

    pub fn weight_bytes(self) -> usize {
        self.validate();
        let pack_factor = if self.bits == 3 {
            8
        } else if self.bits == 6 {
            4
        } else {
            8 / self.bits
        };
        let bytes_per_pack = if self.bits == 3 || self.bits == 6 { 3 } else { 1 };
        checked_product(
            "affine matmul packed weight byte length",
            &[self.n as usize, self.k as usize, bytes_per_pack as usize],
        ) / pack_factor as usize
    }

    pub fn affine_param_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "affine matmul parameter",
            &[self.n as usize, (self.k / self.group_size) as usize],
            self.affine_dtype,
        )
    }

    fn uses_same_dtype(self) -> bool {
        self.input_dtype == self.output_dtype && self.input_dtype == self.affine_dtype
    }
}

pub struct AffineQuantizedMatmulKernel {
    shape: AffineQuantizedMatmulShape,
    kernel: Kernel,
    dispatch: AffineQuantizedDispatch,
}

#[derive(Clone, Copy, Debug)]
pub struct AffineQuantizedGateUpSiluShape {
    pub m: i32,
    pub intermediate_dim: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub dtype: Dtype,
}

impl AffineQuantizedGateUpSiluShape {
    pub fn validate(self) {
        assert!(self.m > 0);
        assert!(self.intermediate_dim > 0);
        assert!(self.k > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.k % self.group_size, 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16));
    }

    pub fn output_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "fused dense MLP output",
            &[self.m as usize, self.intermediate_dim as usize],
            self.dtype,
        )
    }

    pub fn input_bytes(self) -> usize {
        self.validate();
        checked_bytes("fused dense MLP input", &[self.m as usize, self.k as usize], self.dtype)
    }

    pub fn gate_up_weight_bytes(self) -> usize {
        AffineQuantizedMatmulShape {
            m: self.m,
            n: self
                .intermediate_dim
                .checked_mul(2)
                .expect("fused dense MLP gate/up dim must fit i32"),
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .weight_bytes()
    }

    pub fn gate_up_affine_param_bytes(self) -> usize {
        AffineQuantizedMatmulShape {
            m: self.m,
            n: self
                .intermediate_dim
                .checked_mul(2)
                .expect("fused dense MLP gate/up dim must fit i32"),
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .affine_param_bytes()
    }

    fn single_projection_shape(self) -> AffineQuantizedMatmulShape {
        self.validate();
        AffineQuantizedMatmulShape {
            m: self.m,
            n: self.intermediate_dim,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
    }
}

pub struct AffineQuantizedGateUpSiluKernel {
    shape: AffineQuantizedGateUpSiluShape,
    kernel: Kernel,
    dispatch: AffineQuantizedGateUpSiluDispatch,
}

#[derive(Clone, Copy, Debug)]
pub struct AffineQuantizedSplitGateUpSiluShape {
    pub m: i32,
    pub intermediate_dim: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub dtype: Dtype,
}

impl AffineQuantizedSplitGateUpSiluShape {
    pub fn validate(self) {
        AffineQuantizedGateUpSiluShape {
            m: self.m,
            intermediate_dim: self.intermediate_dim,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            dtype: self.dtype,
        }
        .validate();
    }

    pub fn output_bytes(self) -> usize {
        self.gate_up_shape().output_bytes()
    }

    pub fn input_bytes(self) -> usize {
        self.gate_up_shape().input_bytes()
    }

    pub fn gate_up_weight_bytes(self) -> usize {
        self.gate_up_shape().gate_up_weight_bytes()
    }

    pub fn gate_up_affine_param_bytes(self) -> usize {
        self.gate_up_shape().gate_up_affine_param_bytes()
    }

    fn single_projection_shape(self) -> AffineQuantizedMatmulShape {
        self.gate_up_shape().single_projection_shape()
    }

    fn gate_up_shape(self) -> AffineQuantizedGateUpSiluShape {
        AffineQuantizedGateUpSiluShape {
            m: self.m,
            intermediate_dim: self.intermediate_dim,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            dtype: self.dtype,
        }
    }
}

pub struct AffineQuantizedSplitGateUpSiluKernel {
    shape: AffineQuantizedSplitGateUpSiluShape,
    kernel: Kernel,
    dispatch: AffineQuantizedGateUpSiluDispatch,
}

#[derive(Clone, Copy, Debug)]
pub struct GatherAffineQuantizedMatmulShape {
    pub num_routes: i32,
    pub num_input_vectors: i32,
    pub n: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub dtype: Dtype,
}

impl GatherAffineQuantizedMatmulShape {
    pub fn validate(self) {
        assert!(self.num_routes > 0);
        assert!(self.num_input_vectors > 0);
        assert!(self.n > 0);
        assert!(self.k > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.k % self.group_size, 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16));
    }

    pub fn output_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "gather affine output",
            &[self.num_routes as usize, self.n as usize],
            self.dtype,
        )
    }

    pub fn input_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "gather affine input",
            &[self.num_input_vectors as usize, self.k as usize],
            self.dtype,
        )
    }

    pub fn weight_bytes_per_expert(self) -> usize {
        AffineQuantizedMatmulShape {
            m: 1,
            n: self.n,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .weight_bytes()
    }

    pub fn affine_param_bytes_per_expert(self) -> usize {
        AffineQuantizedMatmulShape {
            m: 1,
            n: self.n,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .affine_param_bytes()
    }
}

pub struct GatherAffineQuantizedMatmulKernel {
    shape: GatherAffineQuantizedMatmulShape,
    kernel: Kernel,
    fast: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct GatherAffineQuantizedGateUpSiluShape {
    pub num_routes: i32,
    pub num_input_vectors: i32,
    pub intermediate_dim: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub dtype: Dtype,
}

impl GatherAffineQuantizedGateUpSiluShape {
    pub fn validate(self) {
        assert!(self.num_routes > 0);
        assert!(self.num_input_vectors > 0);
        assert!(self.intermediate_dim > 0);
        assert!(self.k > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.k % self.group_size, 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16));
    }

    pub fn output_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "gather fused MLP output",
            &[self.num_routes as usize, self.intermediate_dim as usize],
            self.dtype,
        )
    }

    pub fn input_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "gather fused MLP input",
            &[self.num_input_vectors as usize, self.k as usize],
            self.dtype,
        )
    }

    pub fn weight_bytes_per_expert(self) -> usize {
        AffineQuantizedMatmulShape {
            m: 1,
            n: self.intermediate_dim,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .weight_bytes()
    }

    pub fn affine_param_bytes_per_expert(self) -> usize {
        AffineQuantizedMatmulShape {
            m: 1,
            n: self.intermediate_dim,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .affine_param_bytes()
    }
}

pub struct GatherAffineQuantizedGateUpSiluKernel {
    shape: GatherAffineQuantizedGateUpSiluShape,
    kernel: Kernel,
}

#[derive(Clone, Copy, Debug)]
pub struct RaggedExpertMajorAffineQuantizedGateUpSiluShape {
    pub num_experts: i32,
    pub num_routes: i32,
    pub intermediate_dim: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub dtype: Dtype,
}

impl RaggedExpertMajorAffineQuantizedGateUpSiluShape {
    pub fn validate(self) {
        assert!(self.num_experts > 0);
        assert!(self.num_routes > 0);
        assert!(self.intermediate_dim > 0);
        assert!(self.k > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.k % self.group_size, 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16));
    }

    pub fn output_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "expert-major fused MLP output",
            &[self.num_routes as usize, self.intermediate_dim as usize],
            self.dtype,
        )
    }

    pub fn input_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "expert-major fused MLP input",
            &[self.num_routes as usize, self.k as usize],
            self.dtype,
        )
    }

    pub fn weight_bytes_per_expert(self) -> usize {
        AffineQuantizedMatmulShape {
            m: 1,
            n: self.intermediate_dim,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .weight_bytes()
    }

    pub fn affine_param_bytes_per_expert(self) -> usize {
        AffineQuantizedMatmulShape {
            m: 1,
            n: self.intermediate_dim,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .affine_param_bytes()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RaggedExpertMajorAffineQuantizedMatmulShape {
    pub num_experts: i32,
    pub num_routes: i32,
    pub n: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub dtype: Dtype,
}

impl RaggedExpertMajorAffineQuantizedMatmulShape {
    pub fn validate(self) {
        assert!(self.num_experts > 0);
        assert!(self.num_routes > 0);
        assert!(self.n > 0);
        assert!(self.k > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.k % self.group_size, 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16));
    }

    pub fn output_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "expert-major affine output",
            &[self.num_routes as usize, self.n as usize],
            self.dtype,
        )
    }

    pub fn input_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "expert-major affine input",
            &[self.num_routes as usize, self.k as usize],
            self.dtype,
        )
    }

    pub fn weight_bytes_per_expert(self) -> usize {
        AffineQuantizedMatmulShape {
            m: 1,
            n: self.n,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .weight_bytes()
    }

    pub fn affine_param_bytes_per_expert(self) -> usize {
        AffineQuantizedMatmulShape {
            m: 1,
            n: self.n,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
        .affine_param_bytes()
    }
}

pub struct RaggedExpertMajorAffineQuantizedGateUpSiluKernel {
    shape: RaggedExpertMajorAffineQuantizedGateUpSiluShape,
    kernel: Kernel,
}

pub struct RaggedExpertMajorAffineQuantizedMatmulKernel {
    shape: RaggedExpertMajorAffineQuantizedMatmulShape,
    kernel: Kernel,
}

impl GatherAffineQuantizedMatmulKernel {
    pub fn new(device: &Device, shape: GatherAffineQuantizedMatmulShape) -> Self {
        shape.validate();
        let type_string = metal_type_string(shape.dtype);
        let bn = 8;
        let fast = shape.n % bn == 0 && shape.k % 512 == 0;
        let func = if fast { "gather_qmv_fast" } else { "gather_qmv" };
        let kernel_name = format!("{func}_{type_string}_gs_{}_b_{}", shape.group_size, shape.bits);
        let template_definition = template_definition(
            &kernel_name,
            func,
            &[
                type_string.to_string(),
                shape.group_size.to_string(),
                shape.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&template_definition);
        let kernel = Kernel::new(device, &source, &kernel_name);
        Self { shape, kernel, fast }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        lhs_indices: &'a Buffer,
        rhs_indices: &'a Buffer,
    ) -> GatherAffineQuantizedMatmulInvocation<'a> {
        self.invoke_with_shape(
            self.shape,
            output,
            input,
            weight,
            scales,
            biases,
            lhs_indices,
            rhs_indices,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_shape<'a>(
        &'a self,
        shape: GatherAffineQuantizedMatmulShape,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        lhs_indices: &'a Buffer,
        rhs_indices: &'a Buffer,
    ) -> GatherAffineQuantizedMatmulInvocation<'a> {
        GatherAffineQuantizedMatmulInvocation {
            kernel: self,
            shape,
            output,
            input,
            weight,
            scales,
            biases,
            lhs_indices,
            rhs_indices,
        }
    }
}

impl GatherAffineQuantizedGateUpSiluKernel {
    pub fn new(device: &Device, shape: GatherAffineQuantizedGateUpSiluShape) -> Self {
        shape.validate();
        let type_string = metal_type_string(shape.dtype);
        let kernel_name = format!(
            "token_major_fused_gate_up_silu_{type_string}_gs_{}_b_{}",
            shape.group_size, shape.bits
        );
        let template_definition = template_definition(
            &kernel_name,
            "token_major_fused_gate_up_silu",
            &[
                type_string.to_string(),
                shape.group_size.to_string(),
                shape.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&format!("{FUSED_GATE_UP_SILU_SOURCE}\n{template_definition}"));
        let kernel = Kernel::new(device, &source, &kernel_name);
        Self { shape, kernel }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        output: &'a Buffer,
        input: &'a Buffer,
        gate_weight: &'a Buffer,
        gate_scales: &'a Buffer,
        gate_biases: &'a Buffer,
        up_weight: &'a Buffer,
        up_scales: &'a Buffer,
        up_biases: &'a Buffer,
        lhs_indices: &'a Buffer,
        rhs_indices: &'a Buffer,
    ) -> GatherAffineQuantizedGateUpSiluInvocation<'a> {
        self.invoke_with_shape(
            self.shape,
            output,
            input,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            lhs_indices,
            rhs_indices,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_shape<'a>(
        &'a self,
        shape: GatherAffineQuantizedGateUpSiluShape,
        output: &'a Buffer,
        input: &'a Buffer,
        gate_weight: &'a Buffer,
        gate_scales: &'a Buffer,
        gate_biases: &'a Buffer,
        up_weight: &'a Buffer,
        up_scales: &'a Buffer,
        up_biases: &'a Buffer,
        lhs_indices: &'a Buffer,
        rhs_indices: &'a Buffer,
    ) -> GatherAffineQuantizedGateUpSiluInvocation<'a> {
        GatherAffineQuantizedGateUpSiluInvocation {
            kernel: self,
            shape,
            output,
            input,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            lhs_indices,
            rhs_indices,
        }
    }
}

impl RaggedExpertMajorAffineQuantizedGateUpSiluKernel {
    pub fn new(device: &Device, shape: RaggedExpertMajorAffineQuantizedGateUpSiluShape) -> Self {
        shape.validate();
        let type_string = metal_type_string(shape.dtype);
        let kernel_name = format!(
            "expert_major_fused_gate_up_silu_{type_string}_gs_{}_b_{}",
            shape.group_size, shape.bits
        );
        let template_definition = template_definition(
            &kernel_name,
            "expert_major_fused_gate_up_silu",
            &[
                type_string.to_string(),
                shape.group_size.to_string(),
                shape.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&format!("{FUSED_GATE_UP_SILU_SOURCE}\n{template_definition}"));
        let kernel = Kernel::new(device, &source, &kernel_name);
        Self { shape, kernel }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        output: &'a Buffer,
        input: &'a Buffer,
        gate_weight: &'a Buffer,
        gate_scales: &'a Buffer,
        gate_biases: &'a Buffer,
        up_weight: &'a Buffer,
        up_scales: &'a Buffer,
        up_biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorAffineQuantizedGateUpSiluInvocation<'a> {
        self.invoke_with_shape(
            self.shape,
            output,
            input,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            experts_by_route,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_shape<'a>(
        &'a self,
        shape: RaggedExpertMajorAffineQuantizedGateUpSiluShape,
        output: &'a Buffer,
        input: &'a Buffer,
        gate_weight: &'a Buffer,
        gate_scales: &'a Buffer,
        gate_biases: &'a Buffer,
        up_weight: &'a Buffer,
        up_scales: &'a Buffer,
        up_biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorAffineQuantizedGateUpSiluInvocation<'a> {
        RaggedExpertMajorAffineQuantizedGateUpSiluInvocation {
            kernel: self,
            shape,
            output,
            input,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            experts_by_route,
        }
    }
}

impl RaggedExpertMajorAffineQuantizedMatmulKernel {
    pub fn new(device: &Device, shape: RaggedExpertMajorAffineQuantizedMatmulShape) -> Self {
        shape.validate();
        let type_string = metal_type_string(shape.dtype);
        let kernel_name = format!(
            "expert_major_down_matmul_{type_string}_gs_{}_b_{}",
            shape.group_size, shape.bits
        );
        let template_definition = template_definition(
            &kernel_name,
            "expert_major_down_matmul",
            &[
                type_string.to_string(),
                shape.group_size.to_string(),
                shape.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&format!("{FUSED_GATE_UP_SILU_SOURCE}\n{template_definition}"));
        let kernel = Kernel::new(device, &source, &kernel_name);
        Self { shape, kernel }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorAffineQuantizedMatmulInvocation<'a> {
        self.invoke_with_shape(self.shape, output, input, weight, scales, biases, experts_by_route)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_shape<'a>(
        &'a self,
        shape: RaggedExpertMajorAffineQuantizedMatmulShape,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorAffineQuantizedMatmulInvocation<'a> {
        RaggedExpertMajorAffineQuantizedMatmulInvocation {
            kernel: self,
            shape,
            output,
            input,
            weight,
            scales,
            biases,
            experts_by_route,
        }
    }
}

pub struct GatherAffineQuantizedMatmulInvocation<'a> {
    kernel: &'a GatherAffineQuantizedMatmulKernel,
    shape: GatherAffineQuantizedMatmulShape,
    output: &'a Buffer,
    input: &'a Buffer,
    weight: &'a Buffer,
    scales: &'a Buffer,
    biases: &'a Buffer,
    lhs_indices: &'a Buffer,
    rhs_indices: &'a Buffer,
}

pub struct GatherAffineQuantizedGateUpSiluInvocation<'a> {
    kernel: &'a GatherAffineQuantizedGateUpSiluKernel,
    shape: GatherAffineQuantizedGateUpSiluShape,
    output: &'a Buffer,
    input: &'a Buffer,
    gate_weight: &'a Buffer,
    gate_scales: &'a Buffer,
    gate_biases: &'a Buffer,
    up_weight: &'a Buffer,
    up_scales: &'a Buffer,
    up_biases: &'a Buffer,
    lhs_indices: &'a Buffer,
    rhs_indices: &'a Buffer,
}

pub struct RaggedExpertMajorAffineQuantizedGateUpSiluInvocation<'a> {
    kernel: &'a RaggedExpertMajorAffineQuantizedGateUpSiluKernel,
    shape: RaggedExpertMajorAffineQuantizedGateUpSiluShape,
    output: &'a Buffer,
    input: &'a Buffer,
    gate_weight: &'a Buffer,
    gate_scales: &'a Buffer,
    gate_biases: &'a Buffer,
    up_weight: &'a Buffer,
    up_scales: &'a Buffer,
    up_biases: &'a Buffer,
    experts_by_route: &'a Buffer,
}

pub struct RaggedExpertMajorAffineQuantizedMatmulInvocation<'a> {
    kernel: &'a RaggedExpertMajorAffineQuantizedMatmulKernel,
    shape: RaggedExpertMajorAffineQuantizedMatmulShape,
    output: &'a Buffer,
    input: &'a Buffer,
    weight: &'a Buffer,
    scales: &'a Buffer,
    biases: &'a Buffer,
    experts_by_route: &'a Buffer,
}

impl Operator for RaggedExpertMajorAffineQuantizedMatmulInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let shape = self.shape;
        validate_ragged_expert_major_down_matmul_kernel_shape(self.kernel.shape, shape);
        validate_ragged_expert_major_down_matmul_buffer_ranges(
            shape,
            self.output,
            self.input,
            self.weight,
            self.scales,
            self.biases,
            self.experts_by_route,
        );

        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.weight, 0);
        builder.set_buffer_read(1, self.scales, 0);
        builder.set_buffer_read(2, self.biases, 0);
        builder.set_buffer_read(3, self.input, 0);
        builder.set_buffer_read(4, self.experts_by_route, 0);
        builder.set_buffer_write(5, self.output, 0);
        builder.set_i32(6, shape.k);
        builder.set_i32(7, shape.n);
        builder.set_i32(8, shape.num_experts);
        builder.dispatch_threadblocks(
            (shape.num_routes as usize, ceil_div_i32(shape.n, 8) as usize, 1),
            (32, 2, 1),
        );
    }
}

impl Operator for RaggedExpertMajorAffineQuantizedGateUpSiluInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let shape = self.shape;
        validate_ragged_expert_major_gate_up_silu_kernel_shape(self.kernel.shape, shape);
        validate_ragged_expert_major_gate_up_silu_buffer_ranges(
            shape,
            self.output,
            self.input,
            self.gate_weight,
            self.gate_scales,
            self.gate_biases,
            self.up_weight,
            self.up_scales,
            self.up_biases,
            self.experts_by_route,
        );

        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.gate_weight, 0);
        builder.set_buffer_read(1, self.gate_scales, 0);
        builder.set_buffer_read(2, self.gate_biases, 0);
        builder.set_buffer_read(3, self.up_weight, 0);
        builder.set_buffer_read(4, self.up_scales, 0);
        builder.set_buffer_read(5, self.up_biases, 0);
        builder.set_buffer_read(6, self.input, 0);
        builder.set_buffer_read(7, self.experts_by_route, 0);
        builder.set_buffer_write(8, self.output, 0);
        builder.set_i32(9, shape.k);
        builder.set_i32(10, shape.intermediate_dim);
        builder.set_i32(11, shape.num_experts);
        builder.dispatch_threadblocks(
            (
                shape.num_routes as usize,
                ceil_div_i32(shape.intermediate_dim, 8) as usize,
                1,
            ),
            (32, 2, 1),
        );
    }
}

impl Operator for GatherAffineQuantizedGateUpSiluInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.record_compute(builder);
    }
}

impl GatherAffineQuantizedGateUpSiluInvocation<'_> {
    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.shape;
        validate_gather_gate_up_silu_kernel_shape(self.kernel.shape, shape);
        validate_gather_gate_up_silu_buffer_ranges(
            shape,
            self.output,
            self.input,
            self.gate_weight,
            self.gate_scales,
            self.gate_biases,
            self.up_weight,
            self.up_scales,
            self.up_biases,
            self.lhs_indices,
            self.rhs_indices,
        );
        let expert_weight_bytes = shape.weight_bytes_per_expert();
        let num_experts = self.gate_weight.len_bytes() / expert_weight_bytes;

        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.gate_weight, 0);
        builder.set_buffer_read(1, self.gate_scales, 0);
        builder.set_buffer_read(2, self.gate_biases, 0);
        builder.set_buffer_read(3, self.up_weight, 0);
        builder.set_buffer_read(4, self.up_scales, 0);
        builder.set_buffer_read(5, self.up_biases, 0);
        builder.set_buffer_read(6, self.input, 0);
        builder.set_buffer_read(7, self.lhs_indices, 0);
        builder.set_buffer_read(8, self.rhs_indices, 0);
        builder.set_buffer_write(9, self.output, 0);
        builder.set_i32(10, shape.k);
        builder.set_i32(11, shape.intermediate_dim);
        builder.set_i32(
            12,
            num_experts
                .try_into()
                .expect("gather fused MLP expert count must fit shader i32"),
        );
        builder.dispatch_threadblocks(
            (
                1,
                ceil_div_i32(shape.intermediate_dim, 8) as usize,
                shape.num_routes as usize,
            ),
            (32, 2, 1),
        );
    }
}

impl Operator for GatherAffineQuantizedMatmulInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.record_compute(builder);
    }
}

impl GatherAffineQuantizedMatmulInvocation<'_> {
    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.shape;
        validate_gather_matmul_kernel_shape(self.kernel.shape, shape);
        validate_gather_buffer_ranges(
            shape,
            self.output,
            self.input,
            self.weight,
            self.scales,
            self.biases,
            self.lhs_indices,
            self.rhs_indices,
        );
        let packed_k = packed_dim(shape.k, shape.bits);
        let groups = shape.k / shape.group_size;
        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.weight, 0);
        builder.set_buffer_read(1, self.scales, 0);
        builder.set_buffer_read(2, self.biases, 0);
        builder.set_buffer_read(3, self.input, 0);
        builder.set_buffer_read(4, self.lhs_indices, 0);
        builder.set_buffer_read(5, self.rhs_indices, 0);
        builder.set_buffer_write(6, self.output, 0);
        builder.set_i32(7, shape.k);
        builder.set_i32(8, shape.n);

        let x_shape = [shape.num_input_vectors, 1, 1, shape.k];
        let k_stride = i64::from(shape.k);
        let x_strides = [k_stride, k_stride, k_stride, 1_i64];
        let w_shape = [
            num_experts_from_buffer(shape, self.weight)
                .try_into()
                .expect("gather affine expert count must fit shader i32"),
            shape.n,
            packed_k,
        ];
        let w_expert_stride = i64::from(shape.n)
            .checked_mul(i64::from(packed_k))
            .expect("gather affine weight stride must fit i64");
        let affine_expert_stride = i64::from(shape.n)
            .checked_mul(i64::from(groups))
            .expect("gather affine scale/bias stride must fit i64");
        let w_strides = [w_expert_stride, i64::from(packed_k), 1_i64];
        let affine_strides = [affine_expert_stride, i64::from(groups), 1_i64];
        let batch_shape = [shape.num_routes];
        let route_strides = [1_i64];

        builder.set_i32(9, 2);
        builder.set_i32_slice(10, &x_shape);
        builder.set_i64_slice(11, &x_strides);
        builder.set_i32(12, 1);
        builder.set_i32_slice(13, &w_shape);
        builder.set_i64_slice(14, &w_strides);
        builder.set_i64_slice(15, &affine_strides);
        builder.set_i64_slice(16, &affine_strides);
        builder.set_i32(17, 1);
        builder.set_i32_slice(18, &batch_shape);
        builder.set_i64_slice(19, &route_strides);
        builder.set_i64_slice(20, &route_strides);

        let bn = 8;
        let bk = 32;
        let _ = self.kernel.fast;
        builder.dispatch_threadblocks(
            (1, ceil_div_i32(shape.n, bn) as usize, shape.num_routes as usize),
            (bk as usize, 2, 1),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_gather_buffer_ranges(
    shape: GatherAffineQuantizedMatmulShape,
    output: &Buffer,
    input: &Buffer,
    weight: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    lhs_indices: &Buffer,
    rhs_indices: &Buffer,
) {
    shape.validate();
    let output_bytes = shape.output_bytes();
    let input_bytes = shape.input_bytes();
    let expert_weight_bytes = shape.weight_bytes_per_expert();
    let expert_affine_bytes = shape.affine_param_bytes_per_expert();
    assert!(
        output_bytes <= output.len_bytes(),
        "gather affine quantized matmul output range out of bounds: shape={shape:?} required_bytes={output_bytes} \
         buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "gather affine quantized matmul input range out of bounds: shape={shape:?} required_bytes={input_bytes} \
         buffer_bytes={}",
        input.len_bytes()
    );
    assert!(
        weight.len_bytes() >= expert_weight_bytes && weight.len_bytes().is_multiple_of(expert_weight_bytes),
        "gather affine quantized matmul weight stack mismatch: shape={shape:?} per_expert_bytes={expert_weight_bytes} \
         buffer_bytes={}",
        weight.len_bytes()
    );
    let num_experts = num_experts_from_buffer(shape, weight);
    let required_affine_bytes = num_experts
        .checked_mul(expert_affine_bytes)
        .expect("gather affine quantized matmul affine byte count must fit usize");
    assert!(
        required_affine_bytes <= scales.len_bytes(),
        "gather affine quantized matmul scales stack too short: shape={shape:?} \
         required_bytes={required_affine_bytes} buffer_bytes={}",
        scales.len_bytes()
    );
    assert!(
        required_affine_bytes <= biases.len_bytes(),
        "gather affine quantized matmul biases stack too short: shape={shape:?} \
         required_bytes={required_affine_bytes} buffer_bytes={}",
        biases.len_bytes()
    );
    let index_bytes = shape.num_routes as usize * size_of::<u32>();
    assert!(
        index_bytes <= lhs_indices.len_bytes(),
        "gather affine quantized matmul lhs index buffer too short: shape={shape:?} required_bytes={index_bytes} \
         buffer_bytes={}",
        lhs_indices.len_bytes()
    );
    assert!(
        index_bytes <= rhs_indices.len_bytes(),
        "gather affine quantized matmul rhs index buffer too short: shape={shape:?} required_bytes={index_bytes} \
         buffer_bytes={}",
        rhs_indices.len_bytes()
    );
}

#[allow(clippy::too_many_arguments)]
fn validate_gather_gate_up_silu_buffer_ranges(
    shape: GatherAffineQuantizedGateUpSiluShape,
    output: &Buffer,
    input: &Buffer,
    gate_weight: &Buffer,
    gate_scales: &Buffer,
    gate_biases: &Buffer,
    up_weight: &Buffer,
    up_scales: &Buffer,
    up_biases: &Buffer,
    lhs_indices: &Buffer,
    rhs_indices: &Buffer,
) {
    shape.validate();
    let output_bytes = shape.output_bytes();
    let input_bytes = shape.input_bytes();
    let expert_weight_bytes = shape.weight_bytes_per_expert();
    let expert_affine_bytes = shape.affine_param_bytes_per_expert();
    assert!(
        output_bytes <= output.len_bytes(),
        "gather affine quantized gate/up/silu output range out of bounds: shape={shape:?} \
         required_bytes={output_bytes} buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "gather affine quantized gate/up/silu input range out of bounds: shape={shape:?} required_bytes={input_bytes} \
         buffer_bytes={}",
        input.len_bytes()
    );
    assert!(
        gate_weight.len_bytes() >= expert_weight_bytes && gate_weight.len_bytes().is_multiple_of(expert_weight_bytes),
        "gather affine quantized gate weight stack mismatch: shape={shape:?} per_expert_bytes={expert_weight_bytes} \
         buffer_bytes={}",
        gate_weight.len_bytes()
    );
    assert_eq!(
        gate_weight.len_bytes(),
        up_weight.len_bytes(),
        "gather affine quantized fused gate/up weight stacks must have matching expert count"
    );
    let num_experts = gate_weight.len_bytes() / expert_weight_bytes;
    let required_affine_bytes = num_experts
        .checked_mul(expert_affine_bytes)
        .expect("gather affine quantized gate/up/silu affine byte count must fit usize");
    assert!(required_affine_bytes <= gate_scales.len_bytes());
    assert!(required_affine_bytes <= gate_biases.len_bytes());
    assert!(required_affine_bytes <= up_scales.len_bytes());
    assert!(required_affine_bytes <= up_biases.len_bytes());
    let index_bytes = shape.num_routes as usize * size_of::<u32>();
    assert!(index_bytes <= lhs_indices.len_bytes());
    assert!(index_bytes <= rhs_indices.len_bytes());
}

#[allow(clippy::too_many_arguments)]
fn validate_ragged_expert_major_gate_up_silu_buffer_ranges(
    shape: RaggedExpertMajorAffineQuantizedGateUpSiluShape,
    output: &Buffer,
    input: &Buffer,
    gate_weight: &Buffer,
    gate_scales: &Buffer,
    gate_biases: &Buffer,
    up_weight: &Buffer,
    up_scales: &Buffer,
    up_biases: &Buffer,
    experts_by_route: &Buffer,
) {
    shape.validate();
    let output_bytes = shape.output_bytes();
    let input_bytes = shape.input_bytes();
    let weight_bytes = checked_product(
        "ragged expert-major gate/up weight byte length",
        &[shape.num_experts as usize, shape.weight_bytes_per_expert()],
    );
    let affine_param_bytes = checked_product(
        "ragged expert-major gate/up affine byte length",
        &[shape.num_experts as usize, shape.affine_param_bytes_per_expert()],
    );
    let route_index_bytes = checked_product(
        "ragged expert-major gate/up route-index byte length",
        &[shape.num_routes as usize, size_of::<u32>()],
    );
    assert!(
        output_bytes <= output.len_bytes(),
        "ragged expert-major gate/up/silu output range out of bounds: shape={shape:?} required_bytes={output_bytes} \
         buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "ragged expert-major gate/up/silu input range out of bounds: shape={shape:?} required_bytes={input_bytes} \
         buffer_bytes={}",
        input.len_bytes()
    );
    assert!(weight_bytes <= gate_weight.len_bytes());
    assert!(affine_param_bytes <= gate_scales.len_bytes());
    assert!(affine_param_bytes <= gate_biases.len_bytes());
    assert!(weight_bytes <= up_weight.len_bytes());
    assert!(affine_param_bytes <= up_scales.len_bytes());
    assert!(affine_param_bytes <= up_biases.len_bytes());
    assert!(route_index_bytes <= experts_by_route.len_bytes());
}

fn validate_ragged_expert_major_down_matmul_buffer_ranges(
    shape: RaggedExpertMajorAffineQuantizedMatmulShape,
    output: &Buffer,
    input: &Buffer,
    weight: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    experts_by_route: &Buffer,
) {
    shape.validate();
    let output_bytes = shape.output_bytes();
    let input_bytes = shape.input_bytes();
    let weight_bytes = checked_product(
        "ragged expert-major down weight byte length",
        &[shape.num_experts as usize, shape.weight_bytes_per_expert()],
    );
    let affine_param_bytes = checked_product(
        "ragged expert-major down affine byte length",
        &[shape.num_experts as usize, shape.affine_param_bytes_per_expert()],
    );
    let route_index_bytes = checked_product(
        "ragged expert-major down route-index byte length",
        &[shape.num_routes as usize, size_of::<u32>()],
    );
    assert!(
        output_bytes <= output.len_bytes(),
        "ragged expert-major matmul output range out of bounds: shape={shape:?} required_bytes={output_bytes} \
         buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "ragged expert-major matmul input range out of bounds: shape={shape:?} required_bytes={input_bytes} \
         buffer_bytes={}",
        input.len_bytes()
    );
    assert!(weight_bytes <= weight.len_bytes());
    assert!(affine_param_bytes <= scales.len_bytes());
    assert!(affine_param_bytes <= biases.len_bytes());
    assert!(route_index_bytes <= experts_by_route.len_bytes());
}

fn num_experts_from_buffer(shape: GatherAffineQuantizedMatmulShape, weight: &Buffer) -> usize {
    let expert_bytes = shape.weight_bytes_per_expert();
    assert!(expert_bytes > 0);
    assert_eq!(
        weight.len_bytes() % expert_bytes,
        0,
        "gather affine quantized matmul weight stack must contain whole experts"
    );
    weight.len_bytes() / expert_bytes
}

fn packed_dim(k: i32, bits: i32) -> i32 {
    assert!(k > 0);
    assert!(matches!(bits, 2 | 3 | 4 | 6 | 8));
    let total_bits = k.checked_mul(bits).expect("packed affine dimension must fit i32");
    assert_eq!(total_bits % 32, 0);
    total_bits / 32
}

#[derive(Clone, Copy, Debug)]
enum AffineQuantizedDispatch {
    QmmT { bm: usize, bn: usize, wm: usize, wn: usize },
    QmvQuad { bn: usize },
    Qmv { bn: usize, bk: usize },
}

#[derive(Clone, Copy, Debug)]
enum AffineQuantizedGateUpSiluDispatch {
    QmmT { bm: usize, bn: usize, wm: usize, wn: usize },
    Qmv { bn: usize, bk: usize },
}

impl AffineQuantizedMatmulKernel {
    pub fn new(device: &Device, shape: AffineQuantizedMatmulShape) -> Self {
        shape.validate();
        let (kernel_name, source, dispatch) = affine_kernel_source(shape);
        let kernel = Kernel::new(device, &source, &kernel_name);
        Self {
            shape,
            kernel,
            dispatch,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        output: &'a Buffer,
        output_offset_bytes: usize,
        input: &'a Buffer,
        input_offset_bytes: usize,
        weight: &'a Buffer,
        weight_offset_bytes: usize,
        scales: &'a Buffer,
        scales_offset_bytes: usize,
        biases: &'a Buffer,
        biases_offset_bytes: usize,
    ) -> AffineQuantizedMatmulInvocation<'a> {
        self.invoke_with_shape(
            self.shape,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            scales,
            scales_offset_bytes,
            biases,
            biases_offset_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_shape<'a>(
        &'a self,
        shape: AffineQuantizedMatmulShape,
        output: &'a Buffer,
        output_offset_bytes: usize,
        input: &'a Buffer,
        input_offset_bytes: usize,
        weight: &'a Buffer,
        weight_offset_bytes: usize,
        scales: &'a Buffer,
        scales_offset_bytes: usize,
        biases: &'a Buffer,
        biases_offset_bytes: usize,
    ) -> AffineQuantizedMatmulInvocation<'a> {
        AffineQuantizedMatmulInvocation {
            kernel: self,
            shape,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            scales,
            scales_offset_bytes,
            biases,
            biases_offset_bytes,
        }
    }
}

impl AffineQuantizedGateUpSiluKernel {
    pub fn new(device: &Device, shape: AffineQuantizedGateUpSiluShape) -> Self {
        shape.validate();
        let type_string = metal_type_string(shape.dtype);
        let (kernel_name, template_definition, dispatch) = affine_gate_up_silu_kernel_metadata(shape, type_string);
        let source = affine_quantized_source(&format!("{FUSED_GATE_UP_SILU_SOURCE}\n{template_definition}"));
        let kernel = Kernel::new(device, &source, &kernel_name);
        Self {
            shape,
            kernel,
            dispatch,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
    ) -> AffineQuantizedGateUpSiluInvocation<'a> {
        self.invoke_with_shape(self.shape, output, 0, input, 0, weight, 0, scales, 0, biases, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_offsets<'a>(
        &'a self,
        output: &'a Buffer,
        output_offset_bytes: usize,
        input: &'a Buffer,
        input_offset_bytes: usize,
        weight: &'a Buffer,
        weight_offset_bytes: usize,
        scales: &'a Buffer,
        scales_offset_bytes: usize,
        biases: &'a Buffer,
        biases_offset_bytes: usize,
    ) -> AffineQuantizedGateUpSiluInvocation<'a> {
        self.invoke_with_shape(
            self.shape,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            scales,
            scales_offset_bytes,
            biases,
            biases_offset_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_shape<'a>(
        &'a self,
        shape: AffineQuantizedGateUpSiluShape,
        output: &'a Buffer,
        output_offset_bytes: usize,
        input: &'a Buffer,
        input_offset_bytes: usize,
        weight: &'a Buffer,
        weight_offset_bytes: usize,
        scales: &'a Buffer,
        scales_offset_bytes: usize,
        biases: &'a Buffer,
        biases_offset_bytes: usize,
    ) -> AffineQuantizedGateUpSiluInvocation<'a> {
        AffineQuantizedGateUpSiluInvocation {
            kernel: self,
            shape,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            scales,
            scales_offset_bytes,
            biases,
            biases_offset_bytes,
        }
    }
}

impl AffineQuantizedSplitGateUpSiluKernel {
    pub fn new(device: &Device, shape: AffineQuantizedSplitGateUpSiluShape) -> Self {
        shape.validate();
        let type_string = metal_type_string(shape.dtype);
        let (kernel_name, template_definition, dispatch) =
            affine_split_gate_up_silu_kernel_metadata(shape, type_string);
        let source = affine_quantized_source(&format!("{FUSED_GATE_UP_SILU_SOURCE}\n{template_definition}"));
        let kernel = Kernel::new(device, &source, &kernel_name);
        Self {
            shape,
            kernel,
            dispatch,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        output: &'a Buffer,
        output_offset_bytes: usize,
        input: &'a Buffer,
        input_offset_bytes: usize,
        gate_weight: &'a Buffer,
        gate_weight_offset_bytes: usize,
        gate_scales: &'a Buffer,
        gate_scales_offset_bytes: usize,
        gate_biases: &'a Buffer,
        gate_biases_offset_bytes: usize,
        up_weight: &'a Buffer,
        up_weight_offset_bytes: usize,
        up_scales: &'a Buffer,
        up_scales_offset_bytes: usize,
        up_biases: &'a Buffer,
        up_biases_offset_bytes: usize,
    ) -> AffineQuantizedSplitGateUpSiluInvocation<'a> {
        self.invoke_with_shape(
            self.shape,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            gate_weight,
            gate_weight_offset_bytes,
            gate_scales,
            gate_scales_offset_bytes,
            gate_biases,
            gate_biases_offset_bytes,
            up_weight,
            up_weight_offset_bytes,
            up_scales,
            up_scales_offset_bytes,
            up_biases,
            up_biases_offset_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_shape<'a>(
        &'a self,
        shape: AffineQuantizedSplitGateUpSiluShape,
        output: &'a Buffer,
        output_offset_bytes: usize,
        input: &'a Buffer,
        input_offset_bytes: usize,
        gate_weight: &'a Buffer,
        gate_weight_offset_bytes: usize,
        gate_scales: &'a Buffer,
        gate_scales_offset_bytes: usize,
        gate_biases: &'a Buffer,
        gate_biases_offset_bytes: usize,
        up_weight: &'a Buffer,
        up_weight_offset_bytes: usize,
        up_scales: &'a Buffer,
        up_scales_offset_bytes: usize,
        up_biases: &'a Buffer,
        up_biases_offset_bytes: usize,
    ) -> AffineQuantizedSplitGateUpSiluInvocation<'a> {
        AffineQuantizedSplitGateUpSiluInvocation {
            kernel: self,
            shape,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            gate_weight,
            gate_weight_offset_bytes,
            gate_scales,
            gate_scales_offset_bytes,
            gate_biases,
            gate_biases_offset_bytes,
            up_weight,
            up_weight_offset_bytes,
            up_scales,
            up_scales_offset_bytes,
            up_biases,
            up_biases_offset_bytes,
        }
    }
}

pub struct AffineQuantizedGateUpSiluInvocation<'a> {
    kernel: &'a AffineQuantizedGateUpSiluKernel,
    shape: AffineQuantizedGateUpSiluShape,
    output: &'a Buffer,
    output_offset_bytes: usize,
    input: &'a Buffer,
    input_offset_bytes: usize,
    weight: &'a Buffer,
    weight_offset_bytes: usize,
    scales: &'a Buffer,
    scales_offset_bytes: usize,
    biases: &'a Buffer,
    biases_offset_bytes: usize,
}

pub struct AffineQuantizedSplitGateUpSiluInvocation<'a> {
    kernel: &'a AffineQuantizedSplitGateUpSiluKernel,
    shape: AffineQuantizedSplitGateUpSiluShape,
    output: &'a Buffer,
    output_offset_bytes: usize,
    input: &'a Buffer,
    input_offset_bytes: usize,
    gate_weight: &'a Buffer,
    gate_weight_offset_bytes: usize,
    gate_scales: &'a Buffer,
    gate_scales_offset_bytes: usize,
    gate_biases: &'a Buffer,
    gate_biases_offset_bytes: usize,
    up_weight: &'a Buffer,
    up_weight_offset_bytes: usize,
    up_scales: &'a Buffer,
    up_scales_offset_bytes: usize,
    up_biases: &'a Buffer,
    up_biases_offset_bytes: usize,
}

impl Operator for AffineQuantizedSplitGateUpSiluInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.record_compute(builder);
    }
}

impl AffineQuantizedSplitGateUpSiluInvocation<'_> {
    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.shape;
        validate_gate_up_silu_kernel_shape(self.kernel.shape.gate_up_shape(), shape.gate_up_shape());
        let projection_shape = shape.single_projection_shape();
        validate_buffer_ranges(
            projection_shape,
            self.output,
            self.output_offset_bytes,
            self.input,
            self.input_offset_bytes,
            self.gate_weight,
            self.gate_weight_offset_bytes,
            self.gate_scales,
            self.gate_scales_offset_bytes,
            self.gate_biases,
            self.gate_biases_offset_bytes,
        );
        validate_buffer_ranges(
            projection_shape,
            self.output,
            self.output_offset_bytes,
            self.input,
            self.input_offset_bytes,
            self.up_weight,
            self.up_weight_offset_bytes,
            self.up_scales,
            self.up_scales_offset_bytes,
            self.up_biases,
            self.up_biases_offset_bytes,
        );

        builder.set_kernel(&self.kernel.kernel);
        match self.kernel.dispatch {
            AffineQuantizedGateUpSiluDispatch::QmmT { bm, bn, wm, wn } => {
                builder.set_buffer_read(0, self.gate_weight, self.gate_weight_offset_bytes);
                builder.set_buffer_read(1, self.gate_scales, self.gate_scales_offset_bytes);
                builder.set_buffer_read(2, self.gate_biases, self.gate_biases_offset_bytes);
                builder.set_buffer_read(3, self.up_weight, self.up_weight_offset_bytes);
                builder.set_buffer_read(4, self.up_scales, self.up_scales_offset_bytes);
                builder.set_buffer_read(5, self.up_biases, self.up_biases_offset_bytes);
                builder.set_buffer_read(6, self.input, self.input_offset_bytes);
                builder.set_buffer_write(7, self.output, self.output_offset_bytes);
                builder.set_i32(8, shape.k);
                builder.set_i32(9, shape.intermediate_dim);
                builder.set_i32(10, shape.m);
                builder.dispatch_threadblocks(
                    (
                        ceil_div_i32(shape.intermediate_dim, bn as i32) as usize,
                        ceil_div_i32(shape.m, bm as i32) as usize,
                        1,
                    ),
                    (32, wn, wm),
                );
            },
            AffineQuantizedGateUpSiluDispatch::Qmv { bn, bk } => {
                builder.set_buffer_read(0, self.gate_weight, self.gate_weight_offset_bytes);
                builder.set_buffer_read(1, self.gate_scales, self.gate_scales_offset_bytes);
                builder.set_buffer_read(2, self.gate_biases, self.gate_biases_offset_bytes);
                builder.set_buffer_read(3, self.up_weight, self.up_weight_offset_bytes);
                builder.set_buffer_read(4, self.up_scales, self.up_scales_offset_bytes);
                builder.set_buffer_read(5, self.up_biases, self.up_biases_offset_bytes);
                builder.set_buffer_read(6, self.input, self.input_offset_bytes);
                builder.set_buffer_write(7, self.output, self.output_offset_bytes);
                builder.set_i32(8, shape.k);
                builder.set_i32(9, shape.intermediate_dim);
                builder.dispatch_threadblocks(
                    (
                        shape.m as usize,
                        ceil_div_i32(shape.intermediate_dim, bn as i32) as usize,
                        1,
                    ),
                    (bk, 2, 1),
                );
            },
        }
    }
}

impl Operator for AffineQuantizedGateUpSiluInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.record_compute(builder);
    }
}

impl AffineQuantizedGateUpSiluInvocation<'_> {
    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.shape;
        validate_gate_up_silu_kernel_shape(self.kernel.shape, shape);
        validate_gate_up_silu_buffer_ranges(
            shape,
            self.output,
            self.output_offset_bytes,
            self.input,
            self.input_offset_bytes,
            self.weight,
            self.weight_offset_bytes,
            self.scales,
            self.scales_offset_bytes,
            self.biases,
            self.biases_offset_bytes,
        );

        builder.set_kernel(&self.kernel.kernel);
        match self.kernel.dispatch {
            AffineQuantizedGateUpSiluDispatch::QmmT { bm, bn, wm, wn } => {
                let projection_shape = shape.single_projection_shape();
                let weight_offset = projection_shape.weight_bytes();
                let affine_offset = projection_shape.affine_param_bytes();
                builder.set_buffer_read(0, self.weight, self.weight_offset_bytes);
                builder.set_buffer_read(1, self.scales, self.scales_offset_bytes);
                builder.set_buffer_read(2, self.biases, self.biases_offset_bytes);
                builder.set_buffer_read(3, self.weight, self.weight_offset_bytes + weight_offset);
                builder.set_buffer_read(4, self.scales, self.scales_offset_bytes + affine_offset);
                builder.set_buffer_read(5, self.biases, self.biases_offset_bytes + affine_offset);
                builder.set_buffer_read(6, self.input, self.input_offset_bytes);
                builder.set_buffer_write(7, self.output, self.output_offset_bytes);
                builder.set_i32(8, shape.k);
                builder.set_i32(9, shape.intermediate_dim);
                builder.set_i32(10, shape.m);
                builder.dispatch_threadblocks(
                    (
                        ceil_div_i32(shape.intermediate_dim, bn as i32) as usize,
                        ceil_div_i32(shape.m, bm as i32) as usize,
                        1,
                    ),
                    (32, wn, wm),
                );
            },
            AffineQuantizedGateUpSiluDispatch::Qmv { bn, bk } => {
                builder.set_buffer_read(0, self.weight, self.weight_offset_bytes);
                builder.set_buffer_read(1, self.scales, self.scales_offset_bytes);
                builder.set_buffer_read(2, self.biases, self.biases_offset_bytes);
                builder.set_buffer_read(3, self.input, self.input_offset_bytes);
                builder.set_buffer_write(4, self.output, self.output_offset_bytes);
                builder.set_i32(5, shape.k);
                builder.set_i32(6, shape.intermediate_dim);
                builder.dispatch_threadblocks(
                    (
                        shape.m as usize,
                        ceil_div_i32(shape.intermediate_dim, bn as i32) as usize,
                        1,
                    ),
                    (bk, 2, 1),
                );
            },
        }
    }
}

pub struct AffineQuantizedMatmulInvocation<'a> {
    kernel: &'a AffineQuantizedMatmulKernel,
    shape: AffineQuantizedMatmulShape,
    output: &'a Buffer,
    output_offset_bytes: usize,
    input: &'a Buffer,
    input_offset_bytes: usize,
    weight: &'a Buffer,
    weight_offset_bytes: usize,
    scales: &'a Buffer,
    scales_offset_bytes: usize,
    biases: &'a Buffer,
    biases_offset_bytes: usize,
}

impl Operator for AffineQuantizedMatmulInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.record_compute(builder);
    }
}

impl AffineQuantizedMatmulInvocation<'_> {
    fn record_compute(self, builder: &CommandRecorder) {
        let kernel = self.kernel;
        let output = self.output;
        let output_offset_bytes = self.output_offset_bytes;
        let input = self.input;
        let input_offset_bytes = self.input_offset_bytes;
        let weight = self.weight;
        let weight_offset_bytes = self.weight_offset_bytes;
        let scales = self.scales;
        let scales_offset_bytes = self.scales_offset_bytes;
        let biases = self.biases;
        let biases_offset_bytes = self.biases_offset_bytes;
        let shape = self.shape;
        validate_matmul_kernel_shape(kernel.shape, shape);
        validate_buffer_ranges(
            shape,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            scales,
            scales_offset_bytes,
            biases,
            biases_offset_bytes,
        );

        builder.set_kernel(&kernel.kernel);
        builder.set_buffer_read(0, weight, weight_offset_bytes);
        builder.set_buffer_read(1, scales, scales_offset_bytes);
        builder.set_buffer_read(2, biases, biases_offset_bytes);
        builder.set_buffer_read(3, input, input_offset_bytes);
        builder.set_buffer_write(4, output, output_offset_bytes);
        builder.set_i32(5, shape.k);
        builder.set_i32(6, shape.n);

        match kernel.dispatch {
            AffineQuantizedDispatch::QmmT { bm, bn, wm, wn } => {
                builder.set_i32(7, shape.m);
                set_non_batched_qmm_metadata(builder);
                builder.dispatch_threadblocks(
                    (
                        ceil_div_i32(shape.n, bn as i32) as usize,
                        ceil_div_i32(shape.m, bm as i32) as usize,
                        1,
                    ),
                    (32, wn, wm),
                );
            },
            AffineQuantizedDispatch::QmvQuad { bn } => {
                set_non_batched_qmv_metadata(builder);
                builder.dispatch_threadblocks(
                    (shape.m as usize, ceil_div_i32(shape.n, bn as i32) as usize, 1),
                    (32, 1, 1),
                );
            },
            AffineQuantizedDispatch::Qmv { bn, bk } => {
                set_non_batched_qmv_metadata(builder);
                builder.dispatch_threadblocks(
                    (shape.m as usize, ceil_div_i32(shape.n, bn as i32) as usize, 1),
                    (bk, 2, 1),
                );
            },
        }
    }
}

fn set_non_batched_qmv_metadata(builder: &CommandRecorder) {
    const DUMMY_SHAPE: [i32; 1] = [0];
    const DUMMY_STRIDES: [i64; 1] = [0];
    builder.set_i32(7, 0);
    builder.set_i32_slice(8, &DUMMY_SHAPE);
    builder.set_i64_slice(9, &DUMMY_STRIDES);
    builder.set_i32(10, 0);
    builder.set_i32_slice(11, &DUMMY_SHAPE);
    builder.set_i64_slice(12, &DUMMY_STRIDES);
    builder.set_i64_slice(13, &DUMMY_STRIDES);
    builder.set_i64_slice(14, &DUMMY_STRIDES);
}

fn set_non_batched_qmm_metadata(builder: &CommandRecorder) {
    const DUMMY_SHAPE: [i32; 1] = [0];
    const DUMMY_STRIDES: [i64; 1] = [0];
    builder.set_i32(8, 0);
    builder.set_i32_slice(9, &DUMMY_SHAPE);
    builder.set_i64_slice(10, &DUMMY_STRIDES);
    builder.set_i32(11, 0);
    builder.set_i32_slice(12, &DUMMY_SHAPE);
    builder.set_i64_slice(13, &DUMMY_STRIDES);
    builder.set_i64_slice(14, &DUMMY_STRIDES);
    builder.set_i64_slice(15, &DUMMY_STRIDES);
}

#[allow(clippy::too_many_arguments)]
fn validate_buffer_ranges(
    shape: AffineQuantizedMatmulShape,
    output: &Buffer,
    output_offset_bytes: usize,
    input: &Buffer,
    input_offset_bytes: usize,
    weight: &Buffer,
    weight_offset_bytes: usize,
    scales: &Buffer,
    scales_offset_bytes: usize,
    biases: &Buffer,
    biases_offset_bytes: usize,
) {
    shape.validate();
    let output_bytes = shape.output_bytes();
    let input_bytes = shape.input_bytes();
    let weight_bytes = shape.weight_bytes();
    let affine_param_bytes = shape.affine_param_bytes();
    assert!(
        output_offset_bytes + output_bytes <= output.len_bytes(),
        "affine quantized matmul output range out of bounds: shape={shape:?} offset_bytes={output_offset_bytes} \
         required_bytes={output_bytes} buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_offset_bytes + input_bytes <= input.len_bytes(),
        "affine quantized matmul input range out of bounds: shape={shape:?} offset_bytes={input_offset_bytes} \
         required_bytes={input_bytes} buffer_bytes={}",
        input.len_bytes()
    );
    assert!(
        weight_offset_bytes + weight_bytes <= weight.len_bytes(),
        "affine quantized matmul weight range out of bounds: shape={shape:?} offset_bytes={weight_offset_bytes} \
         required_bytes={weight_bytes} buffer_bytes={}",
        weight.len_bytes()
    );
    assert!(
        scales_offset_bytes + affine_param_bytes <= scales.len_bytes(),
        "affine quantized matmul scales range out of bounds: shape={shape:?} offset_bytes={scales_offset_bytes} \
         required_bytes={affine_param_bytes} buffer_bytes={}",
        scales.len_bytes()
    );
    assert!(
        biases_offset_bytes + affine_param_bytes <= biases.len_bytes(),
        "affine quantized matmul biases range out of bounds: shape={shape:?} offset_bytes={biases_offset_bytes} \
         required_bytes={affine_param_bytes} buffer_bytes={}",
        biases.len_bytes()
    );
}

fn validate_matmul_kernel_shape(
    kernel_shape: AffineQuantizedMatmulShape,
    invocation_shape: AffineQuantizedMatmulShape,
) {
    invocation_shape.validate();
    debug_assert_eq!(kernel_shape.n, invocation_shape.n);
    debug_assert_eq!(kernel_shape.k, invocation_shape.k);
    debug_assert_eq!(kernel_shape.group_size, invocation_shape.group_size);
    debug_assert_eq!(kernel_shape.bits, invocation_shape.bits);
    debug_assert_eq!(kernel_shape.input_dtype, invocation_shape.input_dtype);
    debug_assert_eq!(kernel_shape.output_dtype, invocation_shape.output_dtype);
    debug_assert_eq!(kernel_shape.affine_dtype, invocation_shape.affine_dtype);
}

fn validate_gate_up_silu_kernel_shape(
    kernel_shape: AffineQuantizedGateUpSiluShape,
    invocation_shape: AffineQuantizedGateUpSiluShape,
) {
    invocation_shape.validate();
    debug_assert_eq!(kernel_shape.intermediate_dim, invocation_shape.intermediate_dim);
    debug_assert_eq!(kernel_shape.k, invocation_shape.k);
    debug_assert_eq!(kernel_shape.group_size, invocation_shape.group_size);
    debug_assert_eq!(kernel_shape.bits, invocation_shape.bits);
    debug_assert_eq!(kernel_shape.dtype, invocation_shape.dtype);
}

fn validate_gather_matmul_kernel_shape(
    kernel_shape: GatherAffineQuantizedMatmulShape,
    invocation_shape: GatherAffineQuantizedMatmulShape,
) {
    invocation_shape.validate();
    debug_assert_eq!(kernel_shape.n, invocation_shape.n);
    debug_assert_eq!(kernel_shape.k, invocation_shape.k);
    debug_assert_eq!(kernel_shape.group_size, invocation_shape.group_size);
    debug_assert_eq!(kernel_shape.bits, invocation_shape.bits);
    debug_assert_eq!(kernel_shape.dtype, invocation_shape.dtype);
}

fn validate_gather_gate_up_silu_kernel_shape(
    kernel_shape: GatherAffineQuantizedGateUpSiluShape,
    invocation_shape: GatherAffineQuantizedGateUpSiluShape,
) {
    invocation_shape.validate();
    debug_assert_eq!(kernel_shape.intermediate_dim, invocation_shape.intermediate_dim);
    debug_assert_eq!(kernel_shape.k, invocation_shape.k);
    debug_assert_eq!(kernel_shape.group_size, invocation_shape.group_size);
    debug_assert_eq!(kernel_shape.bits, invocation_shape.bits);
    debug_assert_eq!(kernel_shape.dtype, invocation_shape.dtype);
}

fn validate_ragged_expert_major_gate_up_silu_kernel_shape(
    kernel_shape: RaggedExpertMajorAffineQuantizedGateUpSiluShape,
    invocation_shape: RaggedExpertMajorAffineQuantizedGateUpSiluShape,
) {
    invocation_shape.validate();
    debug_assert_eq!(kernel_shape.intermediate_dim, invocation_shape.intermediate_dim);
    debug_assert_eq!(kernel_shape.k, invocation_shape.k);
    debug_assert_eq!(kernel_shape.group_size, invocation_shape.group_size);
    debug_assert_eq!(kernel_shape.bits, invocation_shape.bits);
    debug_assert_eq!(kernel_shape.dtype, invocation_shape.dtype);
}

fn validate_ragged_expert_major_down_matmul_kernel_shape(
    kernel_shape: RaggedExpertMajorAffineQuantizedMatmulShape,
    invocation_shape: RaggedExpertMajorAffineQuantizedMatmulShape,
) {
    invocation_shape.validate();
    debug_assert_eq!(kernel_shape.n, invocation_shape.n);
    debug_assert_eq!(kernel_shape.k, invocation_shape.k);
    debug_assert_eq!(kernel_shape.group_size, invocation_shape.group_size);
    debug_assert_eq!(kernel_shape.bits, invocation_shape.bits);
    debug_assert_eq!(kernel_shape.dtype, invocation_shape.dtype);
}

#[allow(clippy::too_many_arguments)]
fn validate_gate_up_silu_buffer_ranges(
    shape: AffineQuantizedGateUpSiluShape,
    output: &Buffer,
    output_offset_bytes: usize,
    input: &Buffer,
    input_offset_bytes: usize,
    weight: &Buffer,
    weight_offset_bytes: usize,
    scales: &Buffer,
    scales_offset_bytes: usize,
    biases: &Buffer,
    biases_offset_bytes: usize,
) {
    shape.validate();
    let output_bytes = shape.output_bytes();
    let input_bytes = shape.input_bytes();
    let weight_bytes = shape.gate_up_weight_bytes();
    let affine_param_bytes = shape.gate_up_affine_param_bytes();
    assert!(
        output_offset_bytes + output_bytes <= output.len_bytes(),
        "affine quantized gate/up/silu output range out of bounds: shape={shape:?} offset_bytes={output_offset_bytes} \
         required_bytes={output_bytes} buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_offset_bytes + input_bytes <= input.len_bytes(),
        "affine quantized gate/up/silu input range out of bounds: shape={shape:?} offset_bytes={input_offset_bytes} \
         required_bytes={input_bytes} buffer_bytes={}",
        input.len_bytes()
    );
    assert!(
        weight_offset_bytes + weight_bytes <= weight.len_bytes(),
        "affine quantized gate/up/silu weight range out of bounds: shape={shape:?} offset_bytes={weight_offset_bytes} \
         required_bytes={weight_bytes} buffer_bytes={}",
        weight.len_bytes()
    );
    assert!(
        scales_offset_bytes + affine_param_bytes <= scales.len_bytes(),
        "affine quantized gate/up/silu scales range out of bounds: shape={shape:?} offset_bytes={scales_offset_bytes} \
         required_bytes={affine_param_bytes} buffer_bytes={}",
        scales.len_bytes()
    );
    assert!(
        biases_offset_bytes + affine_param_bytes <= biases.len_bytes(),
        "affine quantized gate/up/silu biases range out of bounds: shape={shape:?} offset_bytes={biases_offset_bytes} \
         required_bytes={affine_param_bytes} buffer_bytes={}",
        biases.len_bytes()
    );
}

fn affine_kernel_source(shape: AffineQuantizedMatmulShape) -> (String, String, AffineQuantizedDispatch) {
    if shape.uses_same_dtype() {
        let type_string = metal_type_string(shape.input_dtype);
        let (kernel_name, template_definition, dispatch) = affine_kernel_metadata_same_dtype(shape, type_string);
        return (kernel_name, affine_quantized_source(&template_definition), dispatch);
    }

    let input_type = metal_type_string(shape.input_dtype);
    let output_type = metal_type_string(shape.output_dtype);
    let affine_type = metal_type_string(shape.affine_dtype);
    if shape.m >= qmv_batch_limit(shape.k, shape.n) {
        let wm = 2;
        let wn = 2;
        let bm = 32;
        let bn = 32;
        let aligned = shape.n % 32 == 0;
        let kernel_name = format!(
            "mixed_qmm_t_{input_type}_{affine_type}_{output_type}_gs_{}_b_{}_alN_{}",
            shape.group_size, shape.bits, aligned
        );
        let template_definition = template_definition(
            &kernel_name,
            "mixed_qmm_t",
            &[
                input_type.to_string(),
                affine_type.to_string(),
                output_type.to_string(),
                shape.group_size.to_string(),
                shape.bits.to_string(),
                aligned.to_string(),
            ],
        );
        return (
            kernel_name,
            affine_quantized_source(&format!("{MIXED_AFFINE_SOURCE}\n{template_definition}")),
            AffineQuantizedDispatch::QmmT { bm, bn, wm, wn },
        );
    }

    let fast = shape.n % 8 == 0 && shape.k % 512 == 0;
    let func = if fast { "mixed_qmv_fast" } else { "mixed_qmv" };
    let kernel_name = format!(
        "{func}_{input_type}_{affine_type}_{output_type}_gs_{}_b_{}",
        shape.group_size, shape.bits
    );
    let template_definition = template_definition(
        &kernel_name,
        func,
        &[
            input_type.to_string(),
            affine_type.to_string(),
            output_type.to_string(),
            shape.group_size.to_string(),
            shape.bits.to_string(),
        ],
    );
    (
        kernel_name,
        affine_quantized_source(&format!("{MIXED_AFFINE_SOURCE}\n{template_definition}")),
        AffineQuantizedDispatch::Qmv { bn: 8, bk: 32 },
    )
}

fn affine_kernel_metadata_same_dtype(
    shape: AffineQuantizedMatmulShape,
    type_string: &str,
) -> (String, String, AffineQuantizedDispatch) {
    if shape.m >= qmv_batch_limit(shape.k, shape.n) {
        let wm = 2;
        let wn = 2;
        let bm = 32;
        let bn = 32;
        let aligned = shape.n % 32 == 0;
        let kernel_name = format!(
            "qmm_t_{type_string}_gs_{}_b_{}_alN_{}_batch_0",
            shape.group_size, shape.bits, aligned
        );
        let template_definition = template_definition(
            &kernel_name,
            "qmm_t",
            &[
                type_string.to_string(),
                shape.group_size.to_string(),
                shape.bits.to_string(),
                aligned.to_string(),
                "false".to_string(),
            ],
        );
        return (
            kernel_name,
            template_definition,
            AffineQuantizedDispatch::QmmT { bm, bn, wm, wn },
        );
    }

    if matches!(shape.k, 64 | 128) && is_power_of_two(shape.bits) {
        let bn = 64;
        let kernel_name = format!(
            "qmv_quad_{type_string}_gs_{}_b_{}_d_{}_batch_0",
            shape.group_size, shape.bits, shape.k
        );
        let template_definition = template_definition(
            &kernel_name,
            "qmv_quad",
            &[
                type_string.to_string(),
                shape.group_size.to_string(),
                shape.bits.to_string(),
                shape.k.to_string(),
                "false".to_string(),
            ],
        );
        return (
            kernel_name,
            template_definition,
            AffineQuantizedDispatch::QmvQuad { bn },
        );
    }

    let bn = 8;
    let bk = 32;
    let fast = shape.n % bn as i32 == 0 && shape.k % 512 == 0;
    let func = if fast { "qmv_fast" } else { "qmv" };
    let kernel_name = format!("{func}_{type_string}_gs_{}_b_{}_batch_0", shape.group_size, shape.bits);
    let template_func = if fast { "qmv_fast" } else { "qmv" };
    let template_definition = template_definition(
        &kernel_name,
        template_func,
        &[
            type_string.to_string(),
            shape.group_size.to_string(),
            shape.bits.to_string(),
            "false".to_string(),
        ],
    );
    (
        kernel_name,
        template_definition,
        AffineQuantizedDispatch::Qmv { bn, bk },
    )
}

fn affine_gate_up_silu_kernel_metadata(
    shape: AffineQuantizedGateUpSiluShape,
    type_string: &str,
) -> (String, String, AffineQuantizedGateUpSiluDispatch) {
    let stacked_n = shape
        .intermediate_dim
        .checked_mul(2)
        .expect("dense MLP stacked gate/up dim must fit i32");
    if shape.m >= qmv_batch_limit(shape.k, stacked_n) {
        let wm = 2;
        let wn = 2;
        let bm = 32;
        let bn = 32;
        let aligned = shape.intermediate_dim % 32 == 0;
        let kernel_name = format!(
            "qmm_t_fused_gate_up_silu_{type_string}_gs_{}_b_{}_alN_{}",
            shape.group_size, shape.bits, aligned
        );
        let template_definition = template_definition(
            &kernel_name,
            "qmm_t_fused_gate_up_silu",
            &[
                type_string.to_string(),
                shape.group_size.to_string(),
                shape.bits.to_string(),
                aligned.to_string(),
            ],
        );
        return (
            kernel_name,
            template_definition,
            AffineQuantizedGateUpSiluDispatch::QmmT { bm, bn, wm, wn },
        );
    }

    let bn = 8;
    let bk = 32;
    let kernel_name = format!(
        "dense_fused_gate_up_silu_{type_string}_gs_{}_b_{}",
        shape.group_size, shape.bits
    );
    let template_definition = template_definition(
        &kernel_name,
        "dense_fused_gate_up_silu",
        &[
            type_string.to_string(),
            shape.group_size.to_string(),
            shape.bits.to_string(),
        ],
    );
    (
        kernel_name,
        template_definition,
        AffineQuantizedGateUpSiluDispatch::Qmv { bn, bk },
    )
}

fn affine_split_gate_up_silu_kernel_metadata(
    shape: AffineQuantizedSplitGateUpSiluShape,
    type_string: &str,
) -> (String, String, AffineQuantizedGateUpSiluDispatch) {
    let stacked_n = shape
        .intermediate_dim
        .checked_mul(2)
        .expect("split dense MLP stacked gate/up dim must fit i32");
    if shape.m >= qmv_batch_limit(shape.k, stacked_n) {
        let wm = 2;
        let wn = 2;
        let bm = 32;
        let bn = 32;
        let aligned = shape.intermediate_dim % 32 == 0;
        let kernel_name = format!(
            "qmm_t_split_fused_gate_up_silu_{type_string}_gs_{}_b_{}_alN_{}",
            shape.group_size, shape.bits, aligned
        );
        let template_definition = template_definition(
            &kernel_name,
            "qmm_t_fused_gate_up_silu",
            &[
                type_string.to_string(),
                shape.group_size.to_string(),
                shape.bits.to_string(),
                aligned.to_string(),
            ],
        );
        return (
            kernel_name,
            template_definition,
            AffineQuantizedGateUpSiluDispatch::QmmT { bm, bn, wm, wn },
        );
    }

    let bn = 8;
    let bk = 32;
    let kernel_name = format!(
        "split_fused_gate_up_silu_{type_string}_gs_{}_b_{}",
        shape.group_size, shape.bits
    );
    let template_definition = template_definition(
        &kernel_name,
        "split_fused_gate_up_silu",
        &[
            type_string.to_string(),
            shape.group_size.to_string(),
            shape.bits.to_string(),
        ],
    );
    (
        kernel_name,
        template_definition,
        AffineQuantizedGateUpSiluDispatch::Qmv { bn, bk },
    )
}

fn ceil_div_i32(value: i32, divisor: i32) -> i32 {
    assert!(value > 0);
    assert!(divisor > 0);
    (value + divisor - 1) / divisor
}

fn is_power_of_two(value: i32) -> bool {
    value > 0 && (value & (value - 1)) == 0
}

fn qmv_batch_limit(input_dim: i32, output_dim: i32) -> i32 {
    if input_dim <= 2048 && output_dim <= 2048 {
        18
    } else if input_dim <= 4096 && output_dim <= 4096 {
        12
    } else {
        10
    }
}

fn metal_type_string(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Float32 => "float",
        Dtype::Float16 => "float16_t",
        Dtype::Bfloat16 => "bfloat16_t",
        _ => panic!("affine quantized matmul dtype must be f32, f16, or bf16"),
    }
}

fn template_definition(kernel_name: &str, function_name: &str, args: &[String]) -> String {
    let instantiation = format!("{function_name}<{}>", args.join(", "));
    format!("\ntemplate [[host_name(\"{kernel_name}\")]] [[kernel]] decltype({instantiation}) {instantiation};\n")
}

fn affine_quantized_source(template_definition: &str) -> String {
    let root = mlx_metal_header_root();
    let mut included = HashSet::new();
    let mut source = String::new();
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/utils.h",
        &mut included,
    ));
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/steel/gemm/gemm.h",
        &mut included,
    ));
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/quantized.h",
        &mut included,
    ));
    source.push_str(template_definition);
    source
}

const MIXED_AFFINE_SOURCE: &str = r#"
template <typename InT, typename ParamT, typename OutT, int group_size, int bits>
[[kernel]] void mixed_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device ParamT* scales [[buffer(1)]],
    const device ParamT* biases [[buffer(2)]],
    const device InT* x [[buffer(3)]],
    device OutT* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)x_batch_ndims;
  (void)x_shape;
  (void)x_strides;
  (void)w_batch_ndims;
  (void)w_shape;
  (void)w_strides;
  (void)s_strides;
  (void)b_strides;

  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * static_cast<int64_t>(in_vec_size) + simd_lid * values_per_thread;
  y += tid.x * static_cast<int64_t>(out_vec_size) + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<InT, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device ParamT* sl = scales + row * in_vec_size_g;
      const device ParamT* bl = biases + row * in_vec_size_g;

      U s = static_cast<U>(sl[0]);
      U b = static_cast<U>(bl[0]);
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<OutT>(result[row]);
    }
  }
}

template <typename InT, typename ParamT, typename OutT, int group_size, int bits>
[[kernel]] void mixed_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device ParamT* scales [[buffer(1)]],
    const device ParamT* biases [[buffer(2)]],
    const device InT* x [[buffer(3)]],
    device OutT* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)x_batch_ndims;
  (void)x_shape;
  (void)x_strides;
  (void)w_batch_ndims;
  (void)w_shape;
  (void)w_strides;
  (void)s_strides;
  (void)b_strides;

  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * static_cast<int64_t>(in_vec_size) + simd_lid * values_per_thread;
  y += tid.x * static_cast<int64_t>(out_vec_size) + out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    U sum = load_vector<InT, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device ParamT* sl = scales + row * in_vec_size_g;
        const device ParamT* bl = biases + row * in_vec_size_g;
        result[row] += qdot<U, values_per_thread, bits>(
            wl, x_thread, static_cast<U>(sl[0]), static_cast<U>(bl[0]), sum);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    U sum = load_vector_safe<InT, U, values_per_thread, bits>(x, x_thread, remaining);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device ParamT* sl = scales + row * in_vec_size_g;
        const device ParamT* bl = biases + row * in_vec_size_g;
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, static_cast<U>(sl[0]), static_cast<U>(bl[0]), sum, remaining);
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    if (out_row + row < out_vec_size) {
      U value = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<OutT>(value);
      }
    }
  }
}

template <
    typename InT,
    typename OutT,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct PsiDecMixedBlockLoader {
  STEEL_CONST short vec_size = n_reads;

  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup OutT* dst;
  const device InT* src;

  PsiDecMixedBlockLoader(
      const device InT* src_,
      const int src_ld_,
      threadgroup OutT* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = static_cast<OutT>(src[i * src_ld + j]);
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = OutT(0);
        }
      }
      return;
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        const bool valid = (i < src_tile_dim.y) && (j < src_tile_dim.x);
        dst[i * dst_ld + j] = valid ? static_cast<OutT>(src[i * src_ld + j]) : OutT(0);
      }
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
  }
};

template <
    typename ParamT,
    typename OutT,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct PsiDecMixedQuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
      "Template undefined for bits not in {2, 3, 4, 6, 8}");

  MLX_MTL_CONST short pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  MLX_MTL_CONST short bytes_per_pack = (bits == 3 || bits == 6) ? 3 : 1;
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup OutT* dst;
  const device uint8_t* src;
  const device ParamT* scales;
  const device ParamT* biases;

  PsiDecMixedQuantizedBlockLoader(
      const device uint8_t* src_,
      const device ParamT* scales_,
      const device ParamT* biases_,
      const int src_ld_,
      threadgroup OutT* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  METAL_FUNC void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    OutT scale = static_cast<OutT>(*scales);
    OutT bias = static_cast<OutT>(*biases);
    for (int i = 0; i < n_reads; i++) {
      dequantize<OutT, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = OutT(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = OutT(0);
      }
      return;
    }

    OutT scale = static_cast<OutT>(*scales);
    OutT bias = static_cast<OutT>(*biases);
    for (int i = 0; i < n_reads; i++) {
      dequantize<OutT, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <
    typename InT,
    typename ParamT,
    typename OutT,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void mixed_qmm_t_impl(
    const device uint32_t* w,
    const device ParamT* scales,
    const device ParamT* biases,
    const device InT* x,
    device OutT* y,
    threadgroup float* Xs,
    threadgroup float* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  constexpr int BK_padded = (BK + 16 / sizeof(float));
  constexpr int bytes_per_pack = (bits == 3 || bits == 6) ? 3 : 1;

  using mma_t = mlx::steel::
      BlockMMA<float, OutT, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      PsiDecMixedBlockLoader<InT, float, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = PsiDecMixedQuantizedBlockLoader<
      ParamT,
      float,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * K;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * N + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <
    typename InT,
    typename ParamT,
    typename OutT,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void mixed_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device ParamT* scales [[buffer(1)]],
    const device ParamT* biases [[buffer(2)]],
    const device InT* x [[buffer(3)]],
    device OutT* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)x_batch_ndims;
  (void)x_shape;
  (void)x_strides;
  (void)w_batch_ndims;
  (void)w_shape;
  (void)w_strides;
  (void)s_strides;
  (void)b_strides;

  constexpr int BK_padded = (BK + 16 / sizeof(float));

  threadgroup float Xs[BM * BK_padded];
  threadgroup float Ws[BN * BK_padded];

  mixed_qmm_t_impl<InT, ParamT, OutT, group_size, bits, aligned_N, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}
"#;

const FUSED_GATE_UP_SILU_SOURCE: &str = r#"
template <typename T, const int group_size, const int bits, const bool aligned_N, const int BM = 32, const int BK = 32, const int BN = 32>
[[kernel]] void qmm_t_fused_gate_up_silu(
    const device uint32_t* w_gate [[buffer(0)]],
    const device T* scales_gate [[buffer(1)]],
    const device T* biases_gate [[buffer(2)]],
    const device uint32_t* w_up [[buffer(3)]],
    const device T* scales_up [[buffer(4)]],
    const device T* biases_up [[buffer(5)]],
    const device T* x [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    const constant int& M [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int bytes_per_pack = (bits == 3 || bits == 6) ? 3 : 1;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  const device uint8_t* gate_wl = (const device uint8_t*)w_gate + y_col * K_w;
  const device uint8_t* up_wl = (const device uint8_t*)w_up + y_col * K_w;
  scales_gate += y_col * K_g;
  biases_gate += y_col * K_g;
  scales_up += y_col * K_g;
  biases_up += y_col * K_g;
  x += y_row * static_cast<int64_t>(K);
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_gate(gate_wl, scales_gate, biases_gate, K, Ws, simd_gid, simd_lid);
  loader_w_t loader_up(up_wl, scales_up, biases_up, K, Ws, simd_gid, simd_lid);
  mma_t mma_gate(simd_gid, simd_lid);
  mma_t mma_up(simd_gid, simd_lid);

  const bool x_safe = num_els < BM;
  const bool w_safe = !aligned_N && num_outs < BN;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (x_safe) {
      loader_x.load_safe(short2(BK, num_els));
    } else {
      loader_x.load_unsafe();
    }
    if (w_safe) {
      loader_gate.load_safe(short2(BK, num_outs));
    } else {
      loader_gate.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_gate.mma(Xs, Ws);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w_safe) {
      loader_up.load_safe(short2(BK, num_outs));
    } else {
      loader_up.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_up.mma(Xs, Ws);

    loader_x.next();
    loader_gate.next();
    loader_up.next();
  }

  for (short i = 0; i < decltype(mma_gate.Ctile)::kElemsPerTile; ++i) {
    T gate = static_cast<T>(mma_gate.Ctile.elems()[i]);
    T up = static_cast<T>(mma_up.Ctile.elems()[i]);
    T sigmoid = static_cast<T>(1.0f / (1.0f + exp(-static_cast<float>(gate))));
    T silu = static_cast<T>(static_cast<float>(gate) * static_cast<float>(sigmoid));
    mma_gate.Ctile.elems()[i] = static_cast<T>(static_cast<float>(silu) * static_cast<float>(up));
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_gate.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_gate.store_result(y, N);
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint column_group,
    uint simd_gid,
    uint simd_lid) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = column_group * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;
        result[row] += qdot<U, values_per_thread, bits>(
            wl, x_thread, sl[0], bl[0], sum);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, sl[0], bl[0], sum, remaining);
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    if (out_row + row < out_vec_size) {
      U value = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(value);
      }
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_gate_up_silu(
    const device uint32_t* gate_w,
    const device T* gate_scales,
    const device T* gate_biases,
    const device uint32_t* up_w,
    const device T* up_scales,
    const device T* up_biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint column_group,
    uint simd_gid,
    uint simd_lid) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* gate_ws = (const device uint8_t*)gate_w;
  const device uint8_t* up_ws = (const device uint8_t*)up_w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U gate_result[results_per_simdgroup] = {0};
  thread U up_result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = column_group * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  gate_ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  up_ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  gate_scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  gate_biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  up_scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  up_biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* gate_wl = gate_ws + row * in_vec_size_w;
        const device T* gate_sl = gate_scales + row * in_vec_size_g;
        const device T* gate_bl = gate_biases + row * in_vec_size_g;
        const device uint8_t* up_wl = up_ws + row * in_vec_size_w;
        const device T* up_sl = up_scales + row * in_vec_size_g;
        const device T* up_bl = up_biases + row * in_vec_size_g;
        gate_result[row] += qdot<U, values_per_thread, bits>(
            gate_wl, x_thread, gate_sl[0], gate_bl[0], sum);
        up_result[row] += qdot<U, values_per_thread, bits>(
            up_wl, x_thread, up_sl[0], up_bl[0], sum);
      }
    }

    gate_ws += block_size * bytes_per_pack / pack_factor;
    up_ws += block_size * bytes_per_pack / pack_factor;
    gate_scales += block_size / group_size;
    gate_biases += block_size / group_size;
    up_scales += block_size / group_size;
    up_biases += block_size / group_size;
    x += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* gate_wl = gate_ws + row * in_vec_size_w;
        const device T* gate_sl = gate_scales + row * in_vec_size_g;
        const device T* gate_bl = gate_biases + row * in_vec_size_g;
        const device uint8_t* up_wl = up_ws + row * in_vec_size_w;
        const device T* up_sl = up_scales + row * in_vec_size_g;
        const device T* up_bl = up_biases + row * in_vec_size_g;
        gate_result[row] += qdot_safe<U, values_per_thread, bits>(
            gate_wl, x_thread, gate_sl[0], gate_bl[0], sum, remaining);
        up_result[row] += qdot_safe<U, values_per_thread, bits>(
            up_wl, x_thread, up_sl[0], up_bl[0], sum, remaining);
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    if (out_row + row < out_vec_size) {
      U gate = simd_sum(gate_result[row]);
      U up = simd_sum(up_result[row]);
      if (simd_lid == 0) {
        T gate_t = static_cast<T>(gate);
        T up_t = static_cast<T>(up);
        T sigmoid = static_cast<T>(1.0f / (1.0f + exp(-static_cast<float>(gate_t))));
        T silu = static_cast<T>(static_cast<float>(gate_t) * static_cast<float>(sigmoid));
        y[row] = static_cast<T>(static_cast<float>(silu) * static_cast<float>(up_t));
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void dense_fused_gate_up_silu(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int in_vec_size_w = in_vec_size * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint32_t* gate_w = w;
  const device uint32_t* up_w = (const device uint32_t*)((const device uint8_t*)w + out_vec_size * in_vec_size_w);
  const device T* gate_scales = scales;
  const device T* gate_biases = biases;
  const device T* up_scales = scales + out_vec_size * in_vec_size_g;
  const device T* up_biases = biases + out_vec_size * in_vec_size_g;
  qmv_gate_up_silu<T, group_size, bits>(
      gate_w,
      gate_scales,
      gate_biases,
      up_w,
      up_scales,
      up_biases,
      x + tid.x * in_vec_size,
      y + tid.x * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void split_fused_gate_up_silu(
    const device uint32_t* gate_w [[buffer(0)]],
    const device T* gate_scales [[buffer(1)]],
    const device T* gate_biases [[buffer(2)]],
    const device uint32_t* up_w [[buffer(3)]],
    const device T* up_scales [[buffer(4)]],
    const device T* up_biases [[buffer(5)]],
    const device T* x [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& in_vec_size [[buffer(8)]],
    const constant int& out_vec_size [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_gate_up_silu<T, group_size, bits>(
      gate_w,
      gate_scales,
      gate_biases,
      up_w,
      up_scales,
      up_biases,
      x + tid.x * in_vec_size,
      y + tid.x * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void token_major_fused_gate_up_silu(
    const device uint32_t* gate_w [[buffer(0)]],
    const device T* gate_scales [[buffer(1)]],
    const device T* gate_biases [[buffer(2)]],
    const device uint32_t* up_w [[buffer(3)]],
    const device T* up_scales [[buffer(4)]],
    const device T* up_biases [[buffer(5)]],
    const device T* x [[buffer(6)]],
    const device uint32_t* lhs_indices [[buffer(7)]],
    const device uint32_t* rhs_indices [[buffer(8)]],
    device T* y [[buffer(9)]],
    const constant int& in_vec_size [[buffer(10)]],
    const constant int& out_vec_size [[buffer(11)]],
    const constant int& num_experts [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.z;
  const uint input_row = lhs_indices[route];
  const uint expert = rhs_indices[route];
  if (expert >= uint(num_experts)) {
    return;
  }
  const int in_vec_size_w = in_vec_size * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const int expert_weight_stride = out_vec_size * in_vec_size_w;
  const int expert_affine_stride = out_vec_size * in_vec_size_g;
  qmv_gate_up_silu<T, group_size, bits>(
      (const device uint32_t*)((const device uint8_t*)gate_w + expert * expert_weight_stride),
      gate_scales + expert * expert_affine_stride,
      gate_biases + expert * expert_affine_stride,
      (const device uint32_t*)((const device uint8_t*)up_w + expert * expert_weight_stride),
      up_scales + expert * expert_affine_stride,
      up_biases + expert * expert_affine_stride,
      x + input_row * in_vec_size,
      y + route * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void expert_major_down_matmul(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* experts_by_route [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant int& num_experts [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.x;
  const uint expert = experts_by_route[route];
  if (expert >= uint(num_experts)) {
    return;
  }
  const int in_vec_size_w = in_vec_size * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const int expert_weight_stride = out_vec_size * in_vec_size_w;
  const int expert_affine_stride = out_vec_size * in_vec_size_g;

  qmv_impl<T, group_size, bits>(
      (const device uint32_t*)((const device uint8_t*)w + expert * expert_weight_stride),
      scales + expert * expert_affine_stride,
      biases + expert * expert_affine_stride,
      x + route * in_vec_size,
      y + route * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void expert_major_fused_gate_up_silu(
    const device uint32_t* gate_w [[buffer(0)]],
    const device T* gate_scales [[buffer(1)]],
    const device T* gate_biases [[buffer(2)]],
    const device uint32_t* up_w [[buffer(3)]],
    const device T* up_scales [[buffer(4)]],
    const device T* up_biases [[buffer(5)]],
    const device T* x [[buffer(6)]],
    const device uint32_t* experts_by_route [[buffer(7)]],
    device T* y [[buffer(8)]],
    const constant int& in_vec_size [[buffer(9)]],
    const constant int& out_vec_size [[buffer(10)]],
    const constant int& num_experts [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.x;
  const uint expert = experts_by_route[route];
  if (expert >= uint(num_experts)) {
    return;
  }
  const int in_vec_size_w = in_vec_size * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const int expert_weight_stride = out_vec_size * in_vec_size_w;
  const int expert_affine_stride = out_vec_size * in_vec_size_g;

  qmv_gate_up_silu<T, group_size, bits>(
      (const device uint32_t*)((const device uint8_t*)gate_w + expert * expert_weight_stride),
      gate_scales + expert * expert_affine_stride,
      gate_biases + expert * expert_affine_stride,
      (const device uint32_t*)((const device uint8_t*)up_w + expert * expert_weight_stride),
      up_scales + expert * expert_affine_stride,
      up_biases + expert * expert_affine_stride,
      x + route * in_vec_size,
      y + route * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}
"#;

fn mlx_metal_header_root() -> PathBuf {
    find_mlx_metal_header_root(
        "quantized.h",
        has_compatible_quantized_headers,
        "affine quantized matmul",
    )
}

fn has_compatible_quantized_headers(root: &Path) -> bool {
    let quantized = root.join("mlx/backend/metal/kernels/quantized.h");
    let Ok(content) = std::fs::read_to_string(quantized) else {
        return false;
    };
    content.contains("[[kernel]] void qmv_quad(") && content.contains("[[kernel]] void qmm_t(")
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::Stream;

    fn execute_matmul(
        stream: &Stream,
        kernel: &AffineQuantizedMatmulKernel,
        output: &Buffer,
        input: &Buffer,
        weight: &Buffer,
        scales: &Buffer,
        biases: &Buffer,
    ) {
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(output, 0, input, 0, weight, 0, scales, 0, biases, 0));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();
    }

    #[test]
    fn test_qmv_reference() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = AffineQuantizedMatmulShape {
            m: 2,
            n: 4,
            k: 32,
            group_size: 32,
            bits: 8,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Float32,
            affine_dtype: Dtype::Float32,
        };
        let input_f32 = fixture_values(shape.m as usize * shape.k as usize, 0.03125);
        let input_bf16 = input_f32
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let weight = fixture_weight_bytes(shape.n as usize * shape.k as usize);
        let scales = fixture_values(shape.n as usize, 0.015625);
        let biases = fixture_values(shape.n as usize, -0.0078125);
        let input = Buffer::from_slice(&device, &input_bf16);
        let output = Buffer::new_zeroed(&device, shape.output_bytes());
        let weight_buffer = Buffer::from_slice(&device, &weight);
        let scales_buffer = Buffer::from_slice(&device, &scales);
        let biases_buffer = Buffer::from_slice(&device, &biases);

        execute_matmul(
            &stream,
            &AffineQuantizedMatmulKernel::new(&device, shape),
            &output,
            &input,
            &weight_buffer,
            &scales_buffer,
            &biases_buffer,
        );

        let actual = output.read_typed::<f32>(0, shape.m as usize * shape.n as usize);
        let expected = cpu_affine_q8(
            shape,
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
        let shape = AffineQuantizedMatmulShape {
            m: 2,
            n: 8,
            k: 512,
            group_size: 64,
            bits: 8,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Float32,
            affine_dtype: Dtype::Float32,
        };
        let input_f32 = fixture_values(shape.m as usize * shape.k as usize, 0.00390625);
        let input_bf16 = input_f32
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let weight = fixture_weight_bytes(shape.n as usize * shape.k as usize);
        let scales = fixture_values(shape.n as usize * (shape.k / shape.group_size) as usize, 0.001953125);
        let biases = fixture_values(shape.n as usize * (shape.k / shape.group_size) as usize, -0.0009765625);
        let input = Buffer::from_slice(&device, &input_bf16);
        let output = Buffer::new_zeroed(&device, shape.output_bytes());
        let weight_buffer = Buffer::from_slice(&device, &weight);
        let scales_buffer = Buffer::from_slice(&device, &scales);
        let biases_buffer = Buffer::from_slice(&device, &biases);

        execute_matmul(
            &stream,
            &AffineQuantizedMatmulKernel::new(&device, shape),
            &output,
            &input,
            &weight_buffer,
            &scales_buffer,
            &biases_buffer,
        );

        let actual = output.read_typed::<f32>(0, shape.m as usize * shape.n as usize);
        let expected = cpu_affine_q8(
            shape,
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
        let shape = AffineQuantizedMatmulShape {
            m: 18,
            n: 4,
            k: 32,
            group_size: 32,
            bits: 8,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Float32,
            affine_dtype: Dtype::Float32,
        };
        let input_f32 = fixture_values(shape.m as usize * shape.k as usize, 0.03125);
        let input_bf16 = input_f32
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let weight = fixture_weight_bytes(shape.n as usize * shape.k as usize);
        let scales = fixture_values(shape.n as usize, 0.015625);
        let biases = fixture_values(shape.n as usize, -0.0078125);
        let input = Buffer::from_slice(&device, &input_bf16);
        let output = Buffer::new_zeroed(&device, shape.output_bytes());
        let weight_buffer = Buffer::from_slice(&device, &weight);
        let scales_buffer = Buffer::from_slice(&device, &scales);
        let biases_buffer = Buffer::from_slice(&device, &biases);

        execute_matmul(
            &stream,
            &AffineQuantizedMatmulKernel::new(&device, shape),
            &output,
            &input,
            &weight_buffer,
            &scales_buffer,
            &biases_buffer,
        );

        let actual = output.read_typed::<f32>(0, shape.m as usize * shape.n as usize);
        let expected = cpu_affine_q8(
            shape,
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
    fn test_qmv_bf16() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = AffineQuantizedMatmulShape {
            m: 1,
            n: 4,
            k: 32,
            group_size: 32,
            bits: 8,
            input_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let input = fixture_values(shape.m as usize * shape.k as usize, 0.03125);
        let weight = fixture_weight_bytes(shape.n as usize * shape.k as usize);
        let scales_f32 = fixture_values(shape.n as usize, 0.015625);
        let biases_f32 = fixture_values(shape.n as usize, -0.0078125);
        let scales_bf16 = scales_f32
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let biases_bf16 = biases_f32
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let input_buffer = Buffer::from_slice(&device, &input);
        let output = Buffer::new_zeroed(&device, shape.output_bytes());
        let weight_buffer = Buffer::from_slice(&device, &weight);
        let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
        let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

        execute_matmul(
            &stream,
            &AffineQuantizedMatmulKernel::new(&device, shape),
            &output,
            &input_buffer,
            &weight_buffer,
            &scales_buffer,
            &biases_buffer,
        );

        let actual = output
            .read_typed::<u16>(0, shape.m as usize * shape.n as usize)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let expected = cpu_affine_q8(
            shape,
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
        let shape = AffineQuantizedMatmulShape {
            m: 2,
            n: 8,
            k: 512,
            group_size: 64,
            bits: 8,
            input_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let input = fixture_values(shape.m as usize * shape.k as usize, 0.00390625);
        let weight = fixture_weight_bytes(shape.n as usize * shape.k as usize);
        let scales_f32 = fixture_values(shape.n as usize * (shape.k / shape.group_size) as usize, 0.001953125);
        let biases_f32 = fixture_values(shape.n as usize * (shape.k / shape.group_size) as usize, -0.0009765625);
        let scales_bf16 = scales_f32
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let biases_bf16 = biases_f32
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let input_buffer = Buffer::from_slice(&device, &input);
        let output = Buffer::new_zeroed(&device, shape.output_bytes());
        let weight_buffer = Buffer::from_slice(&device, &weight);
        let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
        let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

        execute_matmul(
            &stream,
            &AffineQuantizedMatmulKernel::new(&device, shape),
            &output,
            &input_buffer,
            &weight_buffer,
            &scales_buffer,
            &biases_buffer,
        );

        let actual = output
            .read_typed::<u16>(0, shape.m as usize * shape.n as usize)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let expected = cpu_affine_q8(
            shape,
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
        let shape = AffineQuantizedMatmulShape {
            m: 18,
            n: 4,
            k: 32,
            group_size: 32,
            bits: 8,
            input_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let input = fixture_values(shape.m as usize * shape.k as usize, 0.03125);
        let weight = fixture_weight_bytes(shape.n as usize * shape.k as usize);
        let scales_f32 = fixture_values(shape.n as usize, 0.015625);
        let biases_f32 = fixture_values(shape.n as usize, -0.0078125);
        let scales_bf16 = scales_f32
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let biases_bf16 = biases_f32
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let input_buffer = Buffer::from_slice(&device, &input);
        let output = Buffer::new_zeroed(&device, shape.output_bytes());
        let weight_buffer = Buffer::from_slice(&device, &weight);
        let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
        let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

        execute_matmul(
            &stream,
            &AffineQuantizedMatmulKernel::new(&device, shape),
            &output,
            &input_buffer,
            &weight_buffer,
            &scales_buffer,
            &biases_buffer,
        );

        let actual = output
            .read_typed::<u16>(0, shape.m as usize * shape.n as usize)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let expected = cpu_affine_q8(
            shape,
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

    fn cpu_affine_q8(
        shape: AffineQuantizedMatmulShape,
        input: &[f32],
        weight: &[u8],
        scales: &[f32],
        biases: &[f32],
    ) -> Vec<f32> {
        let m = shape.m as usize;
        let n = shape.n as usize;
        let k = shape.k as usize;
        let mut output = vec![0.0_f32; m * n];
        for row in 0..m {
            let input_row = &input[row * k..(row + 1) * k];
            for col in 0..n {
                let weight_row = &weight[col * k..(col + 1) * k];
                let mut value = 0.0_f32;
                for group in 0..(k / shape.group_size as usize) {
                    let group_start = group * shape.group_size as usize;
                    let group_end = group_start + shape.group_size as usize;
                    let input_group = &input_row[group_start..group_end];
                    let weight_group = &weight_row[group_start..group_end];
                    let input_sum = input_group.iter().copied().sum::<f32>();
                    let dot = input_group
                        .iter()
                        .zip(weight_group)
                        .map(|(x, w)| *x * f32::from(*w))
                        .sum::<f32>();
                    let affine_index = col * (k / shape.group_size as usize) + group;
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

    fn fixture_weight_bytes(len: usize) -> Vec<u8> {
        (0..len).map(|index| ((index * 7 + 3) % 251) as u8).collect()
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
}
