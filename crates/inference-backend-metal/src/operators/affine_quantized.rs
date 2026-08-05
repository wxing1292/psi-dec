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
use crate::metal::ReplayParameterKey;
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

fn checked_range_end(name: &str, offset_bytes: usize, required_bytes: usize) -> usize {
    offset_bytes
        .checked_add(required_bytes)
        .unwrap_or_else(|| panic!("{name} byte range must fit usize"))
}

#[derive(Clone, Copy, Debug)]
pub struct AffineQuantizedMatmulConfig {
    pub n: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
    pub scale_bias_dtype: Dtype,
}

impl AffineQuantizedMatmulConfig {
    pub fn same_dtype(n: i32, k: i32, group_size: i32, bits: i32, dtype: Dtype) -> Self {
        Self {
            n,
            k,
            group_size,
            bits,
            input_dtype: dtype,
            output_dtype: dtype,
            scale_bias_dtype: dtype,
        }
    }

    pub fn validate(self) {
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
            self.scale_bias_dtype,
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ));
    }

    pub fn output_bytes(self, m: i32) -> usize {
        self.validate();
        assert!(m > 0);
        checked_bytes(
            "affine matmul output",
            &[m as usize, self.n as usize],
            self.output_dtype,
        )
    }

    pub fn input_bytes(self, m: i32) -> usize {
        self.validate();
        assert!(m > 0);
        checked_bytes("affine matmul input", &[m as usize, self.k as usize], self.input_dtype)
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

    pub fn scale_or_bias_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "affine matmul scale or bias",
            &[self.n as usize, (self.k / self.group_size) as usize],
            self.scale_bias_dtype,
        )
    }

    fn uses_same_dtype(self) -> bool {
        self.input_dtype == self.output_dtype && self.input_dtype == self.scale_bias_dtype
    }
}

pub struct AffineQuantizedMatmulKernel {
    config: AffineQuantizedMatmulConfig,
    kind: AffineQuantizedMatmulKernelKind,
    kernel: Kernel,
}

pub struct AffineQuantizedMatmul {
    config: AffineQuantizedMatmulConfig,
    qmv: AffineQuantizedMatmulKernel,
    qmm_bm8_bn32: AffineQuantizedMatmulKernel,
    qmm_bm16_bn32: AffineQuantizedMatmulKernel,
    qmm_bm32_bn32: AffineQuantizedMatmulKernel,
}

#[derive(Clone, Copy, Debug)]
pub struct ExpertAffineQuantizedConfig {
    pub num_experts: i32,
    pub matmul: AffineQuantizedMatmulConfig,
}

impl ExpertAffineQuantizedConfig {
    pub fn validate(self) {
        assert!(self.num_experts > 0);
        self.matmul.validate();
        // TODO: Generalize the gather and ragged expert templates to separate
        // input, output, and scale/bias element types.
        assert!(
            self.matmul.uses_same_dtype(),
            "expert affine quantized kernels do not yet support mixed dtypes"
        );
    }

    pub fn output_bytes(self, num_vectors: i32) -> usize {
        self.validate();
        self.matmul.output_bytes(num_vectors)
    }

    pub fn input_bytes(self, num_vectors: i32) -> usize {
        self.validate();
        self.matmul.input_bytes(num_vectors)
    }

    pub fn weight_bytes_per_expert(self) -> usize {
        self.validate();
        self.matmul.weight_bytes()
    }

    pub fn affine_param_bytes_per_expert(self) -> usize {
        self.validate();
        self.matmul.scale_or_bias_bytes()
    }

    fn weight_bytes(self) -> usize {
        checked_product(
            "expert affine weight byte length",
            &[self.num_experts as usize, self.weight_bytes_per_expert()],
        )
    }

    fn affine_param_bytes(self) -> usize {
        checked_product(
            "expert affine parameter byte length",
            &[self.num_experts as usize, self.affine_param_bytes_per_expert()],
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GatherAffineQuantizedShape {
    pub num_routes: i32,
    pub num_input_vectors: i32,
}

impl GatherAffineQuantizedShape {
    pub fn validate(self) {
        assert!(self.num_routes > 0);
        assert!(self.num_input_vectors > 0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RaggedExpertMajorAffineQuantizedShape {
    pub num_routes: i32,
}

impl RaggedExpertMajorAffineQuantizedShape {
    pub fn validate(self) {
        assert!(self.num_routes > 0);
    }
}

#[derive(Clone, Copy)]
struct ExpertAffineBucketedRoutes {
    num_total_tokens: u32,
    num_experts_per_token: u32,
    num_active_tokens_key: ReplayParameterKey,
}

impl ExpertAffineBucketedRoutes {
    fn new(
        config: ExpertAffineQuantizedConfig,
        num_total_routes: i32,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens_key: ReplayParameterKey,
    ) -> Self {
        config.validate();
        assert!(num_total_tokens > 0);
        assert!(num_experts_per_token > 0);
        assert!(
            num_experts_per_token <= config.num_experts as u32,
            "expert affine routes per token must not exceed expert count"
        );
        let derived_num_total_routes = num_total_tokens
            .checked_mul(num_experts_per_token)
            .expect("expert affine total route count must fit u32");
        assert_eq!(
            u32::try_from(num_total_routes).expect("expert affine total route count must be positive"),
            derived_num_total_routes,
            "expert affine total routes must equal total tokens times experts per token"
        );
        i32::try_from(derived_num_total_routes).expect("expert affine total route count must fit i32");
        Self {
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        }
    }

    fn bind(self, builder: &CommandRecorder<'_>, token_binding_index: usize, topk_binding_index: usize) {
        builder.bind_u32(
            token_binding_index,
            self.num_active_tokens_key,
            1,
            self.num_total_tokens,
        );
        builder.set_u32(topk_binding_index, self.num_experts_per_token);
    }
}

pub struct GatherAffineQuantizedMatmulKernel {
    config: ExpertAffineQuantizedConfig,
    kernel: Kernel,
    bucketed_kernel: Kernel,
}

pub struct GatherAffineQuantizedGateUpSwiGLUKernel {
    config: ExpertAffineQuantizedConfig,
    kernel: Kernel,
    bucketed_kernel: Kernel,
}

pub struct RaggedExpertMajorAffineQuantizedGateUpSwiGLUKernel {
    config: ExpertAffineQuantizedConfig,
    kernel: Kernel,
    bucketed_kernel: Kernel,
}

pub struct RaggedExpertMajorAffineQuantizedMatmulKernel {
    config: ExpertAffineQuantizedConfig,
    kernel: Kernel,
    bucketed_kernel: Kernel,
}

impl GatherAffineQuantizedMatmulKernel {
    pub fn new(device: &Device, config: ExpertAffineQuantizedConfig) -> Self {
        config.validate();
        let matmul = config.matmul;
        let type_string = metal_type_string(matmul.input_dtype);
        let bn = 8;
        let fast = matmul.n % bn == 0 && matmul.k % 512 == 0;
        let func = if fast { "gather_qmv_fast" } else { "gather_qmv" };
        let kernel_name = format!("{func}_{type_string}_gs_{}_b_{}", matmul.group_size, matmul.bits);
        let exact_template_definition = template_definition(
            &kernel_name,
            func,
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&exact_template_definition);
        let kernel = Kernel::new(device, &source, &kernel_name);
        let bucketed_kernel_name = format!(
            "token_major_down_matmul_bucketed_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let bucketed_template_definition = template_definition(
            &bucketed_kernel_name,
            "token_major_down_matmul_bucketed",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
                fast.to_string(),
            ],
        );
        let bucketed_source =
            affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{bucketed_template_definition}"));
        let bucketed_kernel = Kernel::new(device, &bucketed_source, &bucketed_kernel_name);
        Self {
            config,
            kernel,
            bucketed_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: GatherAffineQuantizedShape,
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
            bucketed_routes: None,
        }
    }

    /// Records a fixed route-capacity grid whose active route count derives from active tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_bucketed<'a>(
        &'a self,
        shape: GatherAffineQuantizedShape,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens_key: ReplayParameterKey,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        lhs_indices: &'a Buffer,
        rhs_indices: &'a Buffer,
    ) -> GatherAffineQuantizedMatmulInvocation<'a> {
        let bucketed_routes = ExpertAffineBucketedRoutes::new(
            self.config,
            shape.num_routes,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
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
            bucketed_routes: Some(bucketed_routes),
        }
    }
}

impl GatherAffineQuantizedGateUpSwiGLUKernel {
    pub fn new(device: &Device, config: ExpertAffineQuantizedConfig) -> Self {
        config.validate();
        let matmul = config.matmul;
        let type_string = metal_type_string(matmul.input_dtype);
        let kernel_name = format!(
            "token_major_gate_up_swiglu_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let exact_template_definition = template_definition(
            &kernel_name,
            "token_major_gate_up_swiglu",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{exact_template_definition}"));
        let kernel = Kernel::new(device, &source, &kernel_name);
        let bucketed_kernel_name = format!(
            "token_major_gate_up_swiglu_bucketed_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let bucketed_template_definition = template_definition(
            &bucketed_kernel_name,
            "token_major_gate_up_swiglu_bucketed",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let bucketed_source =
            affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{bucketed_template_definition}"));
        let bucketed_kernel = Kernel::new(device, &bucketed_source, &bucketed_kernel_name);
        Self {
            config,
            kernel,
            bucketed_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: GatherAffineQuantizedShape,
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
    ) -> GatherAffineQuantizedGateUpSwiGLUInvocation<'a> {
        GatherAffineQuantizedGateUpSwiGLUInvocation {
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
            bucketed_routes: None,
        }
    }

    /// Records a fixed route-capacity grid whose active route count derives from active tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_bucketed<'a>(
        &'a self,
        shape: GatherAffineQuantizedShape,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens_key: ReplayParameterKey,
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
    ) -> GatherAffineQuantizedGateUpSwiGLUInvocation<'a> {
        let bucketed_routes = ExpertAffineBucketedRoutes::new(
            self.config,
            shape.num_routes,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
        GatherAffineQuantizedGateUpSwiGLUInvocation {
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
            bucketed_routes: Some(bucketed_routes),
        }
    }
}

impl RaggedExpertMajorAffineQuantizedGateUpSwiGLUKernel {
    pub fn new(device: &Device, config: ExpertAffineQuantizedConfig) -> Self {
        config.validate();
        let matmul = config.matmul;
        let type_string = metal_type_string(matmul.input_dtype);
        let kernel_name = format!(
            "expert_major_gate_up_swiglu_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let exact_template_definition = template_definition(
            &kernel_name,
            "expert_major_gate_up_swiglu",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{exact_template_definition}"));
        let kernel = Kernel::new(device, &source, &kernel_name);
        let bucketed_kernel_name = format!(
            "expert_major_gate_up_swiglu_bucketed_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let bucketed_template_definition = template_definition(
            &bucketed_kernel_name,
            "expert_major_gate_up_swiglu_bucketed",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let bucketed_source =
            affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{bucketed_template_definition}"));
        let bucketed_kernel = Kernel::new(device, &bucketed_source, &bucketed_kernel_name);
        Self {
            config,
            kernel,
            bucketed_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: RaggedExpertMajorAffineQuantizedShape,
        output: &'a Buffer,
        input: &'a Buffer,
        gate_weight: &'a Buffer,
        gate_scales: &'a Buffer,
        gate_biases: &'a Buffer,
        up_weight: &'a Buffer,
        up_scales: &'a Buffer,
        up_biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorAffineQuantizedGateUpSwiGLUInvocation<'a> {
        RaggedExpertMajorAffineQuantizedGateUpSwiGLUInvocation {
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
            bucketed_routes: None,
        }
    }

    /// Records a fixed route-capacity grid whose active route count derives from active tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_bucketed<'a>(
        &'a self,
        shape: RaggedExpertMajorAffineQuantizedShape,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens_key: ReplayParameterKey,
        output: &'a Buffer,
        input: &'a Buffer,
        gate_weight: &'a Buffer,
        gate_scales: &'a Buffer,
        gate_biases: &'a Buffer,
        up_weight: &'a Buffer,
        up_scales: &'a Buffer,
        up_biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorAffineQuantizedGateUpSwiGLUInvocation<'a> {
        let bucketed_routes = ExpertAffineBucketedRoutes::new(
            self.config,
            shape.num_routes,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
        RaggedExpertMajorAffineQuantizedGateUpSwiGLUInvocation {
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
            bucketed_routes: Some(bucketed_routes),
        }
    }
}

impl RaggedExpertMajorAffineQuantizedMatmulKernel {
    pub fn new(device: &Device, config: ExpertAffineQuantizedConfig) -> Self {
        config.validate();
        let matmul = config.matmul;
        let type_string = metal_type_string(matmul.input_dtype);
        let kernel_name = format!(
            "expert_major_down_matmul_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let exact_template_definition = template_definition(
            &kernel_name,
            "expert_major_down_matmul",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{exact_template_definition}"));
        let kernel = Kernel::new(device, &source, &kernel_name);
        let bucketed_kernel_name = format!(
            "expert_major_down_matmul_bucketed_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let bucketed_template_definition = template_definition(
            &bucketed_kernel_name,
            "expert_major_down_matmul_bucketed",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let bucketed_source =
            affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{bucketed_template_definition}"));
        let bucketed_kernel = Kernel::new(device, &bucketed_source, &bucketed_kernel_name);
        Self {
            config,
            kernel,
            bucketed_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: RaggedExpertMajorAffineQuantizedShape,
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
            bucketed_routes: None,
        }
    }

    /// Records a fixed route-capacity grid whose active route count derives from active tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_bucketed<'a>(
        &'a self,
        shape: RaggedExpertMajorAffineQuantizedShape,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens_key: ReplayParameterKey,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorAffineQuantizedMatmulInvocation<'a> {
        let bucketed_routes = ExpertAffineBucketedRoutes::new(
            self.config,
            shape.num_routes,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
        RaggedExpertMajorAffineQuantizedMatmulInvocation {
            kernel: self,
            shape,
            output,
            input,
            weight,
            scales,
            biases,
            experts_by_route,
            bucketed_routes: Some(bucketed_routes),
        }
    }
}

pub struct GatherAffineQuantizedMatmulInvocation<'a> {
    kernel: &'a GatherAffineQuantizedMatmulKernel,
    shape: GatherAffineQuantizedShape,
    output: &'a Buffer,
    input: &'a Buffer,
    weight: &'a Buffer,
    scales: &'a Buffer,
    biases: &'a Buffer,
    lhs_indices: &'a Buffer,
    rhs_indices: &'a Buffer,
    bucketed_routes: Option<ExpertAffineBucketedRoutes>,
}

pub struct GatherAffineQuantizedGateUpSwiGLUInvocation<'a> {
    kernel: &'a GatherAffineQuantizedGateUpSwiGLUKernel,
    shape: GatherAffineQuantizedShape,
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
    bucketed_routes: Option<ExpertAffineBucketedRoutes>,
}

pub struct RaggedExpertMajorAffineQuantizedGateUpSwiGLUInvocation<'a> {
    kernel: &'a RaggedExpertMajorAffineQuantizedGateUpSwiGLUKernel,
    shape: RaggedExpertMajorAffineQuantizedShape,
    output: &'a Buffer,
    input: &'a Buffer,
    gate_weight: &'a Buffer,
    gate_scales: &'a Buffer,
    gate_biases: &'a Buffer,
    up_weight: &'a Buffer,
    up_scales: &'a Buffer,
    up_biases: &'a Buffer,
    experts_by_route: &'a Buffer,
    bucketed_routes: Option<ExpertAffineBucketedRoutes>,
}

pub struct RaggedExpertMajorAffineQuantizedMatmulInvocation<'a> {
    kernel: &'a RaggedExpertMajorAffineQuantizedMatmulKernel,
    shape: RaggedExpertMajorAffineQuantizedShape,
    output: &'a Buffer,
    input: &'a Buffer,
    weight: &'a Buffer,
    scales: &'a Buffer,
    biases: &'a Buffer,
    experts_by_route: &'a Buffer,
    bucketed_routes: Option<ExpertAffineBucketedRoutes>,
}

impl Operator for RaggedExpertMajorAffineQuantizedMatmulInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        let matmul = config.matmul;
        let shape = self.shape;
        validate_ragged_expert_major_down_matmul_buffer_ranges(
            config,
            shape,
            self.output,
            self.input,
            self.weight,
            self.scales,
            self.biases,
            self.experts_by_route,
        );

        let kernel = match self.bucketed_routes {
            Some(_) => &self.kernel.bucketed_kernel,
            None => &self.kernel.kernel,
        };
        builder.set_kernel(kernel);
        builder.set_buffer_read(0, self.weight, 0);
        builder.set_buffer_read(1, self.scales, 0);
        builder.set_buffer_read(2, self.biases, 0);
        builder.set_buffer_read(3, self.input, 0);
        builder.set_buffer_read(4, self.experts_by_route, 0);
        builder.set_buffer_write(5, self.output, 0);
        builder.set_i32(6, matmul.k);
        builder.set_i32(7, matmul.n);
        builder.set_i32(8, config.num_experts);
        if let Some(bucketed_routes) = self.bucketed_routes {
            bucketed_routes.bind(builder, 9, 10);
        }
        builder.dispatch_threadblocks(
            (shape.num_routes as usize, ceil_div_i32(matmul.n, 8) as usize, 1),
            (32, 2, 1),
        );
    }
}

impl Operator for RaggedExpertMajorAffineQuantizedGateUpSwiGLUInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        let matmul = config.matmul;
        let shape = self.shape;
        validate_ragged_expert_major_gate_up_swiglu_buffer_ranges(
            config,
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

        let kernel = match self.bucketed_routes {
            Some(_) => &self.kernel.bucketed_kernel,
            None => &self.kernel.kernel,
        };
        builder.set_kernel(kernel);
        builder.set_buffer_read(0, self.gate_weight, 0);
        builder.set_buffer_read(1, self.gate_scales, 0);
        builder.set_buffer_read(2, self.gate_biases, 0);
        builder.set_buffer_read(3, self.up_weight, 0);
        builder.set_buffer_read(4, self.up_scales, 0);
        builder.set_buffer_read(5, self.up_biases, 0);
        builder.set_buffer_read(6, self.input, 0);
        builder.set_buffer_read(7, self.experts_by_route, 0);
        builder.set_buffer_write(8, self.output, 0);
        builder.set_i32(9, matmul.k);
        builder.set_i32(10, matmul.n);
        builder.set_i32(11, config.num_experts);
        if let Some(bucketed_routes) = self.bucketed_routes {
            bucketed_routes.bind(builder, 12, 13);
        }
        builder.dispatch_threadblocks(
            (shape.num_routes as usize, ceil_div_i32(matmul.n, 8) as usize, 1),
            (32, 2, 1),
        );
    }
}

impl Operator for GatherAffineQuantizedGateUpSwiGLUInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        let matmul = config.matmul;
        let shape = self.shape;
        validate_gather_gate_up_swiglu_buffer_ranges(
            config,
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
        let kernel = match self.bucketed_routes {
            Some(_) => &self.kernel.bucketed_kernel,
            None => &self.kernel.kernel,
        };
        builder.set_kernel(kernel);
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
        builder.set_i32(10, matmul.k);
        builder.set_i32(11, matmul.n);
        builder.set_i32(12, config.num_experts);
        if let Some(bucketed_routes) = self.bucketed_routes {
            bucketed_routes.bind(builder, 13, 14);
        }
        builder.dispatch_threadblocks(
            (1, ceil_div_i32(matmul.n, 8) as usize, shape.num_routes as usize),
            (32, 2, 1),
        );
    }
}

impl Operator for GatherAffineQuantizedMatmulInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        let matmul = config.matmul;
        let shape = self.shape;
        validate_gather_buffer_ranges(
            config,
            shape,
            self.output,
            self.input,
            self.weight,
            self.scales,
            self.biases,
            self.lhs_indices,
            self.rhs_indices,
        );
        let packed_k = packed_dim(matmul.k, matmul.bits);
        let groups = matmul.k / matmul.group_size;
        let kernel = match self.bucketed_routes {
            Some(_) => &self.kernel.bucketed_kernel,
            None => &self.kernel.kernel,
        };
        builder.set_kernel(kernel);
        builder.set_buffer_read(0, self.weight, 0);
        builder.set_buffer_read(1, self.scales, 0);
        builder.set_buffer_read(2, self.biases, 0);
        builder.set_buffer_read(3, self.input, 0);
        builder.set_buffer_read(4, self.lhs_indices, 0);
        builder.set_buffer_read(5, self.rhs_indices, 0);
        builder.set_buffer_write(6, self.output, 0);
        builder.set_i32(7, matmul.k);
        builder.set_i32(8, matmul.n);

        let x_shape = [shape.num_input_vectors, 1, 1, matmul.k];
        let k_stride = i64::from(matmul.k);
        let x_strides = [k_stride, k_stride, k_stride, 1_i64];
        let w_shape = [config.num_experts, matmul.n, packed_k];
        let w_expert_stride = i64::from(matmul.n)
            .checked_mul(i64::from(packed_k))
            .expect("gather affine weight stride must fit i64");
        let affine_expert_stride = i64::from(matmul.n)
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
        if let Some(bucketed_routes) = self.bucketed_routes {
            bucketed_routes.bind(builder, 21, 22);
        }

        let bn = 8;
        let bk = 32;
        builder.dispatch_threadblocks(
            (1, ceil_div_i32(matmul.n, bn) as usize, shape.num_routes as usize),
            (bk as usize, 2, 1),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_gather_buffer_ranges(
    config: ExpertAffineQuantizedConfig,
    shape: GatherAffineQuantizedShape,
    output: &Buffer,
    input: &Buffer,
    weight: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    lhs_indices: &Buffer,
    rhs_indices: &Buffer,
) {
    config.validate();
    shape.validate();
    let output_bytes = config.output_bytes(shape.num_routes);
    let input_bytes = config.input_bytes(shape.num_input_vectors);
    let weight_bytes = config.weight_bytes();
    let affine_param_bytes = config.affine_param_bytes();
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
        weight.len_bytes() >= weight_bytes,
        "gather affine quantized matmul weight stack too short: config={config:?} shape={shape:?} \
         required_bytes={weight_bytes} buffer_bytes={}",
        weight.len_bytes()
    );
    assert!(
        affine_param_bytes <= scales.len_bytes(),
        "gather affine quantized matmul scales stack too short: config={config:?} shape={shape:?} \
         required_bytes={affine_param_bytes} buffer_bytes={}",
        scales.len_bytes()
    );
    assert!(
        affine_param_bytes <= biases.len_bytes(),
        "gather affine quantized matmul biases stack too short: config={config:?} shape={shape:?} \
         required_bytes={affine_param_bytes} buffer_bytes={}",
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
fn validate_gather_gate_up_swiglu_buffer_ranges(
    config: ExpertAffineQuantizedConfig,
    shape: GatherAffineQuantizedShape,
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
    config.validate();
    shape.validate();
    let output_bytes = config.output_bytes(shape.num_routes);
    let input_bytes = config.input_bytes(shape.num_input_vectors);
    let weight_bytes = config.weight_bytes();
    let affine_param_bytes = config.affine_param_bytes();
    assert!(
        output_bytes <= output.len_bytes(),
        "gather affine quantized gate/up/swiglu output range out of bounds: shape={shape:?} \
         required_bytes={output_bytes} buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "gather affine quantized gate/up/swiglu input range out of bounds: shape={shape:?} \
         required_bytes={input_bytes} buffer_bytes={}",
        input.len_bytes()
    );
    assert!(
        gate_weight.len_bytes() >= weight_bytes,
        "gather affine quantized gate weight stack too short: config={config:?} shape={shape:?} \
         required_bytes={weight_bytes} buffer_bytes={}",
        gate_weight.len_bytes()
    );
    assert!(weight_bytes <= up_weight.len_bytes());
    assert!(affine_param_bytes <= gate_scales.len_bytes());
    assert!(affine_param_bytes <= gate_biases.len_bytes());
    assert!(affine_param_bytes <= up_scales.len_bytes());
    assert!(affine_param_bytes <= up_biases.len_bytes());
    let index_bytes = shape.num_routes as usize * size_of::<u32>();
    assert!(index_bytes <= lhs_indices.len_bytes());
    assert!(index_bytes <= rhs_indices.len_bytes());
}

#[allow(clippy::too_many_arguments)]
fn validate_ragged_expert_major_gate_up_swiglu_buffer_ranges(
    config: ExpertAffineQuantizedConfig,
    shape: RaggedExpertMajorAffineQuantizedShape,
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
    config.validate();
    shape.validate();
    let output_bytes = config.output_bytes(shape.num_routes);
    let input_bytes = config.input_bytes(shape.num_routes);
    let weight_bytes = config.weight_bytes();
    let affine_param_bytes = config.affine_param_bytes();
    let route_index_bytes = checked_product(
        "ragged expert-major gate/up route-index byte length",
        &[shape.num_routes as usize, size_of::<u32>()],
    );
    assert!(
        output_bytes <= output.len_bytes(),
        "ragged expert-major gate/up/swiglu output range out of bounds: shape={shape:?} required_bytes={output_bytes} \
         buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "ragged expert-major gate/up/swiglu input range out of bounds: shape={shape:?} required_bytes={input_bytes} \
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

#[allow(clippy::too_many_arguments)]
fn validate_ragged_expert_major_down_matmul_buffer_ranges(
    config: ExpertAffineQuantizedConfig,
    shape: RaggedExpertMajorAffineQuantizedShape,
    output: &Buffer,
    input: &Buffer,
    weight: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    experts_by_route: &Buffer,
) {
    config.validate();
    shape.validate();
    let output_bytes = config.output_bytes(shape.num_routes);
    let input_bytes = config.input_bytes(shape.num_routes);
    let weight_bytes = config.weight_bytes();
    let affine_param_bytes = config.affine_param_bytes();
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

fn packed_dim(k: i32, bits: i32) -> i32 {
    assert!(k > 0);
    assert!(matches!(bits, 2 | 3 | 4 | 6 | 8));
    let total_bits = k.checked_mul(bits).expect("packed affine dimension must fit i32");
    assert_eq!(total_bits % 32, 0);
    total_bits / 32
}

/// Stable identity for the kernel topology recorded by one affine invocation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AffineQuantizedMatmulKernelKind {
    QmvBn8Bk32,
    QmvQuadBn64,
    QmmBm8Bn32,
    QmmBm16Bn32,
    QmmBm32Bn32,
}

impl AffineQuantizedMatmulKernel {
    pub fn new(device: &Device, config: AffineQuantizedMatmulConfig, kind: AffineQuantizedMatmulKernelKind) -> Self {
        config.validate();
        validate_kernel_kind(config, kind);
        let (kernel_name, source) = affine_kernel_source(config, kind);
        let kernel = Kernel::new(device, &source, &kernel_name);
        if matches!(
            kind,
            AffineQuantizedMatmulKernelKind::QmmBm8Bn32
                | AffineQuantizedMatmulKernelKind::QmmBm16Bn32
                | AffineQuantizedMatmulKernelKind::QmmBm32Bn32
        ) {
            validate_qmm_pipeline(device, config, kind, &kernel);
        }
        Self { config, kind, kernel }
    }

    pub fn kind(&self) -> AffineQuantizedMatmulKernelKind {
        self.kind
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        m: i32,
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
        assert!(m > 0);
        AffineQuantizedMatmulInvocation {
            kernel: self,
            num_total_rows: m as u32,
            num_active_rows_key: None,
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

    /// Records a fixed-capacity grid whose active row count is supplied at submission.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_bucketed<'a>(
        &'a self,
        num_total_rows: u32,
        num_active_rows_key: ReplayParameterKey,
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
        validate_num_total_rows(num_total_rows);
        AffineQuantizedMatmulInvocation {
            kernel: self,
            num_total_rows,
            num_active_rows_key: Some(num_active_rows_key),
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

impl AffineQuantizedMatmul {
    pub fn new(device: &Device, config: AffineQuantizedMatmulConfig) -> Self {
        config.validate();
        let qmv_kind = select_qmv_kernel_kind(config);
        Self {
            config,
            qmv: AffineQuantizedMatmulKernel::new(device, config, qmv_kind),
            qmm_bm8_bn32: AffineQuantizedMatmulKernel::new(device, config, AffineQuantizedMatmulKernelKind::QmmBm8Bn32),
            qmm_bm16_bn32: AffineQuantizedMatmulKernel::new(
                device,
                config,
                AffineQuantizedMatmulKernelKind::QmmBm16Bn32,
            ),
            qmm_bm32_bn32: AffineQuantizedMatmulKernel::new(
                device,
                config,
                AffineQuantizedMatmulKernelKind::QmmBm32Bn32,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        m: i32,
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
        self.selected_kernel(m).invoke(
            m,
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

    /// Records a fixed-capacity grid whose active row count is supplied at submission.
    ///
    /// Kernel selection and dispatch use `num_total_rows`. Callers must prevent a
    /// replay bucket from crossing a value returned by [`Self::topology_boundaries`].
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_bucketed<'a>(
        &'a self,
        num_total_rows: u32,
        num_active_rows_key: ReplayParameterKey,
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
        self.kernel_for_topology(self.topology(num_total_rows)).invoke_bucketed(
            num_total_rows,
            num_active_rows_key,
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

    /// Returns the recorded kernel topology for one total row count.
    pub fn topology(&self, num_total_rows: u32) -> AffineQuantizedMatmulKernelKind {
        validate_num_total_rows(num_total_rows);
        select_kernel_kind(
            self.config,
            i32::try_from(num_total_rows).expect("affine total row count must fit i32"),
        )
    }

    /// Returns the first row count for each change in recorded kernel topology.
    pub fn topology_boundaries(&self) -> Box<[u32]> {
        adaptive_topology_boundaries(self.config)
    }

    pub fn selected_kernel(&self, m: i32) -> &AffineQuantizedMatmulKernel {
        assert!(m > 0);
        self.kernel_for_topology(select_kernel_kind(self.config, m))
    }

    fn kernel_for_topology(&self, topology: AffineQuantizedMatmulKernelKind) -> &AffineQuantizedMatmulKernel {
        match topology {
            AffineQuantizedMatmulKernelKind::QmvBn8Bk32 | AffineQuantizedMatmulKernelKind::QmvQuadBn64 => &self.qmv,
            AffineQuantizedMatmulKernelKind::QmmBm8Bn32 => &self.qmm_bm8_bn32,
            AffineQuantizedMatmulKernelKind::QmmBm16Bn32 => &self.qmm_bm16_bn32,
            AffineQuantizedMatmulKernelKind::QmmBm32Bn32 => &self.qmm_bm32_bn32,
        }
    }
}

pub struct AffineQuantizedMatmulInvocation<'a> {
    kernel: &'a AffineQuantizedMatmulKernel,
    num_total_rows: u32,
    num_active_rows_key: Option<ReplayParameterKey>,
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
        let config = kernel.config;
        let num_total_rows = self.num_total_rows;
        let num_total_rows_i32 = i32::try_from(num_total_rows).expect("affine total row count must fit i32");
        validate_buffer_ranges(
            config,
            num_total_rows_i32,
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
        builder.set_i32(5, config.k);
        builder.set_i32(6, config.n);
        record_num_active_rows(builder, 7, num_total_rows, self.num_active_rows_key);

        match kernel.kind {
            AffineQuantizedMatmulKernelKind::QmmBm8Bn32 => {
                builder.dispatch_threadblocks(
                    (
                        ceil_div_i32(config.n, 32) as usize,
                        ceil_div_i32(num_total_rows_i32, 8) as usize,
                        1,
                    ),
                    (32, 2, 1),
                );
            },
            AffineQuantizedMatmulKernelKind::QmmBm16Bn32 => {
                builder.dispatch_threadblocks(
                    (
                        ceil_div_i32(config.n, 32) as usize,
                        ceil_div_i32(num_total_rows_i32, 16) as usize,
                        1,
                    ),
                    (32, 2, 1),
                );
            },
            AffineQuantizedMatmulKernelKind::QmmBm32Bn32 => {
                builder.dispatch_threadblocks(
                    (
                        ceil_div_i32(config.n, 32) as usize,
                        ceil_div_i32(num_total_rows_i32, 32) as usize,
                        1,
                    ),
                    (32, 2, 2),
                );
            },
            AffineQuantizedMatmulKernelKind::QmvQuadBn64 => {
                builder.dispatch_threadblocks(
                    (num_total_rows as usize, ceil_div_i32(config.n, 64) as usize, 1),
                    (32, 1, 1),
                );
            },
            AffineQuantizedMatmulKernelKind::QmvBn8Bk32 => {
                builder.dispatch_threadblocks(
                    (num_total_rows as usize, ceil_div_i32(config.n, 8) as usize, 1),
                    (32, 2, 1),
                );
            },
        }
    }
}

fn record_num_active_rows(
    builder: &CommandRecorder,
    binding_index: usize,
    num_total_rows: u32,
    key: Option<ReplayParameterKey>,
) {
    match key {
        Some(key) => builder.bind_u32(binding_index, key, 1, num_total_rows),
        None => builder.set_u32(binding_index, num_total_rows),
    }
}

fn validate_num_total_rows(num_total_rows: u32) {
    assert!(num_total_rows > 0, "affine total row count must be positive");
    assert!(
        i32::try_from(num_total_rows).is_ok(),
        "affine total row count must fit i32"
    );
}

#[allow(clippy::too_many_arguments)]
fn validate_buffer_ranges(
    config: AffineQuantizedMatmulConfig,
    m: i32,
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
    config.validate();
    assert!(m > 0);
    let output_bytes = config.output_bytes(m);
    let input_bytes = config.input_bytes(m);
    let weight_bytes = config.weight_bytes();
    let scale_or_bias_bytes = config.scale_or_bias_bytes();
    assert!(
        checked_range_end("affine quantized matmul output", output_offset_bytes, output_bytes) <= output.len_bytes(),
        "affine quantized matmul output range out of bounds: config={config:?} m={m} \
         offset_bytes={output_offset_bytes} required_bytes={output_bytes} buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        checked_range_end("affine quantized matmul input", input_offset_bytes, input_bytes) <= input.len_bytes(),
        "affine quantized matmul input range out of bounds: config={config:?} m={m} offset_bytes={input_offset_bytes} \
         required_bytes={input_bytes} buffer_bytes={}",
        input.len_bytes()
    );
    assert!(
        checked_range_end("affine quantized matmul weight", weight_offset_bytes, weight_bytes) <= weight.len_bytes(),
        "affine quantized matmul weight range out of bounds: config={config:?} offset_bytes={weight_offset_bytes} \
         required_bytes={weight_bytes} buffer_bytes={}",
        weight.len_bytes()
    );
    assert!(
        checked_range_end(
            "affine quantized matmul scales",
            scales_offset_bytes,
            scale_or_bias_bytes,
        ) <= scales.len_bytes(),
        "affine quantized matmul scales range out of bounds: config={config:?} offset_bytes={scales_offset_bytes} \
         required_bytes={scale_or_bias_bytes} buffer_bytes={}",
        scales.len_bytes()
    );
    assert!(
        checked_range_end(
            "affine quantized matmul biases",
            biases_offset_bytes,
            scale_or_bias_bytes,
        ) <= biases.len_bytes(),
        "affine quantized matmul biases range out of bounds: config={config:?} offset_bytes={biases_offset_bytes} \
         required_bytes={scale_or_bias_bytes} buffer_bytes={}",
        biases.len_bytes()
    );
}

fn affine_kernel_source(
    config: AffineQuantizedMatmulConfig,
    kind: AffineQuantizedMatmulKernelKind,
) -> (String, String) {
    match kind {
        AffineQuantizedMatmulKernelKind::QmvBn8Bk32 => affine_qmv_bn8_bk32_source(config),
        AffineQuantizedMatmulKernelKind::QmvQuadBn64 => affine_qmv_quad_bn64_source(config),
        AffineQuantizedMatmulKernelKind::QmmBm8Bn32 => affine_qmm_bn32_source(config, 8),
        AffineQuantizedMatmulKernelKind::QmmBm16Bn32 => affine_qmm_bn32_source(config, 16),
        AffineQuantizedMatmulKernelKind::QmmBm32Bn32 => affine_qmm_bn32_source(config, 32),
    }
}

fn affine_qmm_bn32_source(config: AffineQuantizedMatmulConfig, bm: usize) -> (String, String) {
    assert!(matches!(bm, 8 | 16 | 32));
    if !config.uses_same_dtype() {
        let input_type = metal_type_string(config.input_dtype);
        let output_type = metal_type_string(config.output_dtype);
        let scale_bias_type = metal_type_string(config.scale_bias_dtype);
        let aligned = config.n % 32 == 0;
        let kernel_name = format!(
            "mixed_qmm_t_bm{bm}_bn32_{input_type}_{scale_bias_type}_{output_type}_gs_{}_b_{}_alN_{}",
            config.group_size, config.bits, aligned
        );
        let template_definition = template_definition(
            &kernel_name,
            "mixed_qmm_t",
            &[
                input_type.to_string(),
                scale_bias_type.to_string(),
                output_type.to_string(),
                config.group_size.to_string(),
                config.bits.to_string(),
                aligned.to_string(),
                bm.to_string(),
            ],
        );
        return (
            kernel_name,
            affine_quantized_source(&format!("{MIXED_AFFINE_SOURCE}\n{template_definition}")),
        );
    }

    if bm == 32 {
        let type_string = metal_type_string(config.input_dtype);
        let aligned = config.n % 32 == 0;
        let kernel_name = format!(
            "psi_dec_qmm_t_{type_string}_gs_{}_b_{}_alN_{}_batch_0",
            config.group_size, config.bits, aligned
        );
        let template_definition = template_definition(
            &kernel_name,
            "psi_dec_qmm_t",
            &[
                type_string.to_string(),
                config.group_size.to_string(),
                config.bits.to_string(),
                aligned.to_string(),
                "false".to_string(),
            ],
        );
        return (
            kernel_name,
            affine_quantized_source(&format!("{GUARDED_AFFINE_SOURCE}\n{template_definition}")),
        );
    }

    let type_string = metal_type_string(config.input_dtype);
    let aligned = config.n % 32 == 0;
    let kernel_name = format!(
        "qmm_t_bm{bm}_bn32_{type_string}_gs_{}_b_{}_alN_{}",
        config.group_size, config.bits, aligned,
    );
    let template_definition = template_definition(
        &kernel_name,
        "qmm_t_bm8_bm16_bn32",
        &[
            type_string.to_string(),
            config.group_size.to_string(),
            config.bits.to_string(),
            aligned.to_string(),
            bm.to_string(),
        ],
    );
    (
        kernel_name,
        affine_quantized_source(&format!("{QMM_BM8_BM16_BN32_SOURCE}\n{template_definition}")),
    )
}

fn affine_qmv_bn8_bk32_source(config: AffineQuantizedMatmulConfig) -> (String, String) {
    if !config.uses_same_dtype() {
        let input_type = metal_type_string(config.input_dtype);
        let output_type = metal_type_string(config.output_dtype);
        let scale_bias_type = metal_type_string(config.scale_bias_dtype);
        let fast = config.n % 8 == 0 && config.k % 512 == 0;
        let function_name = if fast { "mixed_qmv_fast" } else { "mixed_qmv" };
        let kernel_name = format!(
            "{function_name}_{input_type}_{scale_bias_type}_{output_type}_gs_{}_b_{}",
            config.group_size, config.bits
        );
        let template_definition = template_definition(
            &kernel_name,
            function_name,
            &[
                input_type.to_string(),
                scale_bias_type.to_string(),
                output_type.to_string(),
                config.group_size.to_string(),
                config.bits.to_string(),
            ],
        );
        return (
            kernel_name,
            affine_quantized_source(&format!("{MIXED_AFFINE_SOURCE}\n{template_definition}")),
        );
    }

    let type_string = metal_type_string(config.input_dtype);
    let fast = config.n % 8 == 0 && config.k % 512 == 0;
    let function_name = if fast { "psi_dec_qmv_fast" } else { "psi_dec_qmv" };
    let kernel_name = format!(
        "{function_name}_{type_string}_gs_{}_b_{}_batch_0",
        config.group_size, config.bits
    );
    let template_definition = template_definition(
        &kernel_name,
        function_name,
        &[
            type_string.to_string(),
            config.group_size.to_string(),
            config.bits.to_string(),
            "false".to_string(),
        ],
    );
    (
        kernel_name,
        affine_quantized_source(&format!("{GUARDED_AFFINE_SOURCE}\n{template_definition}")),
    )
}

fn affine_qmv_quad_bn64_source(config: AffineQuantizedMatmulConfig) -> (String, String) {
    let type_string = metal_type_string(config.input_dtype);
    let kernel_name = format!(
        "psi_dec_qmv_quad_{type_string}_gs_{}_b_{}_d_{}_batch_0",
        config.group_size, config.bits, config.k
    );
    let template_definition = template_definition(
        &kernel_name,
        "psi_dec_qmv_quad",
        &[
            type_string.to_string(),
            config.group_size.to_string(),
            config.bits.to_string(),
            config.k.to_string(),
            "false".to_string(),
        ],
    );
    (
        kernel_name,
        affine_quantized_source(&format!("{GUARDED_AFFINE_SOURCE}\n{template_definition}")),
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

fn adaptive_qmv_batch_limit(config: AffineQuantizedMatmulConfig) -> i32 {
    if config.input_dtype == Dtype::Bfloat16 && config.n >= 65_536 {
        if config.k <= 2048 { 5 } else { 6 }
    } else if config.n > 4096 || config.k > 4096 {
        6
    } else {
        qmv_batch_limit(config.k, config.n)
    }
}

const QMM_BM8_MAX_ROWS: i32 = 8;
const QMM_BM16_MAX_ROWS: i32 = 16;

fn adaptive_topology_boundaries(config: AffineQuantizedMatmulConfig) -> Box<[u32]> {
    let mut candidates = [
        adaptive_qmv_batch_limit(config),
        QMM_BM8_MAX_ROWS + 1,
        QMM_BM16_MAX_ROWS + 1,
    ];
    candidates.sort_unstable();
    let mut boundaries = Vec::with_capacity(candidates.len());
    for boundary in candidates {
        if boundary > 1 && select_kernel_kind(config, boundary - 1) != select_kernel_kind(config, boundary) {
            let boundary = u32::try_from(boundary).expect("affine topology boundary must fit u32");
            if boundaries.last() != Some(&boundary) {
                boundaries.push(boundary);
            }
        }
    }
    boundaries.into_boxed_slice()
}

fn select_kernel_kind(config: AffineQuantizedMatmulConfig, m: i32) -> AffineQuantizedMatmulKernelKind {
    config.validate();
    assert!(m > 0);
    if m < adaptive_qmv_batch_limit(config) {
        select_qmv_kernel_kind(config)
    } else if config.n < 65_536 && (config.n > 4096 || config.k > 4096) && m <= QMM_BM8_MAX_ROWS {
        AffineQuantizedMatmulKernelKind::QmmBm8Bn32
    } else if m <= QMM_BM16_MAX_ROWS {
        AffineQuantizedMatmulKernelKind::QmmBm16Bn32
    } else {
        AffineQuantizedMatmulKernelKind::QmmBm32Bn32
    }
}

fn select_qmv_kernel_kind(config: AffineQuantizedMatmulConfig) -> AffineQuantizedMatmulKernelKind {
    if config.uses_same_dtype() && matches!(config.k, 64 | 128) && is_power_of_two(config.bits) {
        AffineQuantizedMatmulKernelKind::QmvQuadBn64
    } else {
        AffineQuantizedMatmulKernelKind::QmvBn8Bk32
    }
}

fn validate_kernel_kind(config: AffineQuantizedMatmulConfig, kind: AffineQuantizedMatmulKernelKind) {
    match kind {
        AffineQuantizedMatmulKernelKind::QmvBn8Bk32
        | AffineQuantizedMatmulKernelKind::QmmBm8Bn32
        | AffineQuantizedMatmulKernelKind::QmmBm16Bn32
        | AffineQuantizedMatmulKernelKind::QmmBm32Bn32 => {},
        AffineQuantizedMatmulKernelKind::QmvQuadBn64 => {
            assert!(
                config.uses_same_dtype(),
                "QMV quad BN=64 affine matmul requires one dtype"
            );
            assert!(
                matches!(config.k, 64 | 128) && is_power_of_two(config.bits),
                "QMV quad BN=64 affine matmul requires K=64 or K=128 and power-of-two weight bits"
            );
        },
    }
}

fn validate_qmm_pipeline(
    device: &Device,
    config: AffineQuantizedMatmulConfig,
    kind: AffineQuantizedMatmulKernelKind,
    kernel: &Kernel,
) {
    let (bm, num_simdgroups): (usize, usize) = match kind {
        AffineQuantizedMatmulKernelKind::QmmBm8Bn32 => (8, 2),
        AffineQuantizedMatmulKernelKind::QmmBm16Bn32 => (16, 2),
        AffineQuantizedMatmulKernelKind::QmmBm32Bn32 => (32, 4),
        _ => panic!("QMM pipeline validation requires a QMM kernel kind"),
    };
    let num_threads = num_simdgroups * kernel.thread_execution_width();
    assert_eq!(
        kernel.thread_execution_width(),
        32,
        "QMM BM={bm} BN=32 requires a 32-thread SIMDgroup"
    );
    assert!(
        num_threads <= kernel.max_total_threads_per_threadblock(),
        "QMM BM={bm} BN=32 requires {num_threads} threads, pipeline supports {}",
        kernel.max_total_threads_per_threadblock()
    );
    let item_size = if config.uses_same_dtype() {
        config.input_dtype.item_size()
    } else {
        Dtype::Float32.item_size()
    };
    let bk_padded = 32 + 16 / item_size;
    let expected_threadblock_memory = (bm + 32)
        .checked_mul(bk_padded)
        .and_then(|elements| elements.checked_mul(item_size))
        .expect("QMM BN=32 threadblock memory must fit usize");
    assert!(
        expected_threadblock_memory <= device.max_threadblock_memory_length(),
        "QMM BM={bm} BN=32 requires {expected_threadblock_memory} bytes of threadblock memory, device supports {}",
        device.max_threadblock_memory_length()
    );
    assert_eq!(
        kernel.static_threadblock_memory_length(),
        expected_threadblock_memory,
        "QMM BM={bm} BN=32 pipeline threadblock memory does not match its tile"
    );
    assert!(
        kernel.static_threadblock_memory_length() <= device.max_threadblock_memory_length(),
        "QMM BM={bm} BN=32 pipeline uses {} bytes of threadblock memory, device supports {}",
        kernel.static_threadblock_memory_length(),
        device.max_threadblock_memory_length()
    );
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

const QMM_BM8_BM16_BN32_SOURCE: &str = include_str!("metal/affine_quantized_qmm_bm8_bm16_bn32.metal");

const MIXED_AFFINE_SOURCE: &str = include_str!("metal/affine_quantized_mixed_qmv_qmm.metal");

const GUARDED_AFFINE_SOURCE: &str = include_str!("metal/affine_quantized_guarded_qmv_qmm.metal");

const GATE_UP_SWIGLU_SOURCE: &str = include_str!("metal/affine_quantized_gate_up_swiglu_qmv_qmm.metal");

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
    content.contains("METAL_FUNC void qmv_quad_impl(")
        && content.contains("METAL_FUNC void qmv_fast_impl(")
        && content.contains("METAL_FUNC void qmv_impl(")
        && content.contains("METAL_FUNC void qmm_t_impl(")
}

#[cfg(test)]
mod tests {
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
            select_kernel_kind(adaptive_config(151_936, 2048, Dtype::Bfloat16), 4),
            AffineQuantizedMatmulKernelKind::QmvBn8Bk32
        );
        assert_eq!(
            select_kernel_kind(adaptive_config(151_936, 2048, Dtype::Bfloat16), 5),
            AffineQuantizedMatmulKernelKind::QmmBm16Bn32
        );
        assert_eq!(
            select_kernel_kind(adaptive_config(151_936, 5120, Dtype::Bfloat16), 5),
            AffineQuantizedMatmulKernelKind::QmvBn8Bk32
        );
        assert_eq!(
            select_kernel_kind(adaptive_config(151_936, 5120, Dtype::Bfloat16), 6),
            AffineQuantizedMatmulKernelKind::QmmBm16Bn32
        );
    }

    #[test]
    fn test_adaptive_qmm_tile_crossover() {
        assert_eq!(
            select_kernel_kind(adaptive_config(151_936, 5120, Dtype::Bfloat16), 16),
            AffineQuantizedMatmulKernelKind::QmmBm16Bn32
        );
        assert_eq!(
            select_kernel_kind(adaptive_config(151_936, 5120, Dtype::Bfloat16), 17),
            AffineQuantizedMatmulKernelKind::QmmBm32Bn32
        );
    }

    #[test]
    fn test_adaptive_dense_projection_crossover() {
        let large_projection = adaptive_config(34_816, 5120, Dtype::Bfloat16);
        assert_eq!(
            select_kernel_kind(large_projection, 5),
            AffineQuantizedMatmulKernelKind::QmvBn8Bk32
        );
        assert_eq!(
            select_kernel_kind(large_projection, 6),
            AffineQuantizedMatmulKernelKind::QmmBm8Bn32
        );
        assert_eq!(
            select_kernel_kind(large_projection, 8),
            AffineQuantizedMatmulKernelKind::QmmBm8Bn32
        );
        assert_eq!(
            select_kernel_kind(large_projection, 9),
            AffineQuantizedMatmulKernelKind::QmmBm16Bn32
        );

        let common_projection = adaptive_config(1024, 2048, Dtype::Bfloat16);
        assert_eq!(
            select_kernel_kind(common_projection, 8),
            AffineQuantizedMatmulKernelKind::QmvBn8Bk32
        );
        assert_eq!(
            select_kernel_kind(common_projection, 18),
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
                    select_kernel_kind(config, num_active_rows as i32),
                    select_kernel_kind(config, num_total_rows as i32),
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
        assert_eq!(bucketed_replay.stats().parameter_count, 1);
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
                        (&matmul.qmv, 2),
                        (&matmul.qmm_bm8_bn32, 7),
                        (&matmul.qmm_bm16_bn32, 15),
                        (&matmul.qmm_bm32_bn32, 31),
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
            AffineQuantizedMatmulKernel::new(&device, config, select_kernel_kind(config, m)).invoke(
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
            AffineQuantizedMatmulKernel::new(&device, config, select_kernel_kind(config, m)).invoke(
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
            AffineQuantizedMatmulKernel::new(&device, config, select_kernel_kind(config, m)).invoke(
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
            AffineQuantizedMatmulKernel::new(&device, config, select_kernel_kind(config, m)).invoke(
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
            AffineQuantizedMatmulKernel::new(&device, config, select_kernel_kind(config, m)).invoke(
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
            AffineQuantizedMatmulKernel::new(&device, config, select_kernel_kind(config, m)).invoke(
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
}
