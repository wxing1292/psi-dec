use std::collections::HashSet;
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
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
pub struct Config {
    pub n: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
    pub scale_bias_dtype: Dtype,
}

impl Config {
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

pub struct Kernel {
    config: Config,
    kind: KernelKind,
    kernel: CompiledKernel,
}

pub struct Matmul {
    config: Config,
    registry: Registry,
}

struct Registry {
    entries: Vec<(KernelKind, Kernel)>,
}

struct Selector;

#[derive(Clone, Copy, Debug)]
pub struct ExpertConfig {
    pub num_experts: i32,
    pub matmul: Config,
}

impl ExpertConfig {
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
pub struct GatherShape {
    pub num_routes: i32,
    pub num_input_vectors: i32,
}

impl GatherShape {
    pub fn validate(self) {
        assert!(self.num_routes > 0);
        assert!(self.num_input_vectors > 0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RaggedExpertMajorShape {
    pub num_routes: i32,
}

impl RaggedExpertMajorShape {
    pub fn validate(self) {
        assert!(self.num_routes > 0);
    }
}

#[derive(Clone, Copy)]
struct ExpertBucketedRoutes {
    num_total_tokens: u32,
    num_experts_per_token: u32,
    num_active_tokens_key: ReplayParameterKey,
}

impl ExpertBucketedRoutes {
    fn new(
        config: ExpertConfig,
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

    fn bind(self, recorder: &CommandRecorder<'_>, token_binding_index: usize, topk_binding_index: usize) {
        recorder.bind_u32(
            token_binding_index,
            self.num_active_tokens_key,
            1,
            self.num_total_tokens,
        );
        recorder.set_u32(topk_binding_index, self.num_experts_per_token);
    }
}

pub struct GatherMatmulKernel {
    config: ExpertConfig,
    kernel: CompiledKernel,
    bucketed_kernel: CompiledKernel,
}

pub struct GatherGateUpSwiGLUKernel {
    config: ExpertConfig,
    kernel: CompiledKernel,
    bucketed_kernel: CompiledKernel,
}

pub struct RaggedExpertMajorGateUpSwiGLUKernel {
    config: ExpertConfig,
    kernel: CompiledKernel,
    bucketed_kernel: CompiledKernel,
}

pub struct RaggedExpertMajorMatmulKernel {
    config: ExpertConfig,
    kernel: CompiledKernel,
    bucketed_kernel: CompiledKernel,
}

impl GatherMatmulKernel {
    pub fn new(device: &Device, config: ExpertConfig) -> Self {
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
        let kernel = CompiledKernel::new(device, &source, &kernel_name);
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
        let bucketed_kernel = CompiledKernel::new(device, &bucketed_source, &bucketed_kernel_name);
        Self {
            config,
            kernel,
            bucketed_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: GatherShape,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        lhs_indices: &'a Buffer,
        rhs_indices: &'a Buffer,
    ) -> GatherMatmulInvocation<'a> {
        GatherMatmulInvocation {
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
        shape: GatherShape,
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
    ) -> GatherMatmulInvocation<'a> {
        let bucketed_routes = ExpertBucketedRoutes::new(
            self.config,
            shape.num_routes,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
        GatherMatmulInvocation {
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

impl GatherGateUpSwiGLUKernel {
    pub fn new(device: &Device, config: ExpertConfig) -> Self {
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
        let kernel = CompiledKernel::new(device, &source, &kernel_name);
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
        let bucketed_kernel = CompiledKernel::new(device, &bucketed_source, &bucketed_kernel_name);
        Self {
            config,
            kernel,
            bucketed_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: GatherShape,
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
    ) -> GatherGateUpSwiGLUInvocation<'a> {
        GatherGateUpSwiGLUInvocation {
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
        shape: GatherShape,
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
    ) -> GatherGateUpSwiGLUInvocation<'a> {
        let bucketed_routes = ExpertBucketedRoutes::new(
            self.config,
            shape.num_routes,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
        GatherGateUpSwiGLUInvocation {
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

impl RaggedExpertMajorGateUpSwiGLUKernel {
    pub fn new(device: &Device, config: ExpertConfig) -> Self {
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
        let kernel = CompiledKernel::new(device, &source, &kernel_name);
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
        let bucketed_kernel = CompiledKernel::new(device, &bucketed_source, &bucketed_kernel_name);
        Self {
            config,
            kernel,
            bucketed_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: RaggedExpertMajorShape,
        output: &'a Buffer,
        input: &'a Buffer,
        gate_weight: &'a Buffer,
        gate_scales: &'a Buffer,
        gate_biases: &'a Buffer,
        up_weight: &'a Buffer,
        up_scales: &'a Buffer,
        up_biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorGateUpSwiGLUInvocation<'a> {
        RaggedExpertMajorGateUpSwiGLUInvocation {
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
        shape: RaggedExpertMajorShape,
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
    ) -> RaggedExpertMajorGateUpSwiGLUInvocation<'a> {
        let bucketed_routes = ExpertBucketedRoutes::new(
            self.config,
            shape.num_routes,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
        RaggedExpertMajorGateUpSwiGLUInvocation {
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

impl RaggedExpertMajorMatmulKernel {
    pub fn new(device: &Device, config: ExpertConfig) -> Self {
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
        let kernel = CompiledKernel::new(device, &source, &kernel_name);
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
        let bucketed_kernel = CompiledKernel::new(device, &bucketed_source, &bucketed_kernel_name);
        Self {
            config,
            kernel,
            bucketed_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: RaggedExpertMajorShape,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorMatmulInvocation<'a> {
        RaggedExpertMajorMatmulInvocation {
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
        shape: RaggedExpertMajorShape,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens_key: ReplayParameterKey,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        experts_by_route: &'a Buffer,
    ) -> RaggedExpertMajorMatmulInvocation<'a> {
        let bucketed_routes = ExpertBucketedRoutes::new(
            self.config,
            shape.num_routes,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
        RaggedExpertMajorMatmulInvocation {
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

pub struct GatherMatmulInvocation<'a> {
    kernel: &'a GatherMatmulKernel,
    shape: GatherShape,
    output: &'a Buffer,
    input: &'a Buffer,
    weight: &'a Buffer,
    scales: &'a Buffer,
    biases: &'a Buffer,
    lhs_indices: &'a Buffer,
    rhs_indices: &'a Buffer,
    bucketed_routes: Option<ExpertBucketedRoutes>,
}

pub struct GatherGateUpSwiGLUInvocation<'a> {
    kernel: &'a GatherGateUpSwiGLUKernel,
    shape: GatherShape,
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
    bucketed_routes: Option<ExpertBucketedRoutes>,
}

pub struct RaggedExpertMajorGateUpSwiGLUInvocation<'a> {
    kernel: &'a RaggedExpertMajorGateUpSwiGLUKernel,
    shape: RaggedExpertMajorShape,
    output: &'a Buffer,
    input: &'a Buffer,
    gate_weight: &'a Buffer,
    gate_scales: &'a Buffer,
    gate_biases: &'a Buffer,
    up_weight: &'a Buffer,
    up_scales: &'a Buffer,
    up_biases: &'a Buffer,
    experts_by_route: &'a Buffer,
    bucketed_routes: Option<ExpertBucketedRoutes>,
}

pub struct RaggedExpertMajorMatmulInvocation<'a> {
    kernel: &'a RaggedExpertMajorMatmulKernel,
    shape: RaggedExpertMajorShape,
    output: &'a Buffer,
    input: &'a Buffer,
    weight: &'a Buffer,
    scales: &'a Buffer,
    biases: &'a Buffer,
    experts_by_route: &'a Buffer,
    bucketed_routes: Option<ExpertBucketedRoutes>,
}

impl Operator for RaggedExpertMajorMatmulInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
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
        recorder.set_kernel(kernel);
        recorder.set_buffer_read(0, self.weight, 0);
        recorder.set_buffer_read(1, self.scales, 0);
        recorder.set_buffer_read(2, self.biases, 0);
        recorder.set_buffer_read(3, self.input, 0);
        recorder.set_buffer_read(4, self.experts_by_route, 0);
        recorder.set_buffer_write(5, self.output, 0);
        recorder.set_i32(6, matmul.k);
        recorder.set_i32(7, matmul.n);
        recorder.set_i32(8, config.num_experts);
        if let Some(bucketed_routes) = self.bucketed_routes {
            bucketed_routes.bind(recorder, 9, 10);
        }
        recorder.dispatch_threadblocks(
            (shape.num_routes as usize, ceil_div_i32(matmul.n, 8) as usize, 1),
            (32, 2, 1),
        );
    }
}

impl Operator for RaggedExpertMajorGateUpSwiGLUInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
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
        recorder.set_kernel(kernel);
        recorder.set_buffer_read(0, self.gate_weight, 0);
        recorder.set_buffer_read(1, self.gate_scales, 0);
        recorder.set_buffer_read(2, self.gate_biases, 0);
        recorder.set_buffer_read(3, self.up_weight, 0);
        recorder.set_buffer_read(4, self.up_scales, 0);
        recorder.set_buffer_read(5, self.up_biases, 0);
        recorder.set_buffer_read(6, self.input, 0);
        recorder.set_buffer_read(7, self.experts_by_route, 0);
        recorder.set_buffer_write(8, self.output, 0);
        recorder.set_i32(9, matmul.k);
        recorder.set_i32(10, matmul.n);
        recorder.set_i32(11, config.num_experts);
        if let Some(bucketed_routes) = self.bucketed_routes {
            bucketed_routes.bind(recorder, 12, 13);
        }
        recorder.dispatch_threadblocks(
            (shape.num_routes as usize, ceil_div_i32(matmul.n, 8) as usize, 1),
            (32, 2, 1),
        );
    }
}

impl Operator for GatherGateUpSwiGLUInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
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
        recorder.set_kernel(kernel);
        recorder.set_buffer_read(0, self.gate_weight, 0);
        recorder.set_buffer_read(1, self.gate_scales, 0);
        recorder.set_buffer_read(2, self.gate_biases, 0);
        recorder.set_buffer_read(3, self.up_weight, 0);
        recorder.set_buffer_read(4, self.up_scales, 0);
        recorder.set_buffer_read(5, self.up_biases, 0);
        recorder.set_buffer_read(6, self.input, 0);
        recorder.set_buffer_read(7, self.lhs_indices, 0);
        recorder.set_buffer_read(8, self.rhs_indices, 0);
        recorder.set_buffer_write(9, self.output, 0);
        recorder.set_i32(10, matmul.k);
        recorder.set_i32(11, matmul.n);
        recorder.set_i32(12, config.num_experts);
        if let Some(bucketed_routes) = self.bucketed_routes {
            bucketed_routes.bind(recorder, 13, 14);
        }
        recorder.dispatch_threadblocks(
            (1, ceil_div_i32(matmul.n, 8) as usize, shape.num_routes as usize),
            (32, 2, 1),
        );
    }
}

impl Operator for GatherMatmulInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
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
        recorder.set_kernel(kernel);
        recorder.set_buffer_read(0, self.weight, 0);
        recorder.set_buffer_read(1, self.scales, 0);
        recorder.set_buffer_read(2, self.biases, 0);
        recorder.set_buffer_read(3, self.input, 0);
        recorder.set_buffer_read(4, self.lhs_indices, 0);
        recorder.set_buffer_read(5, self.rhs_indices, 0);
        recorder.set_buffer_write(6, self.output, 0);
        recorder.set_i32(7, matmul.k);
        recorder.set_i32(8, matmul.n);

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

        recorder.set_i32(9, 2);
        recorder.set_i32_slice(10, &x_shape);
        recorder.set_i64_slice(11, &x_strides);
        recorder.set_i32(12, 1);
        recorder.set_i32_slice(13, &w_shape);
        recorder.set_i64_slice(14, &w_strides);
        recorder.set_i64_slice(15, &affine_strides);
        recorder.set_i64_slice(16, &affine_strides);
        recorder.set_i32(17, 1);
        recorder.set_i32_slice(18, &batch_shape);
        recorder.set_i64_slice(19, &route_strides);
        recorder.set_i64_slice(20, &route_strides);
        if let Some(bucketed_routes) = self.bucketed_routes {
            bucketed_routes.bind(recorder, 21, 22);
        }

        let bn = 8;
        let bk = 32;
        recorder.dispatch_threadblocks(
            (1, ceil_div_i32(matmul.n, bn) as usize, shape.num_routes as usize),
            (bk as usize, 2, 1),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_gather_buffer_ranges(
    config: ExpertConfig,
    shape: GatherShape,
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
    config: ExpertConfig,
    shape: GatherShape,
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
    config: ExpertConfig,
    shape: RaggedExpertMajorShape,
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
    config: ExpertConfig,
    shape: RaggedExpertMajorShape,
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
pub enum KernelKind {
    QmvBn8Bk32,
    QmvQuadBn64,
    QmmBm8Bn32,
    QmmBm16Bn32,
    QmmBm32Bn32,
}

impl Kernel {
    pub fn new(device: &Device, config: Config, kind: KernelKind) -> Self {
        config.validate();
        validate_kernel_kind(config, kind);
        let (kernel_name, source) = affine_kernel_source(config, kind);
        let kernel = CompiledKernel::new(device, &source, &kernel_name);
        if matches!(
            kind,
            KernelKind::QmmBm8Bn32 | KernelKind::QmmBm16Bn32 | KernelKind::QmmBm32Bn32
        ) {
            validate_qmm_pipeline(device, config, kind, &kernel);
        }
        Self { config, kind, kernel }
    }

    pub fn kind(&self) -> KernelKind {
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
    ) -> Invocation<'a> {
        assert!(m > 0);
        Invocation {
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
    ) -> Invocation<'a> {
        validate_num_total_rows(num_total_rows);
        Invocation {
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

impl Matmul {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            registry: Registry::new(device, config),
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
    ) -> Invocation<'a> {
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
    ) -> Invocation<'a> {
        let (_, kernel) = Selector::select(
            &self.registry,
            self.config,
            i32::try_from(num_total_rows).expect("affine total row count must fit i32"),
        );
        kernel.invoke_bucketed(
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
    pub fn topology(&self, num_total_rows: u32) -> KernelKind {
        validate_num_total_rows(num_total_rows);
        Selector::select(
            &self.registry,
            self.config,
            i32::try_from(num_total_rows).expect("affine total row count must fit i32"),
        )
        .0
    }

    /// Returns the first row count for each change in recorded kernel topology.
    pub fn topology_boundaries(&self) -> Box<[u32]> {
        adaptive_topology_boundaries(self.config)
    }

    pub fn selected_kernel(&self, m: i32) -> &Kernel {
        assert!(m > 0);
        Selector::select(&self.registry, self.config, m).1
    }
}

impl Registry {
    fn new(device: &Device, config: Config) -> Self {
        let qmv_key = Selector::qmv_key(config);
        Self {
            entries: vec![
                (qmv_key, Kernel::new(device, config, qmv_key)),
                (
                    KernelKind::QmmBm8Bn32,
                    Kernel::new(device, config, KernelKind::QmmBm8Bn32),
                ),
                (
                    KernelKind::QmmBm16Bn32,
                    Kernel::new(device, config, KernelKind::QmmBm16Bn32),
                ),
                (
                    KernelKind::QmmBm32Bn32,
                    Kernel::new(device, config, KernelKind::QmmBm32Bn32),
                ),
            ],
        }
    }

    fn get(&self, key: KernelKind) -> &Kernel {
        self.entries
            .iter()
            .find_map(|(candidate_key, kernel)| (*candidate_key == key).then_some(kernel))
            .unwrap_or_else(|| panic!("missing affine quantized matmul execution variant {key:?}"))
    }
}

impl Selector {
    fn select(registry: &Registry, config: Config, num_rows: i32) -> (KernelKind, &Kernel) {
        let key = Self::key(config, num_rows);
        (key, registry.get(key))
    }

    fn key(config: Config, num_rows: i32) -> KernelKind {
        config.validate();
        assert!(num_rows > 0);
        if num_rows < adaptive_qmv_batch_limit(config) {
            Self::qmv_key(config)
        } else if config.n < 65_536 && (config.n > 4096 || config.k > 4096) && num_rows <= QMM_BM8_MAX_ROWS {
            KernelKind::QmmBm8Bn32
        } else if num_rows <= QMM_BM16_MAX_ROWS {
            KernelKind::QmmBm16Bn32
        } else {
            KernelKind::QmmBm32Bn32
        }
    }

    fn qmv_key(config: Config) -> KernelKind {
        if config.uses_same_dtype() && matches!(config.k, 64 | 128) && is_power_of_two(config.bits) {
            KernelKind::QmvQuadBn64
        } else {
            KernelKind::QmvBn8Bk32
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a Kernel,
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

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
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

        recorder.set_kernel(&kernel.kernel);
        recorder.set_buffer_read(0, weight, weight_offset_bytes);
        recorder.set_buffer_read(1, scales, scales_offset_bytes);
        recorder.set_buffer_read(2, biases, biases_offset_bytes);
        recorder.set_buffer_read(3, input, input_offset_bytes);
        recorder.set_buffer_write(4, output, output_offset_bytes);
        recorder.set_i32(5, config.k);
        recorder.set_i32(6, config.n);
        record_num_active_rows(recorder, 7, num_total_rows, self.num_active_rows_key);

        match kernel.kind {
            KernelKind::QmmBm8Bn32 => {
                recorder.dispatch_threadblocks(
                    (
                        ceil_div_i32(config.n, 32) as usize,
                        ceil_div_i32(num_total_rows_i32, 8) as usize,
                        1,
                    ),
                    (32, 2, 1),
                );
            },
            KernelKind::QmmBm16Bn32 => {
                recorder.dispatch_threadblocks(
                    (
                        ceil_div_i32(config.n, 32) as usize,
                        ceil_div_i32(num_total_rows_i32, 16) as usize,
                        1,
                    ),
                    (32, 2, 1),
                );
            },
            KernelKind::QmmBm32Bn32 => {
                recorder.dispatch_threadblocks(
                    (
                        ceil_div_i32(config.n, 32) as usize,
                        ceil_div_i32(num_total_rows_i32, 32) as usize,
                        1,
                    ),
                    (32, 2, 2),
                );
            },
            KernelKind::QmvQuadBn64 => {
                recorder.dispatch_threadblocks(
                    (num_total_rows as usize, ceil_div_i32(config.n, 64) as usize, 1),
                    (32, 1, 1),
                );
            },
            KernelKind::QmvBn8Bk32 => {
                recorder.dispatch_threadblocks(
                    (num_total_rows as usize, ceil_div_i32(config.n, 8) as usize, 1),
                    (32, 2, 1),
                );
            },
        }
    }
}

fn record_num_active_rows(
    recorder: &CommandRecorder,
    binding_index: usize,
    num_total_rows: u32,
    key: Option<ReplayParameterKey>,
) {
    match key {
        Some(key) => recorder.bind_u32(binding_index, key, 1, num_total_rows),
        None => recorder.set_u32(binding_index, num_total_rows),
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
    config: Config,
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

fn affine_kernel_source(config: Config, kind: KernelKind) -> (String, String) {
    match kind {
        KernelKind::QmvBn8Bk32 => affine_qmv_bn8_bk32_source(config),
        KernelKind::QmvQuadBn64 => affine_qmv_quad_bn64_source(config),
        KernelKind::QmmBm8Bn32 => affine_qmm_bn32_source(config, 8),
        KernelKind::QmmBm16Bn32 => affine_qmm_bn32_source(config, 16),
        KernelKind::QmmBm32Bn32 => affine_qmm_bn32_source(config, 32),
    }
}

fn affine_qmm_bn32_source(config: Config, bm: usize) -> (String, String) {
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

fn affine_qmv_bn8_bk32_source(config: Config) -> (String, String) {
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

fn affine_qmv_quad_bn64_source(config: Config) -> (String, String) {
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

fn adaptive_qmv_batch_limit(config: Config) -> i32 {
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

fn adaptive_topology_boundaries(config: Config) -> Box<[u32]> {
    let mut candidates = [
        adaptive_qmv_batch_limit(config),
        QMM_BM8_MAX_ROWS + 1,
        QMM_BM16_MAX_ROWS + 1,
    ];
    candidates.sort_unstable();
    let mut boundaries = Vec::with_capacity(candidates.len());
    for boundary in candidates {
        if boundary > 1 && Selector::key(config, boundary - 1) != Selector::key(config, boundary) {
            let boundary = u32::try_from(boundary).expect("affine topology boundary must fit u32");
            if boundaries.last() != Some(&boundary) {
                boundaries.push(boundary);
            }
        }
    }
    boundaries.into_boxed_slice()
}

fn validate_kernel_kind(config: Config, kind: KernelKind) {
    match kind {
        KernelKind::QmvBn8Bk32 | KernelKind::QmmBm8Bn32 | KernelKind::QmmBm16Bn32 | KernelKind::QmmBm32Bn32 => {},
        KernelKind::QmvQuadBn64 => {
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

fn validate_qmm_pipeline(device: &Device, config: Config, kind: KernelKind, kernel: &CompiledKernel) {
    let (bm, num_simdgroups): (usize, usize) = match kind {
        KernelKind::QmmBm8Bn32 => (8, 2),
        KernelKind::QmmBm16Bn32 => (16, 2),
        KernelKind::QmmBm32Bn32 => (32, 4),
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
#[path = "affine_quantized_test.rs"]
mod tests;
