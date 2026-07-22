use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::operators::GatherAffineQuantizedMatmulKernel;
use crate::operators::GatherAffineQuantizedMatmulShape;
use crate::operators::RaggedExpertMajorAffineQuantizedGateUpSiluKernel;
use crate::operators::RaggedExpertMajorAffineQuantizedGateUpSiluShape;
use crate::operators::RaggedExpertMajorAffineQuantizedMatmulKernel;
use crate::operators::RaggedExpertMajorAffineQuantizedMatmulShape;
use crate::operators::affine_quantized::GatherAffineQuantizedGateUpSiluKernel;
use crate::operators::affine_quantized::GatherAffineQuantizedGateUpSiluShape;

fn to_i32(value: u32, name: &str) -> i32 {
    value.try_into().unwrap_or_else(|_| panic!("{name} must fit i32"))
}

fn checked_bytes(name: &str, dimensions: &[usize], dtype: Dtype) -> usize {
    dimensions
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .and_then(|elements| elements.checked_mul(dtype.item_size()))
        .unwrap_or_else(|| panic!("{name} byte length must fit usize"))
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedSparseMLPConfig {
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub dtype: Dtype,
}

impl QuantizedSparseMLPConfig {
    pub fn validate(self) {
        assert!(self.hidden_dim > 0);
        assert!(self.intermediate_dim > 0);
        self.stacked_intermediate_dim();
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(
            self.hidden_dim % self.group_size,
            0,
            "sparse MLP hidden_dim must be group aligned"
        );
        assert_eq!(
            self.intermediate_dim % self.group_size,
            0,
            "sparse MLP intermediate_dim must be group aligned"
        );
        assert_eq!(self.dtype, Dtype::Bfloat16, "sparse MLP currently supports bf16 only");
        i32::try_from(self.hidden_dim).expect("sparse MLP hidden_dim must fit i32");
        i32::try_from(self.intermediate_dim).expect("sparse MLP intermediate_dim must fit i32");
        i32::try_from(self.stacked_intermediate_dim()).expect("sparse MLP stacked intermediate_dim must fit i32");
        i32::try_from(self.group_size).expect("sparse MLP group_size must fit i32");
        i32::try_from(self.bits).expect("sparse MLP bits must fit i32");
    }

    pub fn token_major_fused_gate_up_silu_shape(
        self,
        shape: QuantizedSparseMLPTokenMajorShape,
    ) -> GatherAffineQuantizedGateUpSiluShape {
        self.validate();
        shape.validate();
        self.token_major_fused_gate_up_silu_shape_unchecked(shape)
    }

    fn token_major_fused_gate_up_silu_shape_unchecked(
        self,
        shape: QuantizedSparseMLPTokenMajorShape,
    ) -> GatherAffineQuantizedGateUpSiluShape {
        GatherAffineQuantizedGateUpSiluShape {
            num_routes: to_i32(shape.num_routes, "sparse MLP route count"),
            num_input_vectors: to_i32(shape.num_tokens, "sparse MLP token count"),
            intermediate_dim: to_i32(self.intermediate_dim, "sparse MLP intermediate_dim"),
            k: to_i32(self.hidden_dim, "sparse MLP hidden_dim"),
            group_size: to_i32(self.group_size, "sparse MLP group_size"),
            bits: to_i32(self.bits, "sparse MLP bits"),
            dtype: self.dtype,
        }
    }

    pub fn token_major_down_shape(self, shape: QuantizedSparseMLPTokenMajorShape) -> GatherAffineQuantizedMatmulShape {
        self.validate();
        shape.validate();
        self.token_major_down_shape_unchecked(shape)
    }

    fn token_major_down_shape_unchecked(
        self,
        shape: QuantizedSparseMLPTokenMajorShape,
    ) -> GatherAffineQuantizedMatmulShape {
        self.gather_shape_unchecked(shape, shape.num_routes, self.hidden_dim, self.intermediate_dim)
    }

    pub fn token_major_input_bytes(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        self.validate();
        shape.validate();
        self.token_major_input_bytes_unchecked(shape)
    }

    fn token_major_input_bytes_unchecked(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        checked_bytes(
            "sparse MLP token-major input",
            &[shape.num_tokens as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    pub fn token_major_route_indices_bytes(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        shape.validate();
        self.token_major_route_indices_bytes_unchecked(shape)
    }

    fn token_major_route_indices_bytes_unchecked(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        (shape.num_routes as usize)
            .checked_mul(size_of::<u32>())
            .expect("sparse MLP route-index byte length must fit usize")
    }

    pub fn activation_bytes(self, num_routes: u32) -> usize {
        self.validate();
        assert!(num_routes > 0);
        self.activation_bytes_unchecked(num_routes)
    }

    fn activation_bytes_unchecked(self, num_routes: u32) -> usize {
        checked_bytes(
            "sparse MLP activation",
            &[num_routes as usize, self.intermediate_dim as usize],
            self.dtype,
        )
    }

    pub fn token_major_output_bytes(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        self.validate();
        shape.validate();
        self.token_major_output_bytes_unchecked(shape)
    }

    pub fn expert_major_input_bytes(self, shape: QuantizedSparseMLPExpertMajorShape) -> usize {
        self.validate();
        shape.validate();
        checked_bytes(
            "sparse MLP expert-major input",
            &[shape.num_routes as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    pub fn expert_major_output_bytes(self, shape: QuantizedSparseMLPExpertMajorShape) -> usize {
        self.validate();
        shape.validate();
        checked_bytes(
            "sparse MLP expert-major output",
            &[shape.num_routes as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    pub fn expert_major_route_indices_bytes(self, shape: QuantizedSparseMLPExpertMajorShape) -> usize {
        shape.validate();
        (shape.num_routes as usize)
            .checked_mul(size_of::<u32>())
            .expect("sparse MLP expert-major route-index byte length must fit usize")
    }

    fn expert_major_fused_gate_up_silu_shape(
        self,
        shape: QuantizedSparseMLPExpertMajorShape,
    ) -> RaggedExpertMajorAffineQuantizedGateUpSiluShape {
        self.validate();
        shape.validate();
        RaggedExpertMajorAffineQuantizedGateUpSiluShape {
            num_experts: to_i32(shape.num_experts, "sparse MLP expert count"),
            num_routes: to_i32(shape.num_routes, "sparse MLP route count"),
            intermediate_dim: to_i32(self.intermediate_dim, "sparse MLP intermediate_dim"),
            k: to_i32(self.hidden_dim, "sparse MLP hidden_dim"),
            group_size: to_i32(self.group_size, "sparse MLP group_size"),
            bits: to_i32(self.bits, "sparse MLP bits"),
            dtype: self.dtype,
        }
    }

    fn expert_major_down_shape(
        self,
        shape: QuantizedSparseMLPExpertMajorShape,
    ) -> RaggedExpertMajorAffineQuantizedMatmulShape {
        self.validate();
        shape.validate();
        RaggedExpertMajorAffineQuantizedMatmulShape {
            num_experts: to_i32(shape.num_experts, "sparse MLP expert count"),
            num_routes: to_i32(shape.num_routes, "sparse MLP route count"),
            n: to_i32(self.hidden_dim, "sparse MLP hidden_dim"),
            k: to_i32(self.intermediate_dim, "sparse MLP intermediate_dim"),
            group_size: to_i32(self.group_size, "sparse MLP group_size"),
            bits: to_i32(self.bits, "sparse MLP bits"),
            dtype: self.dtype,
        }
    }

    fn token_major_output_bytes_unchecked(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        self.token_major_down_shape_unchecked(shape).output_bytes()
    }

    fn gather_shape_unchecked(
        self,
        shape: QuantizedSparseMLPTokenMajorShape,
        num_input_vectors: u32,
        n: u32,
        k: u32,
    ) -> GatherAffineQuantizedMatmulShape {
        GatherAffineQuantizedMatmulShape {
            num_routes: to_i32(shape.num_routes, "sparse MLP route count"),
            num_input_vectors: to_i32(num_input_vectors, "sparse MLP input-vector count"),
            n: to_i32(n, "sparse MLP output dimension"),
            k: to_i32(k, "sparse MLP input dimension"),
            group_size: to_i32(self.group_size, "sparse MLP group_size"),
            bits: to_i32(self.bits, "sparse MLP bits"),
            dtype: self.dtype,
        }
    }

    fn token_major_compile_shape(self, n: u32, k: u32) -> GatherAffineQuantizedMatmulShape {
        self.validate();
        GatherAffineQuantizedMatmulShape {
            num_routes: 1,
            num_input_vectors: 1,
            n: to_i32(n, "sparse MLP compile output dimension"),
            k: to_i32(k, "sparse MLP compile input dimension"),
            group_size: to_i32(self.group_size, "sparse MLP group_size"),
            bits: to_i32(self.bits, "sparse MLP bits"),
            dtype: self.dtype,
        }
    }

    fn token_major_compile_fused_gate_up_silu_shape(self) -> GatherAffineQuantizedGateUpSiluShape {
        self.validate();
        GatherAffineQuantizedGateUpSiluShape {
            num_routes: 1,
            num_input_vectors: 1,
            intermediate_dim: to_i32(self.intermediate_dim, "sparse MLP intermediate_dim"),
            k: to_i32(self.hidden_dim, "sparse MLP hidden_dim"),
            group_size: to_i32(self.group_size, "sparse MLP group_size"),
            bits: to_i32(self.bits, "sparse MLP bits"),
            dtype: self.dtype,
        }
    }

    fn stacked_intermediate_dim(self) -> u32 {
        self.intermediate_dim
            .checked_mul(2)
            .expect("sparse MLP stacked gate/up dim must fit u32")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedSparseMLPTokenMajorShape {
    pub num_routes: u32,
    pub num_tokens: u32,
}

impl QuantizedSparseMLPTokenMajorShape {
    pub fn validate(self) {
        assert!(self.num_routes > 0);
        assert!(self.num_tokens > 0);
        to_i32(self.num_routes, "sparse MLP route count");
        to_i32(self.num_tokens, "sparse MLP token count");
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedSparseMLPExpertMajorShape {
    pub num_experts: u32,
    pub num_routes: u32,
}

impl QuantizedSparseMLPExpertMajorShape {
    pub fn validate(self) {
        assert!(self.num_experts > 0);
        to_i32(self.num_experts, "sparse MLP expert count");
        to_i32(self.num_routes, "sparse MLP route count");
        assert!(self.num_routes > 0);
    }
}

#[derive(Clone, Copy)]
pub struct QuantizedSparseMLPTokenMajorBuffers<'a> {
    pub input: &'a Buffer,
    pub token_indices: &'a Buffer,
    pub expert_indices: &'a Buffer,
    pub route_indices: &'a Buffer,
    pub output: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QuantizedSparseMLPExpertMajorBuffers<'a> {
    pub packed_input: &'a Buffer,
    pub experts_by_route: &'a Buffer,
    pub route_output: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QuantizedSparseMLPWeights<'a> {
    pub gate_weight: &'a Buffer,
    pub gate_scales: &'a Buffer,
    pub gate_biases: &'a Buffer,
    pub up_weight: &'a Buffer,
    pub up_scales: &'a Buffer,
    pub up_biases: &'a Buffer,
    pub down_weight: &'a Buffer,
    pub down_scales: &'a Buffer,
    pub down_biases: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QuantizedSparseMLPTokenMajorScratch<'a> {
    pub activation: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QuantizedSparseMLPExpertMajorScratch<'a> {
    pub activation: &'a Buffer,
}

pub struct QuantizedSparseMLP {
    token_major: QuantizedSparseMLPTokenMajorKernels,
    expert_major: QuantizedSparseMLPExpertMajorKernels,
}

impl QuantizedSparseMLP {
    fn validate_input(&self) {}

    pub fn new(device: &Device, config: QuantizedSparseMLPConfig) -> Self {
        Self {
            token_major: QuantizedSparseMLPTokenMajorKernels::new(device, config),
            expert_major: QuantizedSparseMLPExpertMajorKernels::new(device, config),
        }
    }

    pub fn token_major(&self) -> &QuantizedSparseMLPTokenMajorKernels {
        &self.token_major
    }

    pub fn invoke_token_major<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPTokenMajorScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorInvocation<'a> {
        self.validate_input();
        self.token_major.invoke(shape, buffers, scratch, weights)
    }

    pub fn invoke_token_major_fused_gate_up_silu<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPTokenMajorScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorFusedGateUpSiluInvocation<'a> {
        self.validate_input();
        self.token_major
            .invoke_fused_gate_up_silu(shape, buffers, scratch, weights)
    }

    pub fn invoke_token_major_down<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPTokenMajorScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorDownInvocation<'a> {
        self.validate_input();
        self.token_major.invoke_down(shape, buffers, scratch, weights)
    }

    pub fn invoke_expert_major<'a>(
        &'a self,
        shape: QuantizedSparseMLPExpertMajorShape,
        buffers: QuantizedSparseMLPExpertMajorBuffers<'a>,
        scratch: QuantizedSparseMLPExpertMajorScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPExpertMajorInvocation<'a> {
        self.validate_input();
        self.expert_major.invoke(shape, buffers, scratch, weights)
    }
}

pub struct QuantizedSparseMLPTokenMajorKernels {
    config: QuantizedSparseMLPConfig,
    fused_gate_up_silu: GatherAffineQuantizedGateUpSiluKernel,
    down: GatherAffineQuantizedMatmulKernel,
}

pub struct QuantizedSparseMLPExpertMajorKernels {
    config: QuantizedSparseMLPConfig,
    fused_gate_up_silu: RaggedExpertMajorAffineQuantizedGateUpSiluKernel,
    down: RaggedExpertMajorAffineQuantizedMatmulKernel,
}

impl QuantizedSparseMLPTokenMajorKernels {
    pub fn new(device: &Device, config: QuantizedSparseMLPConfig) -> Self {
        config.validate();
        Self {
            config,
            fused_gate_up_silu: GatherAffineQuantizedGateUpSiluKernel::new(
                device,
                config.token_major_compile_fused_gate_up_silu_shape(),
            ),
            down: GatherAffineQuantizedMatmulKernel::new(
                device,
                config.token_major_compile_shape(config.hidden_dim, config.intermediate_dim),
            ),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPTokenMajorScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorInvocation<'a> {
        QuantizedSparseMLPTokenMajorInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
        }
    }

    pub fn invoke_fused_gate_up_silu<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPTokenMajorScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorFusedGateUpSiluInvocation<'a> {
        QuantizedSparseMLPTokenMajorFusedGateUpSiluInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
        }
    }

    pub fn invoke_down<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPTokenMajorScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorDownInvocation<'a> {
        QuantizedSparseMLPTokenMajorDownInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
        }
    }
}

impl QuantizedSparseMLPExpertMajorKernels {
    pub fn new(device: &Device, config: QuantizedSparseMLPConfig) -> Self {
        config.validate();
        let compile_shape = QuantizedSparseMLPExpertMajorShape {
            num_experts: 1,
            num_routes: 1,
        };
        Self {
            config,
            fused_gate_up_silu: RaggedExpertMajorAffineQuantizedGateUpSiluKernel::new(
                device,
                config.expert_major_fused_gate_up_silu_shape(compile_shape),
            ),
            down: RaggedExpertMajorAffineQuantizedMatmulKernel::new(
                device,
                config.expert_major_down_shape(compile_shape),
            ),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: QuantizedSparseMLPExpertMajorShape,
        buffers: QuantizedSparseMLPExpertMajorBuffers<'a>,
        scratch: QuantizedSparseMLPExpertMajorScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPExpertMajorInvocation<'a> {
        QuantizedSparseMLPExpertMajorInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
        }
    }
}

pub struct QuantizedSparseMLPTokenMajorInvocation<'a> {
    kernels: &'a QuantizedSparseMLPTokenMajorKernels,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
    scratch: QuantizedSparseMLPTokenMajorScratch<'a>,
    weights: QuantizedSparseMLPWeights<'a>,
}

impl Operator for QuantizedSparseMLPTokenMajorInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.kernels
            .invoke_fused_gate_up_silu(self.shape, self.buffers, self.scratch, self.weights)
            .record(builder);
        builder.record_with_barrier_before(self.kernels.invoke_down(
            self.shape,
            self.buffers,
            self.scratch,
            self.weights,
        ));
    }
}

pub struct QuantizedSparseMLPTokenMajorFusedGateUpSiluInvocation<'a> {
    kernels: &'a QuantizedSparseMLPTokenMajorKernels,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
    scratch: QuantizedSparseMLPTokenMajorScratch<'a>,
    weights: QuantizedSparseMLPWeights<'a>,
}

impl Operator for QuantizedSparseMLPTokenMajorFusedGateUpSiluInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        debug_validate_token_major_buffers(self.kernels.config, self.shape, &self.buffers, &self.scratch);
        self.kernels
            .fused_gate_up_silu
            .invoke_with_shape(
                self.kernels
                    .config
                    .token_major_fused_gate_up_silu_shape_unchecked(self.shape),
                self.scratch.activation,
                self.buffers.input,
                self.weights.gate_weight,
                self.weights.gate_scales,
                self.weights.gate_biases,
                self.weights.up_weight,
                self.weights.up_scales,
                self.weights.up_biases,
                self.buffers.token_indices,
                self.buffers.expert_indices,
            )
            .record(builder);
    }
}

pub struct QuantizedSparseMLPTokenMajorDownInvocation<'a> {
    kernels: &'a QuantizedSparseMLPTokenMajorKernels,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
    scratch: QuantizedSparseMLPTokenMajorScratch<'a>,
    weights: QuantizedSparseMLPWeights<'a>,
}

impl Operator for QuantizedSparseMLPTokenMajorDownInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        debug_validate_token_major_buffers(self.kernels.config, self.shape, &self.buffers, &self.scratch);
        self.kernels
            .down
            .invoke_with_shape(
                self.kernels.config.token_major_down_shape_unchecked(self.shape),
                self.buffers.output,
                self.scratch.activation,
                self.weights.down_weight,
                self.weights.down_scales,
                self.weights.down_biases,
                self.buffers.route_indices,
                self.buffers.expert_indices,
            )
            .record(builder);
    }
}

pub struct QuantizedSparseMLPExpertMajorInvocation<'a> {
    kernels: &'a QuantizedSparseMLPExpertMajorKernels,
    shape: QuantizedSparseMLPExpertMajorShape,
    buffers: QuantizedSparseMLPExpertMajorBuffers<'a>,
    scratch: QuantizedSparseMLPExpertMajorScratch<'a>,
    weights: QuantizedSparseMLPWeights<'a>,
}

impl Operator for QuantizedSparseMLPExpertMajorInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        debug_validate_expert_major_buffers(self.kernels.config, self.shape, &self.buffers, &self.scratch);
        self.kernels
            .fused_gate_up_silu
            .invoke_with_shape(
                self.kernels.config.expert_major_fused_gate_up_silu_shape(self.shape),
                self.scratch.activation,
                self.buffers.packed_input,
                self.weights.gate_weight,
                self.weights.gate_scales,
                self.weights.gate_biases,
                self.weights.up_weight,
                self.weights.up_scales,
                self.weights.up_biases,
                self.buffers.experts_by_route,
            )
            .record(builder);
        builder.record_with_barrier_before(self.kernels.down.invoke_with_shape(
            self.kernels.config.expert_major_down_shape(self.shape),
            self.buffers.route_output,
            self.scratch.activation,
            self.weights.down_weight,
            self.weights.down_scales,
            self.weights.down_biases,
            self.buffers.experts_by_route,
        ));
    }
}

fn debug_validate_expert_major_buffers(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPExpertMajorShape,
    buffers: &QuantizedSparseMLPExpertMajorBuffers<'_>,
    scratch: &QuantizedSparseMLPExpertMajorScratch<'_>,
) {
    #[cfg(debug_assertions)]
    validate_expert_major_buffers(config, shape, buffers, scratch);
}

fn debug_validate_token_major_buffers(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: &QuantizedSparseMLPTokenMajorBuffers<'_>,
    scratch: &QuantizedSparseMLPTokenMajorScratch<'_>,
) {
    #[cfg(debug_assertions)]
    validate_token_major_buffers(config, shape, buffers, scratch);
}

fn validate_expert_major_buffers(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPExpertMajorShape,
    buffers: &QuantizedSparseMLPExpertMajorBuffers<'_>,
    scratch: &QuantizedSparseMLPExpertMajorScratch<'_>,
) {
    shape.validate();
    let input_bytes = config.expert_major_input_bytes(shape);
    let route_indices_bytes = config.expert_major_route_indices_bytes(shape);
    let output_bytes = config.expert_major_output_bytes(shape);
    let activation_bytes = config.activation_bytes(shape.num_routes);
    assert!(
        buffers.packed_input.len_bytes() >= input_bytes,
        "sparse MLP expert-major packed input buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        input_bytes,
        buffers.packed_input.len_bytes()
    );
    assert!(
        buffers.experts_by_route.len_bytes() >= route_indices_bytes,
        "sparse MLP expert-major expert map buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        route_indices_bytes,
        buffers.experts_by_route.len_bytes()
    );
    assert!(
        buffers.route_output.len_bytes() >= output_bytes,
        "sparse MLP expert-major output buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.route_output.len_bytes()
    );
    assert!(
        scratch.activation.len_bytes() >= activation_bytes,
        "sparse MLP expert-major activation buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        activation_bytes,
        scratch.activation.len_bytes()
    );
}

fn validate_token_major_buffers(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: &QuantizedSparseMLPTokenMajorBuffers<'_>,
    scratch: &QuantizedSparseMLPTokenMajorScratch<'_>,
) {
    shape.validate();
    let input_bytes = config.token_major_input_bytes_unchecked(shape);
    let route_indices_bytes = config.token_major_route_indices_bytes_unchecked(shape);
    let output_bytes = config.token_major_output_bytes_unchecked(shape);
    assert!(
        buffers.input.len_bytes() >= input_bytes,
        "sparse MLP input buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        input_bytes,
        buffers.input.len_bytes()
    );
    assert!(
        buffers.token_indices.len_bytes() >= route_indices_bytes,
        "sparse MLP token index buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        route_indices_bytes,
        buffers.token_indices.len_bytes()
    );
    assert!(
        buffers.expert_indices.len_bytes() >= route_indices_bytes,
        "sparse MLP expert index buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        route_indices_bytes,
        buffers.expert_indices.len_bytes()
    );
    assert!(
        buffers.route_indices.len_bytes() >= route_indices_bytes,
        "sparse MLP route index buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        route_indices_bytes,
        buffers.route_indices.len_bytes()
    );
    assert!(
        buffers.output.len_bytes() >= output_bytes,
        "sparse MLP output buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.output.len_bytes()
    );
    validate_token_major_scratch(config, shape, scratch);
}

fn validate_token_major_scratch(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPTokenMajorShape,
    scratch: &QuantizedSparseMLPTokenMajorScratch<'_>,
) {
    let activation_bytes = config.activation_bytes_unchecked(shape.num_routes);
    assert!(
        scratch.activation.len_bytes() >= activation_bytes,
        "sparse MLP activation scratch buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        activation_bytes,
        scratch.activation.len_bytes()
    );
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use inference_executor_core::mlp::moe::reference::QuantizedSparseMLPReferenceWeights;
    use inference_executor_core::mlp::moe::reference::QuantizedSparseMLPTokenMajorReferenceInput;
    use inference_executor_core::mlp::moe::reference::moe_combine_without_common_bf16_reference;
    use inference_executor_core::mlp::moe::reference::quantized_sparse_mlp_token_major_reference;

    use super::*;
    use crate::components::MoEExpertMajorKernels;
    use crate::components::MoEExpertMajorLayoutBuffers;
    use crate::components::MoEExpertMajorPackInputBuffers;
    use crate::components::MoEExpertMajorScatterWithoutCommonBuffers;
    use crate::components::MoEExpertMajorShape;
    use crate::metal::Buffer;
    use crate::metal::Stream;

    #[test]
    fn test_token_major_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = QuantizedSparseMLPConfig {
            hidden_dim: 64,
            intermediate_dim: 64,
            group_size: 32,
            bits: 4,
            dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedSparseMLPTokenMajorShape {
            num_routes: 4,
            num_tokens: 2,
        };
        let fused_gate_up_silu_shape = config.token_major_fused_gate_up_silu_shape(shape);
        let down_shape = config.token_major_down_shape(shape);
        let num_experts = 4;
        let input_values = hidden_fixture(shape.num_tokens as usize, config.hidden_dim as usize);
        let input = bf16_buffer(&device, &input_values);
        let token_index_values = vec![0_u32, 0, 1, 1];
        let expert_index_values = vec![0_u32, 2, 1, 3];
        let route_index_values = vec![0_u32, 1, 2, 3];
        let token_indices = Buffer::from_slice(&device, &token_index_values);
        let expert_indices = Buffer::from_slice(&device, &expert_index_values);
        let route_indices = Buffer::from_slice(&device, &route_index_values);
        let gate_weight_values =
            quantized_weight_stack_values(num_experts, fused_gate_up_silu_shape.weight_bytes_per_expert());
        let gate_weight = Buffer::from_slice(&device, &gate_weight_values);
        let gate_scale_values = affine_param_fixture(
            num_experts * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
        );
        let gate_scales = bf16_buffer(&device, &gate_scale_values);
        let gate_bias_values =
            zero_fixture(num_experts * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>());
        let gate_biases = bf16_buffer(&device, &gate_bias_values);
        let up_weight_values =
            quantized_weight_stack_values(num_experts, fused_gate_up_silu_shape.weight_bytes_per_expert());
        let up_weight = Buffer::from_slice(&device, &up_weight_values);
        let up_scale_values = affine_param_fixture(
            num_experts * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
        );
        let up_scales = bf16_buffer(&device, &up_scale_values);
        let up_bias_values =
            zero_fixture(num_experts * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>());
        let up_biases = bf16_buffer(&device, &up_bias_values);
        let down_weight_values = quantized_weight_stack_values(num_experts, down_shape.weight_bytes_per_expert());
        let down_weight = Buffer::from_slice(&device, &down_weight_values);
        let down_scale_values =
            affine_param_fixture(num_experts * down_shape.affine_param_bytes_per_expert() / size_of::<u16>());
        let down_scales = bf16_buffer(&device, &down_scale_values);
        let down_bias_values =
            zero_fixture(num_experts * down_shape.affine_param_bytes_per_expert() / size_of::<u16>());
        let down_biases = bf16_buffer(&device, &down_bias_values);

        let actual_output = Buffer::new_zeroed(&device, config.token_major_output_bytes(shape));
        let actual_activation = Buffer::new_zeroed(&device, config.activation_bytes(shape.num_routes));
        let sparse_mlp = QuantizedSparseMLPTokenMajorKernels::new(&device, config);
        let mut builder = stream.create_replay_program();
        builder.record(sparse_mlp.invoke(
            shape,
            QuantizedSparseMLPTokenMajorBuffers {
                input: &input,
                token_indices: &token_indices,
                expert_indices: &expert_indices,
                route_indices: &route_indices,
                output: &actual_output,
            },
            QuantizedSparseMLPTokenMajorScratch {
                activation: &actual_activation,
            },
            QuantizedSparseMLPWeights {
                gate_weight: &gate_weight,
                gate_scales: &gate_scales,
                gate_biases: &gate_biases,
                up_weight: &up_weight,
                up_scales: &up_scales,
                up_biases: &up_biases,
                down_weight: &down_weight,
                down_scales: &down_scales,
                down_biases: &down_biases,
            },
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let expected = quantized_sparse_mlp_token_major_reference(QuantizedSparseMLPTokenMajorReferenceInput {
            input: &bf16_values(&input_values),
            token_indices: &token_index_values,
            expert_indices: &expert_index_values,
            route_indices: &route_index_values,
            hidden_dim: config.hidden_dim as usize,
            intermediate_dim: config.intermediate_dim as usize,
            group_size: config.group_size as usize,
            bits: config.bits as usize,
            num_experts,
            weights: QuantizedSparseMLPReferenceWeights {
                gate_weight: &gate_weight_values,
                gate_scales: &bf16_values(&gate_scale_values),
                gate_biases: &bf16_values(&gate_bias_values),
                up_weight: &up_weight_values,
                up_scales: &bf16_values(&up_scale_values),
                up_biases: &bf16_values(&up_bias_values),
                down_weight: &down_weight_values,
                down_scales: &bf16_values(&down_scale_values),
                down_biases: &bf16_values(&down_bias_values),
            },
        })
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
        assert_bf16_close_rel_values(
            &expected,
            &actual_output,
            config.token_major_output_bytes(shape),
            2.0e-5,
            8.0e-3,
        );
    }

    #[test]
    fn test_token_major_random() {
        let random_seed = 0x3A7D_C921;
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = QuantizedSparseMLPConfig {
            hidden_dim: 64,
            intermediate_dim: 64,
            group_size: 32,
            bits: 4,
            dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedSparseMLPTokenMajorShape {
            num_routes: 6,
            num_tokens: 3,
        };
        let num_experts = 5;
        let fused_gate_up_silu_shape = config.token_major_fused_gate_up_silu_shape(shape);
        let down_shape = config.token_major_down_shape(shape);
        let input_values = generated_values(shape.num_tokens as usize * config.hidden_dim as usize, random_seed);
        let input = bf16_buffer(&device, &input_values);
        let token_index_values = generated_indices(
            shape.num_routes as usize,
            shape.num_tokens as usize,
            random_seed.wrapping_add(1),
        );
        let expert_index_values =
            generated_indices(shape.num_routes as usize, num_experts, random_seed.wrapping_add(2));
        let route_index_values = identity_indices(shape.num_routes as usize);
        let token_indices = Buffer::from_slice(&device, &token_index_values);
        let expert_indices = Buffer::from_slice(&device, &expert_index_values);
        let route_indices = Buffer::from_slice(&device, &route_index_values);
        let gate_weight_values = generated_bytes(
            num_experts * fused_gate_up_silu_shape.weight_bytes_per_expert(),
            random_seed.wrapping_add(3),
        );
        let gate_weight = Buffer::from_slice(&device, &gate_weight_values);
        let gate_scale_values = generated_scales(
            num_experts * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(4),
        );
        let gate_scales = bf16_buffer(&device, &gate_scale_values);
        let gate_bias_values = generated_biases(
            num_experts * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(5),
        );
        let gate_biases = bf16_buffer(&device, &gate_bias_values);
        let up_weight_values = generated_bytes(
            num_experts * fused_gate_up_silu_shape.weight_bytes_per_expert(),
            random_seed.wrapping_add(6),
        );
        let up_weight = Buffer::from_slice(&device, &up_weight_values);
        let up_scale_values = generated_scales(
            num_experts * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(7),
        );
        let up_scales = bf16_buffer(&device, &up_scale_values);
        let up_bias_values = generated_biases(
            num_experts * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(8),
        );
        let up_biases = bf16_buffer(&device, &up_bias_values);
        let down_weight_values = generated_bytes(
            num_experts * down_shape.weight_bytes_per_expert(),
            random_seed.wrapping_add(9),
        );
        let down_weight = Buffer::from_slice(&device, &down_weight_values);
        let down_scale_values = generated_scales(
            num_experts * down_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(10),
        );
        let down_scales = bf16_buffer(&device, &down_scale_values);
        let down_bias_values = generated_biases(
            num_experts * down_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(11),
        );
        let down_biases = bf16_buffer(&device, &down_bias_values);

        let actual_output = Buffer::new_zeroed(&device, config.token_major_output_bytes(shape));
        let actual_activation = Buffer::new_zeroed(&device, config.activation_bytes(shape.num_routes));
        let sparse_mlp = QuantizedSparseMLPTokenMajorKernels::new(&device, config);
        let mut builder = stream.create_replay_program();
        builder.record(sparse_mlp.invoke(
            shape,
            QuantizedSparseMLPTokenMajorBuffers {
                input: &input,
                token_indices: &token_indices,
                expert_indices: &expert_indices,
                route_indices: &route_indices,
                output: &actual_output,
            },
            QuantizedSparseMLPTokenMajorScratch {
                activation: &actual_activation,
            },
            QuantizedSparseMLPWeights {
                gate_weight: &gate_weight,
                gate_scales: &gate_scales,
                gate_biases: &gate_biases,
                up_weight: &up_weight,
                up_scales: &up_scales,
                up_biases: &up_biases,
                down_weight: &down_weight,
                down_scales: &down_scales,
                down_biases: &down_biases,
            },
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let expected = quantized_sparse_mlp_token_major_reference(QuantizedSparseMLPTokenMajorReferenceInput {
            input: &bf16_values(&input_values),
            token_indices: &token_index_values,
            expert_indices: &expert_index_values,
            route_indices: &route_index_values,
            hidden_dim: config.hidden_dim as usize,
            intermediate_dim: config.intermediate_dim as usize,
            group_size: config.group_size as usize,
            bits: config.bits as usize,
            num_experts,
            weights: QuantizedSparseMLPReferenceWeights {
                gate_weight: &gate_weight_values,
                gate_scales: &bf16_values(&gate_scale_values),
                gate_biases: &bf16_values(&gate_bias_values),
                up_weight: &up_weight_values,
                up_scales: &bf16_values(&up_scale_values),
                up_biases: &bf16_values(&up_bias_values),
                down_weight: &down_weight_values,
                down_scales: &bf16_values(&down_scale_values),
                down_biases: &bf16_values(&down_bias_values),
            },
        })
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
        assert_bf16_close_rel_values(
            &expected,
            &actual_output,
            config.token_major_output_bytes(shape),
            2.0e-5,
            8.0e-3,
        );
    }

    #[test]
    fn test_expert_major_fixed() {
        assert_expert_major_pipeline_matches_reference(0x51a7_2026);
    }

    #[test]
    fn test_expert_major_random() {
        let random_seed = 0xB650_2FE8;
        assert_expert_major_pipeline_matches_reference(random_seed);
    }

    #[test]
    fn test_shapes() {
        let config = QuantizedSparseMLPConfig {
            hidden_dim: 64,
            intermediate_dim: 128,
            group_size: 32,
            bits: 4,
            dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedSparseMLPTokenMajorShape {
            num_routes: 6,
            num_tokens: 3,
        };

        assert_eq!(config.token_major_input_bytes(shape), 3 * 64 * 2);
        assert_eq!(config.token_major_route_indices_bytes(shape), 6 * size_of::<u32>());
        assert_eq!(config.activation_bytes(shape.num_routes), 6 * 128 * 2);
        assert_eq!(config.token_major_output_bytes(shape), 6 * 64 * 2);

        let fused_gate_up_silu = config.token_major_fused_gate_up_silu_shape(shape);
        assert_eq!(fused_gate_up_silu.num_routes, 6);
        assert_eq!(fused_gate_up_silu.num_input_vectors, 3);
        assert_eq!(fused_gate_up_silu.intermediate_dim, 128);
        assert_eq!(fused_gate_up_silu.k, 64);
        assert_eq!(fused_gate_up_silu.group_size, 32);
        assert_eq!(fused_gate_up_silu.bits, 4);
        assert_eq!(fused_gate_up_silu.dtype, Dtype::Bfloat16);

        let down = config.token_major_down_shape(shape);
        assert_eq!(down.num_routes, 6);
        assert_eq!(down.num_input_vectors, 6);
        assert_eq!(down.n, 64);
        assert_eq!(down.k, 128);
        assert_eq!(down.group_size, 32);
        assert_eq!(down.bits, 4);
        assert_eq!(down.dtype, Dtype::Bfloat16);
    }

    fn assert_expert_major_pipeline_matches_reference(random_seed: u32) {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = QuantizedSparseMLPConfig {
            hidden_dim: 64,
            intermediate_dim: 64,
            group_size: 32,
            bits: 4,
            dtype: Dtype::Bfloat16,
        };
        let num_tokens = 3_u32;
        let num_experts_per_token = 2_u32;
        let num_experts = 5_u32;
        let num_routes = num_tokens * num_experts_per_token;
        let layout_shape = MoEExpertMajorShape::bf16(num_tokens, num_experts, num_experts_per_token, config.hidden_dim);
        let sparse_shape = QuantizedSparseMLPExpertMajorShape {
            num_experts,
            num_routes,
        };
        let fused_gate_up_silu_shape = config.expert_major_fused_gate_up_silu_shape(sparse_shape);
        let down_shape = config.expert_major_down_shape(sparse_shape);

        let input_values = generated_values(num_tokens as usize * config.hidden_dim as usize, random_seed);
        let expert_index_values =
            generated_indices(num_routes as usize, num_experts as usize, random_seed.wrapping_add(1));
        let routed_prob_values = generated_probs(
            num_tokens as usize,
            num_experts_per_token as usize,
            random_seed.wrapping_add(2),
        );
        let token_index_values = (0..num_routes)
            .map(|route_index| route_index / num_experts_per_token)
            .collect::<Vec<_>>();
        let route_index_values = identity_indices(num_routes as usize);
        let gate_weight_values = generated_bytes(
            num_experts as usize * fused_gate_up_silu_shape.weight_bytes_per_expert(),
            random_seed.wrapping_add(3),
        );
        let gate_scale_values = generated_scales(
            num_experts as usize * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(4),
        );
        let gate_bias_values = generated_biases(
            num_experts as usize * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(5),
        );
        let up_weight_values = generated_bytes(
            num_experts as usize * fused_gate_up_silu_shape.weight_bytes_per_expert(),
            random_seed.wrapping_add(6),
        );
        let up_scale_values = generated_scales(
            num_experts as usize * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(7),
        );
        let up_bias_values = generated_biases(
            num_experts as usize * fused_gate_up_silu_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(8),
        );
        let down_weight_values = generated_bytes(
            num_experts as usize * down_shape.weight_bytes_per_expert(),
            random_seed.wrapping_add(9),
        );
        let down_scale_values = generated_scales(
            num_experts as usize * down_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(10),
        );
        let down_bias_values = generated_biases(
            num_experts as usize * down_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
            random_seed.wrapping_add(11),
        );

        let input = bf16_buffer(&device, &input_values);
        let expert_indices = Buffer::from_slice(&device, &expert_index_values);
        let routed_probs = Buffer::from_slice(&device, &routed_prob_values);
        let expert_counts = Buffer::new_zeroed(&device, layout_shape.expert_counts_bytes());
        let expert_offsets = Buffer::new_zeroed(&device, layout_shape.expert_offsets_bytes());
        let expert_cursors = Buffer::new_zeroed(&device, layout_shape.expert_counts_bytes());
        let routes_by_expert = Buffer::new_zeroed(&device, layout_shape.route_indices_bytes());
        let routes_by_token = Buffer::new_zeroed(&device, layout_shape.route_indices_bytes());
        let experts_by_route = Buffer::new_zeroed(&device, layout_shape.route_indices_bytes());
        let packed_input = Buffer::new_zeroed(&device, layout_shape.route_hidden_bytes());
        let route_output = Buffer::new_zeroed(&device, config.expert_major_output_bytes(sparse_shape));
        let activation = Buffer::new_zeroed(&device, config.activation_bytes(num_routes));
        let output = Buffer::new_zeroed(&device, layout_shape.token_hidden_bytes());
        let gate_weight = Buffer::from_slice(&device, &gate_weight_values);
        let gate_scales = bf16_buffer(&device, &gate_scale_values);
        let gate_biases = bf16_buffer(&device, &gate_bias_values);
        let up_weight = Buffer::from_slice(&device, &up_weight_values);
        let up_scales = bf16_buffer(&device, &up_scale_values);
        let up_biases = bf16_buffer(&device, &up_bias_values);
        let down_weight = Buffer::from_slice(&device, &down_weight_values);
        let down_scales = bf16_buffer(&device, &down_scale_values);
        let down_biases = bf16_buffer(&device, &down_bias_values);

        let layout = MoEExpertMajorKernels::new(&device);
        let sparse_mlp = QuantizedSparseMLP::new(&device, config);
        let weights = QuantizedSparseMLPWeights {
            gate_weight: &gate_weight,
            gate_scales: &gate_scales,
            gate_biases: &gate_biases,
            up_weight: &up_weight,
            up_scales: &up_scales,
            up_biases: &up_biases,
            down_weight: &down_weight,
            down_scales: &down_scales,
            down_biases: &down_biases,
        };
        let mut builder = stream.create_replay_program();
        builder.record(layout.invoke_layout(
            layout_shape,
            MoEExpertMajorLayoutBuffers {
                expert_indices: &expert_indices,
                expert_counts: &expert_counts,
                expert_offsets: &expert_offsets,
                expert_cursors: &expert_cursors,
                routes_by_expert: &routes_by_expert,
                routes_by_token: &routes_by_token,
                experts_by_route: &experts_by_route,
            },
        ));
        builder.record_with_barrier_before(layout.invoke_pack_input(
            layout_shape,
            MoEExpertMajorPackInputBuffers {
                input: &input,
                routes_by_expert: &routes_by_expert,
                packed_input: &packed_input,
            },
        ));
        builder.record_with_barrier_before(sparse_mlp.invoke_expert_major(
            sparse_shape,
            QuantizedSparseMLPExpertMajorBuffers {
                packed_input: &packed_input,
                experts_by_route: &experts_by_route,
                route_output: &route_output,
            },
            QuantizedSparseMLPExpertMajorScratch {
                activation: &activation,
            },
            weights,
        ));
        builder.record_with_barrier_before(layout.invoke_scatter_without_common(
            layout_shape,
            MoEExpertMajorScatterWithoutCommonBuffers {
                route_output: &route_output,
                routes_by_token: &routes_by_token,
                routed_probs: &routed_probs,
                output: &output,
            },
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let routed_hidden = quantized_sparse_mlp_token_major_reference(QuantizedSparseMLPTokenMajorReferenceInput {
            input: &bf16_values(&input_values),
            token_indices: &token_index_values,
            expert_indices: &expert_index_values,
            route_indices: &route_index_values,
            hidden_dim: config.hidden_dim as usize,
            intermediate_dim: config.intermediate_dim as usize,
            group_size: config.group_size as usize,
            bits: config.bits as usize,
            num_experts: num_experts as usize,
            weights: QuantizedSparseMLPReferenceWeights {
                gate_weight: &gate_weight_values,
                gate_scales: &bf16_values(&gate_scale_values),
                gate_biases: &bf16_values(&gate_bias_values),
                up_weight: &up_weight_values,
                up_scales: &bf16_values(&up_scale_values),
                up_biases: &bf16_values(&up_bias_values),
                down_weight: &down_weight_values,
                down_scales: &bf16_values(&down_scale_values),
                down_biases: &bf16_values(&down_bias_values),
            },
        })
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
        let expected_bits = moe_combine_without_common_bf16_reference(
            &routed_hidden,
            &routed_prob_values,
            num_tokens as usize,
            num_experts_per_token as usize,
            config.hidden_dim as usize,
        );
        let expected = expected_bits
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        assert_bf16_close_rel_values(&expected, &output, layout_shape.token_hidden_bytes(), 2.0e-5, 8.0e-3);
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits = values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        Buffer::from_slice(device, &bits)
    }

    fn assert_bf16_close_rel_values(
        expected: &[f32],
        actual: &Buffer,
        len_bytes: usize,
        abs_tolerance: f32,
        rel_tolerance: f32,
    ) {
        let actual = actual.read_typed::<u16>(0, len_bytes / size_of::<u16>());
        assert_eq!(expected.len(), actual.len());
        let mut max_abs_diff = 0.0_f32;
        let mut max_rel_diff = 0.0_f32;
        let mut max_index = 0;
        let mut max_expected = 0.0_f32;
        let mut max_actual = 0.0_f32;
        for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
            let expected = *expected;
            let actual = bf16::from_bits(*actual).to_f32();
            let diff = (actual - expected).abs();
            let rel_diff = if expected == 0.0 { diff } else { diff / expected.abs() };
            if diff > max_abs_diff {
                max_abs_diff = diff;
                max_rel_diff = rel_diff;
                max_index = index;
                max_expected = expected;
                max_actual = actual;
            }
            let tolerance = abs_tolerance.max(expected.abs() * rel_tolerance);
            assert!(
                diff <= tolerance,
                "fused sparse MLP output mismatch at {index}: expected={expected} actual={actual} diff={diff} \
                 tolerance={tolerance} abs_tolerance={abs_tolerance} rel_tolerance={rel_tolerance}"
            );
        }
        eprintln!(
            "fused sparse MLP max_abs_diff={max_abs_diff} max_rel_diff={max_rel_diff} index={max_index} \
             expected={max_expected} actual={max_actual}"
        );
    }

    fn hidden_fixture(num_tokens: usize, hidden_dim: usize) -> Vec<f32> {
        (0..num_tokens * hidden_dim)
            .map(|index| ((index * 13 + 5) % 31) as f32 * 0.0625 - 1.0)
            .collect()
    }

    fn quantized_weight_stack_values(num_experts: usize, bytes_per_expert: usize) -> Vec<u8> {
        let total_bytes = num_experts * bytes_per_expert;
        (0..total_bytes).map(|index| ((index * 13 + 17) & 0xff) as u8).collect()
    }

    fn identity_indices(len: usize) -> Vec<u32> {
        (0..len)
            .map(|index| u32::try_from(index).expect("identity index must fit u32"))
            .collect()
    }

    fn affine_param_fixture(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| 0.001 + ((index * 3) % 7) as f32 * 0.0001)
            .collect()
    }

    fn zero_fixture(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn bf16_values(values: &[f32]) -> Vec<f32> {
        values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
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

    fn generated_indices(count: usize, upper_bound: usize, random_seed: u32) -> Vec<u32> {
        assert!(upper_bound > 0);
        let mut state = random_seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(22_695_477).wrapping_add(1);
                (state as usize % upper_bound) as u32
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
}
