use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::operators::AffineQuantizedMatmulConfig;
use crate::operators::ExpertAffineQuantizedConfig;
use crate::operators::GatherAffineQuantizedGateUpSwiGLUKernel;
use crate::operators::GatherAffineQuantizedMatmulKernel;
use crate::operators::GatherAffineQuantizedShape;
use crate::operators::RaggedExpertMajorAffineQuantizedGateUpSwiGLUKernel;
use crate::operators::RaggedExpertMajorAffineQuantizedMatmulKernel;
use crate::operators::RaggedExpertMajorAffineQuantizedShape;

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
    pub num_experts: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub dtype: Dtype,
}

impl QuantizedSparseMLPConfig {
    pub fn validate(self) {
        assert!(self.num_experts > 0);
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
        i32::try_from(self.num_experts).expect("sparse MLP expert count must fit i32");
        i32::try_from(self.intermediate_dim).expect("sparse MLP intermediate_dim must fit i32");
        i32::try_from(self.stacked_intermediate_dim()).expect("sparse MLP stacked intermediate_dim must fit i32");
        i32::try_from(self.group_size).expect("sparse MLP group_size must fit i32");
        i32::try_from(self.bits).expect("sparse MLP bits must fit i32");
    }

    pub fn gate_up_config(self) -> ExpertAffineQuantizedConfig {
        self.expert_affine_config(self.intermediate_dim, self.hidden_dim)
    }

    pub fn down_config(self) -> ExpertAffineQuantizedConfig {
        self.expert_affine_config(self.hidden_dim, self.intermediate_dim)
    }

    fn expert_affine_config(self, n: u32, k: u32) -> ExpertAffineQuantizedConfig {
        self.validate();
        ExpertAffineQuantizedConfig {
            num_experts: to_i32(self.num_experts, "sparse MLP expert count"),
            matmul: AffineQuantizedMatmulConfig::same_dtype(
                to_i32(n, "sparse MLP output dimension"),
                to_i32(k, "sparse MLP input dimension"),
                to_i32(self.group_size, "sparse MLP group size"),
                to_i32(self.bits, "sparse MLP bits"),
                self.dtype,
            ),
        }
    }

    fn token_major_gate_up_shape(self, shape: QuantizedSparseMLPTokenMajorShape) -> GatherAffineQuantizedShape {
        shape.validate();
        GatherAffineQuantizedShape {
            num_routes: to_i32(shape.num_total_routes, "sparse MLP route count"),
            num_input_vectors: to_i32(shape.num_total_tokens, "sparse MLP token count"),
        }
    }

    fn token_major_down_shape(self, shape: QuantizedSparseMLPTokenMajorShape) -> GatherAffineQuantizedShape {
        shape.validate();
        GatherAffineQuantizedShape {
            num_routes: to_i32(shape.num_total_routes, "sparse MLP route count"),
            num_input_vectors: to_i32(shape.num_total_routes, "sparse MLP route count"),
        }
    }

    fn expert_major_affine_shape(
        self,
        shape: QuantizedSparseMLPExpertMajorShape,
    ) -> RaggedExpertMajorAffineQuantizedShape {
        shape.validate();
        RaggedExpertMajorAffineQuantizedShape {
            num_routes: to_i32(shape.num_total_routes, "sparse MLP route count"),
        }
    }

    pub fn token_major_input_bytes(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        self.validate();
        shape.validate();
        self.token_major_input_bytes_unchecked(shape)
    }

    fn token_major_input_bytes_unchecked(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        checked_bytes(
            "sparse MLP token-major input",
            &[shape.num_total_tokens as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    pub fn token_major_route_indices_bytes(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        shape.validate();
        self.token_major_route_indices_bytes_unchecked(shape)
    }

    fn token_major_route_indices_bytes_unchecked(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        (shape.num_total_routes as usize)
            .checked_mul(size_of::<u32>())
            .expect("sparse MLP route-index byte length must fit usize")
    }

    pub fn swiglu_bytes(self, num_routes: u32) -> usize {
        self.validate();
        assert!(num_routes > 0);
        self.swiglu_bytes_unchecked(num_routes)
    }

    fn swiglu_bytes_unchecked(self, num_routes: u32) -> usize {
        checked_bytes(
            "sparse MLP swiglu",
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
            &[shape.num_total_routes as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    pub fn expert_major_output_bytes(self, shape: QuantizedSparseMLPExpertMajorShape) -> usize {
        self.validate();
        shape.validate();
        checked_bytes(
            "sparse MLP expert-major output",
            &[shape.num_total_routes as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    pub fn expert_major_route_indices_bytes(self, shape: QuantizedSparseMLPExpertMajorShape) -> usize {
        shape.validate();
        (shape.num_total_routes as usize)
            .checked_mul(size_of::<u32>())
            .expect("sparse MLP expert-major route-index byte length must fit usize")
    }

    fn token_major_output_bytes_unchecked(self, shape: QuantizedSparseMLPTokenMajorShape) -> usize {
        checked_bytes(
            "sparse MLP token-major output",
            &[shape.num_total_routes as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    fn stacked_intermediate_dim(self) -> u32 {
        self.intermediate_dim
            .checked_mul(2)
            .expect("sparse MLP stacked gate/up dim must fit u32")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedSparseMLPTokenMajorShape {
    pub num_total_routes: u32,
    pub num_total_tokens: u32,
}

impl QuantizedSparseMLPTokenMajorShape {
    pub fn validate(self) {
        assert!(self.num_total_routes > 0);
        assert!(self.num_total_tokens > 0);
        to_i32(self.num_total_routes, "sparse MLP route count");
        to_i32(self.num_total_tokens, "sparse MLP token count");
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedSparseMLPExpertMajorShape {
    pub num_total_routes: u32,
}

impl QuantizedSparseMLPExpertMajorShape {
    pub fn validate(self) {
        to_i32(self.num_total_routes, "sparse MLP route count");
        assert!(self.num_total_routes > 0);
    }
}

#[derive(Clone, Copy)]
struct QuantizedSparseMLPBucketedReplay {
    num_total_tokens: u32,
    num_experts_per_token: u32,
    num_active_tokens_key: ReplayParameterKey,
}

impl QuantizedSparseMLPBucketedReplay {
    fn new(
        config: QuantizedSparseMLPConfig,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens_key: ReplayParameterKey,
    ) -> Self {
        config.validate();
        assert!(num_total_tokens > 0);
        assert!(num_experts_per_token > 0);
        assert!(
            num_experts_per_token <= config.num_experts,
            "sparse MLP experts per token must not exceed expert count"
        );
        let num_total_routes = num_total_tokens
            .checked_mul(num_experts_per_token)
            .expect("sparse MLP total route count must fit u32");
        i32::try_from(num_total_routes).expect("sparse MLP total route count must fit i32");
        Self {
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        }
    }

    fn num_total_routes(self) -> u32 {
        debug_assert!(self.num_total_tokens.checked_mul(self.num_experts_per_token).is_some());
        self.num_total_tokens * self.num_experts_per_token
    }

    fn token_major_shape(self) -> QuantizedSparseMLPTokenMajorShape {
        QuantizedSparseMLPTokenMajorShape {
            num_total_routes: self.num_total_routes(),
            num_total_tokens: self.num_total_tokens,
        }
    }

    fn expert_major_shape(self) -> QuantizedSparseMLPExpertMajorShape {
        QuantizedSparseMLPExpertMajorShape {
            num_total_routes: self.num_total_routes(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct QuantizedSparseMLPTokenMajorBuffers<'a> {
    pub input: &'a Buffer,
    pub token_indices: &'a Buffer,
    pub expert_indices: &'a Buffer,
    pub route_indices: &'a Buffer,
    pub routed_hidden: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QuantizedSparseMLPExpertMajorBuffers<'a> {
    pub packed_input: &'a Buffer,
    pub experts_by_route: &'a Buffer,
    pub packed_output: &'a Buffer,
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
pub struct QuantizedSparseMLPScratch<'a> {
    pub swiglu: &'a Buffer,
}

/// Records one quantized sparse MLP through either supported route layout:
///
/// ```text
/// TokenMajor
///
/// input [T, H] --(token_indices, expert_indices)--> gate_up_swiglu
///                                                       |
///                                                       v
///                                                swiglu [R, I]
///                                                       |
///                              (route_indices, expert_indices)
///                                                       |
///                                                       v
///                                             routed_hidden [R, H]
///
/// ExpertMajor
///
/// packed_input [R, H] --(experts_by_route)--> gate_up_swiglu
///                                                   |
///                                                   v
///                                            swiglu [R, I]
///                                                   |
///                                           experts_by_route
///                                                   |
///                                                   v
///                                         packed_output [R, H]
/// ```
pub struct QuantizedSparseMLP {
    token_major: QuantizedSparseMLPTokenMajorKernels,
    expert_major: QuantizedSparseMLPExpertMajorKernels,
}

impl QuantizedSparseMLP {
    pub fn new(device: &Device, config: QuantizedSparseMLPConfig) -> Self {
        Self {
            token_major: QuantizedSparseMLPTokenMajorKernels::new(device, config),
            expert_major: QuantizedSparseMLPExpertMajorKernels::new(device, config),
        }
    }

    pub fn invoke_token_major<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorInvocation<'a> {
        self.token_major.invoke(shape, buffers, scratch, weights)
    }

    /// Records a fixed token-major capacity whose active route count derives from active tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_token_major_bucketed<'a>(
        &'a self,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens_key: ReplayParameterKey,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorInvocation<'a> {
        let bucketed_replay = QuantizedSparseMLPBucketedReplay::new(
            self.token_major.config,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
        self.token_major
            .invoke_bucketed(bucketed_replay, buffers, scratch, weights)
    }

    pub fn invoke_expert_major<'a>(
        &'a self,
        shape: QuantizedSparseMLPExpertMajorShape,
        buffers: QuantizedSparseMLPExpertMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPExpertMajorInvocation<'a> {
        self.expert_major.invoke(shape, buffers, scratch, weights)
    }

    /// Records a fixed expert-major capacity whose active route count derives from active tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_expert_major_bucketed<'a>(
        &'a self,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens_key: ReplayParameterKey,
        buffers: QuantizedSparseMLPExpertMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPExpertMajorInvocation<'a> {
        let bucketed_replay = QuantizedSparseMLPBucketedReplay::new(
            self.expert_major.config,
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens_key,
        );
        self.expert_major
            .invoke_bucketed(bucketed_replay, buffers, scratch, weights)
    }
}

pub struct QuantizedSparseMLPTokenMajorKernels {
    config: QuantizedSparseMLPConfig,
    gate_up_swiglu: GatherAffineQuantizedGateUpSwiGLUKernel,
    down: GatherAffineQuantizedMatmulKernel,
}

pub struct QuantizedSparseMLPExpertMajorKernels {
    config: QuantizedSparseMLPConfig,
    gate_up_swiglu: RaggedExpertMajorAffineQuantizedGateUpSwiGLUKernel,
    down: RaggedExpertMajorAffineQuantizedMatmulKernel,
}

impl QuantizedSparseMLPTokenMajorKernels {
    pub fn new(device: &Device, config: QuantizedSparseMLPConfig) -> Self {
        config.validate();
        Self {
            config,
            gate_up_swiglu: GatherAffineQuantizedGateUpSwiGLUKernel::new(device, config.gate_up_config()),
            down: GatherAffineQuantizedMatmulKernel::new(device, config.down_config()),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorInvocation<'a> {
        QuantizedSparseMLPTokenMajorInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
            bucketed_replay: None,
        }
    }

    fn invoke_bucketed<'a>(
        &'a self,
        bucketed_replay: QuantizedSparseMLPBucketedReplay,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorInvocation<'a> {
        QuantizedSparseMLPTokenMajorInvocation {
            kernels: self,
            shape: bucketed_replay.token_major_shape(),
            buffers,
            scratch,
            weights,
            bucketed_replay: Some(bucketed_replay),
        }
    }

    pub fn invoke_gate_up_swiglu<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorGateUpSwiGLUInvocation<'a> {
        QuantizedSparseMLPTokenMajorGateUpSwiGLUInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
            bucketed_replay: None,
        }
    }

    pub fn invoke_down<'a>(
        &'a self,
        shape: QuantizedSparseMLPTokenMajorShape,
        buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPTokenMajorDownInvocation<'a> {
        QuantizedSparseMLPTokenMajorDownInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
            bucketed_replay: None,
        }
    }
}

impl QuantizedSparseMLPExpertMajorKernels {
    pub fn new(device: &Device, config: QuantizedSparseMLPConfig) -> Self {
        config.validate();
        Self {
            config,
            gate_up_swiglu: RaggedExpertMajorAffineQuantizedGateUpSwiGLUKernel::new(device, config.gate_up_config()),
            down: RaggedExpertMajorAffineQuantizedMatmulKernel::new(device, config.down_config()),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: QuantizedSparseMLPExpertMajorShape,
        buffers: QuantizedSparseMLPExpertMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPExpertMajorInvocation<'a> {
        QuantizedSparseMLPExpertMajorInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
            bucketed_replay: None,
        }
    }

    fn invoke_bucketed<'a>(
        &'a self,
        bucketed_replay: QuantizedSparseMLPBucketedReplay,
        buffers: QuantizedSparseMLPExpertMajorBuffers<'a>,
        scratch: QuantizedSparseMLPScratch<'a>,
        weights: QuantizedSparseMLPWeights<'a>,
    ) -> QuantizedSparseMLPExpertMajorInvocation<'a> {
        QuantizedSparseMLPExpertMajorInvocation {
            kernels: self,
            shape: bucketed_replay.expert_major_shape(),
            buffers,
            scratch,
            weights,
            bucketed_replay: Some(bucketed_replay),
        }
    }
}

pub struct QuantizedSparseMLPTokenMajorInvocation<'a> {
    kernels: &'a QuantizedSparseMLPTokenMajorKernels,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
    scratch: QuantizedSparseMLPScratch<'a>,
    weights: QuantizedSparseMLPWeights<'a>,
    bucketed_replay: Option<QuantizedSparseMLPBucketedReplay>,
}

impl Operator for QuantizedSparseMLPTokenMajorInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        QuantizedSparseMLPTokenMajorGateUpSwiGLUInvocation {
            kernels: self.kernels,
            shape: self.shape,
            buffers: self.buffers,
            scratch: self.scratch,
            weights: self.weights,
            bucketed_replay: self.bucketed_replay,
        }
        .record(recorder);
        recorder.record_with_barrier_before(QuantizedSparseMLPTokenMajorDownInvocation {
            kernels: self.kernels,
            shape: self.shape,
            buffers: self.buffers,
            scratch: self.scratch,
            weights: self.weights,
            bucketed_replay: self.bucketed_replay,
        });
    }
}

pub struct QuantizedSparseMLPTokenMajorGateUpSwiGLUInvocation<'a> {
    kernels: &'a QuantizedSparseMLPTokenMajorKernels,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
    scratch: QuantizedSparseMLPScratch<'a>,
    weights: QuantizedSparseMLPWeights<'a>,
    bucketed_replay: Option<QuantizedSparseMLPBucketedReplay>,
}

impl Operator for QuantizedSparseMLPTokenMajorGateUpSwiGLUInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        debug_validate_token_major_buffers(self.kernels.config, self.shape, &self.buffers, &self.scratch);
        let invocation = match self.bucketed_replay {
            Some(bucketed_replay) => {
                self.kernels.gate_up_swiglu.invoke_bucketed(
                    self.kernels.config.token_major_gate_up_shape(self.shape),
                    bucketed_replay.num_total_tokens,
                    bucketed_replay.num_experts_per_token,
                    bucketed_replay.num_active_tokens_key,
                    self.scratch.swiglu,
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
            },
            None => {
                self.kernels.gate_up_swiglu.invoke(
                    self.kernels.config.token_major_gate_up_shape(self.shape),
                    self.scratch.swiglu,
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
            },
        };
        invocation.record(recorder);
    }
}

pub struct QuantizedSparseMLPTokenMajorDownInvocation<'a> {
    kernels: &'a QuantizedSparseMLPTokenMajorKernels,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: QuantizedSparseMLPTokenMajorBuffers<'a>,
    scratch: QuantizedSparseMLPScratch<'a>,
    weights: QuantizedSparseMLPWeights<'a>,
    bucketed_replay: Option<QuantizedSparseMLPBucketedReplay>,
}

impl Operator for QuantizedSparseMLPTokenMajorDownInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        debug_validate_token_major_buffers(self.kernels.config, self.shape, &self.buffers, &self.scratch);
        let invocation = match self.bucketed_replay {
            Some(bucketed_replay) => {
                self.kernels.down.invoke_bucketed(
                    self.kernels.config.token_major_down_shape(self.shape),
                    bucketed_replay.num_total_tokens,
                    bucketed_replay.num_experts_per_token,
                    bucketed_replay.num_active_tokens_key,
                    self.buffers.routed_hidden,
                    self.scratch.swiglu,
                    self.weights.down_weight,
                    self.weights.down_scales,
                    self.weights.down_biases,
                    self.buffers.route_indices,
                    self.buffers.expert_indices,
                )
            },
            None => {
                self.kernels.down.invoke(
                    self.kernels.config.token_major_down_shape(self.shape),
                    self.buffers.routed_hidden,
                    self.scratch.swiglu,
                    self.weights.down_weight,
                    self.weights.down_scales,
                    self.weights.down_biases,
                    self.buffers.route_indices,
                    self.buffers.expert_indices,
                )
            },
        };
        invocation.record(recorder);
    }
}

pub struct QuantizedSparseMLPExpertMajorInvocation<'a> {
    kernels: &'a QuantizedSparseMLPExpertMajorKernels,
    shape: QuantizedSparseMLPExpertMajorShape,
    buffers: QuantizedSparseMLPExpertMajorBuffers<'a>,
    scratch: QuantizedSparseMLPScratch<'a>,
    weights: QuantizedSparseMLPWeights<'a>,
    bucketed_replay: Option<QuantizedSparseMLPBucketedReplay>,
}

impl Operator for QuantizedSparseMLPExpertMajorInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        debug_validate_expert_major_buffers(self.kernels.config, self.shape, &self.buffers, &self.scratch);
        let gate_up_swiglu = match self.bucketed_replay {
            Some(bucketed_replay) => {
                self.kernels.gate_up_swiglu.invoke_bucketed(
                    self.kernels.config.expert_major_affine_shape(self.shape),
                    bucketed_replay.num_total_tokens,
                    bucketed_replay.num_experts_per_token,
                    bucketed_replay.num_active_tokens_key,
                    self.scratch.swiglu,
                    self.buffers.packed_input,
                    self.weights.gate_weight,
                    self.weights.gate_scales,
                    self.weights.gate_biases,
                    self.weights.up_weight,
                    self.weights.up_scales,
                    self.weights.up_biases,
                    self.buffers.experts_by_route,
                )
            },
            None => {
                self.kernels.gate_up_swiglu.invoke(
                    self.kernels.config.expert_major_affine_shape(self.shape),
                    self.scratch.swiglu,
                    self.buffers.packed_input,
                    self.weights.gate_weight,
                    self.weights.gate_scales,
                    self.weights.gate_biases,
                    self.weights.up_weight,
                    self.weights.up_scales,
                    self.weights.up_biases,
                    self.buffers.experts_by_route,
                )
            },
        };
        gate_up_swiglu.record(recorder);
        let down = match self.bucketed_replay {
            Some(bucketed_replay) => {
                self.kernels.down.invoke_bucketed(
                    self.kernels.config.expert_major_affine_shape(self.shape),
                    bucketed_replay.num_total_tokens,
                    bucketed_replay.num_experts_per_token,
                    bucketed_replay.num_active_tokens_key,
                    self.buffers.packed_output,
                    self.scratch.swiglu,
                    self.weights.down_weight,
                    self.weights.down_scales,
                    self.weights.down_biases,
                    self.buffers.experts_by_route,
                )
            },
            None => {
                self.kernels.down.invoke(
                    self.kernels.config.expert_major_affine_shape(self.shape),
                    self.buffers.packed_output,
                    self.scratch.swiglu,
                    self.weights.down_weight,
                    self.weights.down_scales,
                    self.weights.down_biases,
                    self.buffers.experts_by_route,
                )
            },
        };
        recorder.record_with_barrier_before(down);
    }
}

fn debug_validate_expert_major_buffers(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPExpertMajorShape,
    buffers: &QuantizedSparseMLPExpertMajorBuffers<'_>,
    scratch: &QuantizedSparseMLPScratch<'_>,
) {
    #[cfg(debug_assertions)]
    validate_expert_major_buffers(config, shape, buffers, scratch);
}

fn debug_validate_token_major_buffers(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: &QuantizedSparseMLPTokenMajorBuffers<'_>,
    scratch: &QuantizedSparseMLPScratch<'_>,
) {
    #[cfg(debug_assertions)]
    validate_token_major_buffers(config, shape, buffers, scratch);
}

fn validate_expert_major_buffers(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPExpertMajorShape,
    buffers: &QuantizedSparseMLPExpertMajorBuffers<'_>,
    scratch: &QuantizedSparseMLPScratch<'_>,
) {
    shape.validate();
    let input_bytes = config.expert_major_input_bytes(shape);
    let route_indices_bytes = config.expert_major_route_indices_bytes(shape);
    let output_bytes = config.expert_major_output_bytes(shape);
    let swiglu_bytes = config.swiglu_bytes(shape.num_total_routes);
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
        buffers.packed_output.len_bytes() >= output_bytes,
        "sparse MLP expert-major output buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.packed_output.len_bytes()
    );
    assert!(
        scratch.swiglu.len_bytes() >= swiglu_bytes,
        "sparse MLP expert-major swiglu buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        swiglu_bytes,
        scratch.swiglu.len_bytes()
    );
}

fn validate_token_major_buffers(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: &QuantizedSparseMLPTokenMajorBuffers<'_>,
    scratch: &QuantizedSparseMLPScratch<'_>,
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
        buffers.routed_hidden.len_bytes() >= output_bytes,
        "sparse MLP output buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.routed_hidden.len_bytes()
    );
    validate_token_major_scratch(config, shape, scratch);
}

fn validate_token_major_scratch(
    config: QuantizedSparseMLPConfig,
    shape: QuantizedSparseMLPTokenMajorShape,
    scratch: &QuantizedSparseMLPScratch<'_>,
) {
    let swiglu_bytes = config.swiglu_bytes_unchecked(shape.num_total_routes);
    assert!(
        scratch.swiglu.len_bytes() >= swiglu_bytes,
        "sparse MLP swiglu scratch buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        swiglu_bytes,
        scratch.swiglu.len_bytes()
    );
}

#[cfg(test)]
#[path = "quantized_sparse_mlp_test.rs"]
mod tests;
