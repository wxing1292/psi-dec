use inference_backend_metal::components::BufferCastBuffers;
use inference_backend_metal::components::BufferCastKernel;
use inference_backend_metal::components::BufferCastShape;
use inference_backend_metal::components::GDNCoreBuffers;
use inference_backend_metal::components::GDNCoreConfig;
use inference_backend_metal::components::GDNCoreForwardCandidateStateUpdateBuffers;
use inference_backend_metal::components::GDNCoreKernels;
use inference_backend_metal::components::GDNCoreShape;
use inference_backend_metal::components::GDNProjectionSplitBuffers;
use inference_backend_metal::components::GDNProjectionSplitKernel;
use inference_backend_metal::components::GDNProjectionSplitShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::Layer;

use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::scratch::GDNScratchBindings;
use crate::attn::gdn::state_table::GDNPreparedRequestState;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GDNMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub recurrent_v_tile_size: u32,
    pub norm_eps: f32,
    pub input_dtype: Dtype,
    pub qkvabz_affine_dtype: Dtype,
    pub output_affine_dtype: Dtype,
}

impl GDNMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(self.recurrent_v_tile_size > 0);
        assert!(self.norm_eps > 0.0);
        assert!(matches!(self.input_dtype, Dtype::Bfloat16 | Dtype::Float32));
        assert!(matches!(self.qkvabz_affine_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(matches!(self.output_affine_dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn internal_dtype(self) -> Dtype {
        Dtype::Float32
    }

    pub fn boundary_dtype(self) -> Dtype {
        Dtype::Bfloat16
    }
}

#[derive(Clone, Copy)]
pub struct GDNLayerStateBindings<'a> {
    pub conv_state: &'a Buffer,
    pub conv_state_offset_bytes: u64,
    pub next_conv_state: &'a Buffer,
    pub next_conv_state_offset_bytes: u64,
    pub recurrent_state_arena: &'a Buffer,
    pub recurrent_state_arena_offset_bytes: u64,
}

#[derive(Clone, Copy)]
pub struct GDNWeights<'a> {
    pub qkvabz_weight: &'a Buffer,
    pub qkvabz_scales: &'a Buffer,
    pub qkvabz_biases: &'a Buffer,
    pub conv_weight: &'a Buffer,
    pub norm_weight: &'a Buffer,
    pub a_log_decay: &'a Buffer,
    pub dt_bias: &'a Buffer,
    pub output_weight: &'a Buffer,
    pub output_scales: &'a Buffer,
    pub output_biases: &'a Buffer,
}

/// Borrowed bindings for one GDN replay recording. The replay shape belongs to
/// `batch_metadata` and is read from it during recording.
#[derive(Clone, Copy)]
pub struct GDNInput<'a> {
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: GDNScratchBindings<'a>,
    pub batch_metadata: &'a GDNMetadataBuffers,
    pub state: GDNLayerStateBindings<'a>,
    pub materialize_candidate_states: bool,
    pub weights: GDNWeights<'a>,
}

/// The caller-owned next-hidden-state buffer returned by one GDN recording.
pub type GDNOutput<'a> = &'a Buffer;

pub struct GDN {
    core: GDNCore,
    config: GDNMetalConfig,
    qkvabz_projection_qmv: AffineQuantizedMatmulKernel,
    qkvabz_projection_qmm: AffineQuantizedMatmulKernel,
    projection_split: GDNProjectionSplitKernel,
    core_backend: GDNCoreBackend,
    cast_pre_output_hidden_states: BufferCastKernel,
    output_projection_qmv: AffineQuantizedMatmulKernel,
    output_projection_qmm: AffineQuantizedMatmulKernel,
}

impl GDN {
    pub fn new(device: &Device, core: GDNCore, config: GDNMetalConfig) -> Self {
        core.validate();
        config.validate();
        let qkvabz_dim = core.qkvabz_dim();
        let qkvabz_qmm_m = qmv_batch_limit(core.hidden_dim, qkvabz_dim);
        let output_qmm_m = qmv_batch_limit(core.v_dim(), core.hidden_dim);
        Self {
            core: core.clone(),
            config,
            qkvabz_projection_qmv: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape(
                    1,
                    qkvabz_dim,
                    core.hidden_dim,
                    config.input_dtype,
                    config.internal_dtype(),
                    config.qkvabz_affine_dtype,
                    config,
                ),
            ),
            qkvabz_projection_qmm: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape(
                    qkvabz_qmm_m,
                    qkvabz_dim,
                    core.hidden_dim,
                    config.input_dtype,
                    config.internal_dtype(),
                    config.qkvabz_affine_dtype,
                    config,
                ),
            ),
            projection_split: GDNProjectionSplitKernel::new(device),
            core_backend: GDNCoreBackend::new(device, core.clone(), config.recurrent_v_tile_size),
            cast_pre_output_hidden_states: BufferCastKernel::new(device),
            output_projection_qmv: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape(
                    1,
                    core.hidden_dim,
                    core.v_dim(),
                    config.boundary_dtype(),
                    config.boundary_dtype(),
                    config.output_affine_dtype,
                    config,
                ),
            ),
            output_projection_qmm: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape(
                    output_qmm_m,
                    core.hidden_dim,
                    core.v_dim(),
                    config.boundary_dtype(),
                    config.boundary_dtype(),
                    config.output_affine_dtype,
                    config,
                ),
            ),
        }
    }

    pub fn prepare(
        &self,
        metadata: &GDNMetadataBuffers,
        cu_tokens: &[u32],
        state: &GDNPreparedRequestState,
    ) -> GDNReplayShape {
        metadata.update(
            cu_tokens,
            &state.src_state_slots,
            &state.dst_state_slots,
            &state.flat_candidate_state_slots,
        )
    }
}

impl Layer for GDN {
    type Input<'a> = GDNInput<'a>;
    type Output<'a> = GDNOutput<'a>;

    type InputShape = GDNCore;
    type OutputShape = GDNCore;

    fn input_shape(&self) -> Self::InputShape {
        self.core.clone()
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.core.clone()
    }
}

impl ReplayLayer for GDN {
    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        input.batch_metadata.replay_shape().validate();
        let shape = input.batch_metadata.replay_shape();
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let scratch = input.scratch;
        let batch_metadata = input.batch_metadata;
        let state = input.state;
        let weights = input.weights;
        let qkvabz_projection_input = match self.config.input_dtype {
            Dtype::Bfloat16 => hidden_state,
            Dtype::Float32 => {
                recorder.record_with_barrier_before(ReplayOp::opaque(
                    self.cast_pre_output_hidden_states.invoke(
                        BufferCastShape::bf16_to_f32(
                            shape
                                .num_tokens
                                .checked_mul(self.core.hidden_dim.try_into().expect("GDN hidden_dim must fit u32"))
                                .expect("GDN hidden-state element count must fit u32"),
                        ),
                        BufferCastBuffers {
                            input: hidden_state,
                            output: scratch.hidden_state_f32,
                        },
                    ),
                ));
                scratch.hidden_state_f32
            },
            dtype => panic!("unsupported GDN input dtype {dtype:?}"),
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qkvabz_projection(shape).invoke_with_shape(
            self.qkvabz_shape(shape),
            scratch.qkvabz,
            0,
            qkvabz_projection_input,
            0,
            weights.qkvabz_weight,
            0,
            weights.qkvabz_scales,
            0,
            weights.qkvabz_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.projection_split.invoke(
            self.projection_split_shape(shape),
            GDNProjectionSplitBuffers {
                qkvabz: scratch.qkvabz,
                projected_qkv: scratch.projected_qkv,
                a: scratch.a,
                b: scratch.b,
                z: scratch.z,
            },
        )));
        let core_buffers = GDNCoreBuffers {
            projected_qkv: scratch.projected_qkv,
            a: scratch.a,
            b: scratch.b,
            z: scratch.z,
            conv_weight: weights.conv_weight,
            norm_weight: weights.norm_weight,
            a_log_decay: weights.a_log_decay,
            dt_bias: weights.dt_bias,
            cu_tokens: batch_metadata.cu_tokens(),
            src_state_slots: batch_metadata.src_state_slots(),
            dst_state_slots: batch_metadata.dst_state_slots(),
            conv_state: state.conv_state,
            conv_state_offset_bytes: state.conv_state_offset_bytes,
            next_conv_state: state.next_conv_state,
            next_conv_state_offset_bytes: state.next_conv_state_offset_bytes,
            recurrent_state_arena: state.recurrent_state_arena,
            recurrent_state_arena_offset_bytes: state.recurrent_state_arena_offset_bytes,
            conv_qkv: scratch.conv_qkv,
            recurrent_output: scratch.recurrent_output,
            pre_output_hidden_states: scratch.pre_output_hidden_states,
        };
        if input.materialize_candidate_states {
            recorder.record_with_barrier_before(ReplayOp::opaque(
                self.core_backend.kernels.invoke_forward_candidate_state_update(
                    self.core_backend.backend_shape(shape),
                    GDNCoreForwardCandidateStateUpdateBuffers {
                        core: core_buffers,
                        flat_candidate_state_slots: batch_metadata.flat_candidate_state_slots(),
                    },
                    self.core.q_scale,
                    self.config.norm_eps,
                ),
            ));
        } else {
            recorder.record_with_barrier_before(ReplayOp::opaque(self.core_backend.kernels.invoke(
                self.core_backend.backend_shape(shape),
                core_buffers,
                self.core.q_scale,
                self.config.norm_eps,
            )));
        }
        let output_projection_input = match self.config.boundary_dtype() {
            Dtype::Bfloat16 => {
                recorder.record_with_barrier_before(ReplayOp::opaque(
                    self.cast_pre_output_hidden_states.invoke(
                        BufferCastShape::f32_to_bf16(
                            shape
                                .num_tokens
                                .checked_mul(u32::try_from(self.core.v_dim()).expect("GDN v_dim must fit u32"))
                                .expect("GDN output element count must fit u32"),
                        ),
                        BufferCastBuffers {
                            input: scratch.pre_output_hidden_states,
                            output: scratch.pre_output_hidden_states_bf16,
                        },
                    ),
                ));
                scratch.pre_output_hidden_states_bf16
            },
            Dtype::Float32 => scratch.pre_output_hidden_states,
            dtype => panic!("unsupported GDN boundary dtype {dtype:?}"),
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(self.output_projection(shape).invoke_with_shape(
            self.output_shape(shape),
            next_hidden_state,
            0,
            output_projection_input,
            0,
            weights.output_weight,
            0,
            weights.output_scales,
            0,
            weights.output_biases,
            0,
        )));
        next_hidden_state
    }
}

impl GDN {
    fn qkvabz_projection(&self, shape: GDNReplayShape) -> &AffineQuantizedMatmulKernel {
        if shape.num_tokens >= qmv_batch_limit(self.core.hidden_dim, self.core.qkvabz_dim()) {
            &self.qkvabz_projection_qmm
        } else {
            &self.qkvabz_projection_qmv
        }
    }

    fn output_projection(&self, shape: GDNReplayShape) -> &AffineQuantizedMatmulKernel {
        if shape.num_tokens >= qmv_batch_limit(self.core.v_dim(), self.core.hidden_dim) {
            &self.output_projection_qmm
        } else {
            &self.output_projection_qmv
        }
    }

    fn qkvabz_shape(&self, shape: GDNReplayShape) -> AffineQuantizedMatmulShape {
        affine_shape(
            shape.num_tokens,
            self.core.qkvabz_dim(),
            self.core.hidden_dim,
            self.config.input_dtype,
            self.config.internal_dtype(),
            self.config.qkvabz_affine_dtype,
            self.config,
        )
    }

    fn output_shape(&self, shape: GDNReplayShape) -> AffineQuantizedMatmulShape {
        affine_shape(
            shape.num_tokens,
            self.core.hidden_dim,
            self.core.v_dim(),
            self.config.boundary_dtype(),
            self.config.boundary_dtype(),
            self.config.output_affine_dtype,
            self.config,
        )
    }

    fn projection_split_shape(&self, shape: GDNReplayShape) -> GDNProjectionSplitShape {
        match self.config.internal_dtype() {
            Dtype::Float32 => {
                GDNProjectionSplitShape::f32(
                    shape.num_tokens,
                    self.core.qkv_dim().try_into().expect("GDN qkv_dim must fit u32"),
                    self.core.num_v_heads.try_into().expect("GDN num_v_heads must fit u32"),
                    self.core.v_dim().try_into().expect("GDN v_dim must fit u32"),
                )
            },
            Dtype::Bfloat16 => {
                GDNProjectionSplitShape::bf16_to_f32(
                    shape.num_tokens,
                    self.core.qkv_dim().try_into().expect("GDN qkv_dim must fit u32"),
                    self.core.num_v_heads.try_into().expect("GDN num_v_heads must fit u32"),
                    self.core.v_dim().try_into().expect("GDN v_dim must fit u32"),
                )
            },
            dtype => panic!("unsupported GDN internal dtype {dtype:?}"),
        }
    }
}

struct GDNCoreBackend {
    core: GDNCore,
    kernels: GDNCoreKernels,
}

impl GDNCoreBackend {
    fn new(device: &Device, core: GDNCore, recurrent_v_dim_tile_size: u32) -> Self {
        core.validate();
        let config = GDNCoreConfig {
            num_qk_heads: core.num_qk_heads.try_into().expect("GDN query/key heads must fit u32"),
            qk_head_dim: core.qk_head_dim.try_into().expect("GDN qk_head_dim must fit u32"),
            num_v_heads: core.num_v_heads.try_into().expect("GDN num_v_heads must fit u32"),
            v_head_dim: core.v_head_dim.try_into().expect("GDN v_head_dim must fit u32"),
            conv_kernel_size: core
                .conv_kernel_size
                .try_into()
                .expect("GDN conv_kernel_size must fit u32"),
            v_dim_tile_size: recurrent_v_dim_tile_size,
        };
        Self {
            core,
            kernels: GDNCoreKernels::new(device, config),
        }
    }

    fn backend_shape(&self, shape: GDNReplayShape) -> GDNCoreShape {
        GDNCoreShape {
            num_reqs: shape.num_reqs,
            num_tokens: shape.num_tokens,
        }
    }
}

fn affine_shape(
    m: u32,
    n: usize,
    k: usize,
    input_dtype: Dtype,
    output_dtype: Dtype,
    affine_dtype: Dtype,
    config: GDNMetalConfig,
) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape {
        m: m.try_into().expect("GDN affine m must fit i32"),
        n: n.try_into().expect("GDN affine n must fit i32"),
        k: k.try_into().expect("GDN affine k must fit i32"),
        group_size: config.group_size.try_into().expect("GDN group size must fit i32"),
        bits: config.bits.try_into().expect("GDN bits must fit i32"),
        input_dtype,
        output_dtype,
        affine_dtype,
    }
}

fn qmv_batch_limit(input_dim: usize, output_dim: usize) -> u32 {
    if input_dim <= 2048 && output_dim <= 2048 {
        18
    } else if input_dim <= 4096 && output_dim <= 4096 {
        12
    } else {
        10
    }
}
