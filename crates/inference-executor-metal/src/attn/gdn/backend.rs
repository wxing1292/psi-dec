use inference_backend_metal::components::GDNCompute;
use inference_backend_metal::components::GDNComputeBuffers;
use inference_backend_metal::components::GDNComputeConfig;
use inference_backend_metal::components::GDNComputeShape;
use inference_backend_metal::components::GDNComputeWithCandidateStateUpdateBuffers;
use inference_backend_metal::components::GDNQKVABZSplitBuffers;
use inference_backend_metal::components::GDNQKVABZSplitConfig;
use inference_backend_metal::components::GDNQKVABZSplitKernel;
use inference_backend_metal::components::GDNQKVABZSplitShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::scratch::GDNScratchBindings;
use crate::attn::gdn::state_table::GDNPreparedRequestState;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GDNMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub norm_eps: f32,
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
    pub qkvabz_scale_bias_dtype: Dtype,
    pub output_scale_bias_dtype: Dtype,
}

impl GDNMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(self.norm_eps > 0.0);
        validate_boundary_dtype("input", self.input_dtype);
        validate_boundary_dtype("output", self.output_dtype);
        assert!(matches!(self.qkvabz_scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(matches!(self.output_scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
    }
}

fn validate_boundary_dtype(name: &str, dtype: Dtype) {
    match dtype {
        Dtype::Bfloat16 => {},
        Dtype::Float32 => todo!("F32 GDN {name} boundary is not supported"),
        dtype => panic!("unsupported GDN {name} boundary dtype {dtype:?}"),
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
    pub a_log: &'a Buffer,
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

/// GDN data flow:
///
/// ```text
/// hidden_state (BF16)
///   -> qkvabz
///   -> scratch.qkvabz (F32)
///   -> qkvabz_to_qkv_a_b_z
///      |- scratch.qkv (F32)
///      |- scratch.a (F32)
///      |- scratch.b (F32)
///      `- scratch.z (F32)
///             |
///             v
///          compute (F32)
///      short_conv -> ragged_recurrent -> output_norm_gate
///             |
///             v
///      scratch.norm_gated_output (F32)
///             |
///             v
///          output
///             |
///             v
///      next_hidden_state (BF16)
/// ```
pub struct GDN {
    qkvabz: AffineQuantizedMatmul,
    qkvabz_to_qkv_a_b_z: GDNQKVABZSplitKernel,
    compute: GDNCompute,
    output: AffineQuantizedMatmul,
}

impl GDN {
    pub fn new(device: &Device, core: GDNCore, config: GDNMetalConfig) -> Self {
        core.validate();
        config.validate();
        let qkvabz_dim = core.qkvabz_dim();
        Self {
            qkvabz: AffineQuantizedMatmul::new(
                device,
                affine_config(
                    qkvabz_dim,
                    core.hidden_dim,
                    config.input_dtype,
                    Dtype::Float32,
                    config.qkvabz_scale_bias_dtype,
                    config,
                ),
            ),
            qkvabz_to_qkv_a_b_z: GDNQKVABZSplitKernel::new(device, qkvabz_split_config(&core)),
            compute: GDNCompute::new(device, compute_config(&core, config)),
            output: AffineQuantizedMatmul::new(
                device,
                affine_config(
                    core.hidden_dim,
                    core.v_dim(),
                    Dtype::Float32,
                    config.output_dtype,
                    config.output_scale_bias_dtype,
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

impl ReplayLayer for GDN {
    type Input<'a> = GDNInput<'a>;
    type Output<'a> = GDNOutput<'a>;

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
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qkvabz.invoke(
            shape.num_tokens.try_into().expect("GDN token count must fit i32"),
            scratch.qkvabz,
            0,
            hidden_state,
            0,
            weights.qkvabz_weight,
            0,
            weights.qkvabz_scales,
            0,
            weights.qkvabz_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qkvabz_to_qkv_a_b_z.invoke(
            GDNQKVABZSplitShape {
                num_tokens: shape.num_tokens,
            },
            GDNQKVABZSplitBuffers {
                qkvabz: scratch.qkvabz,
                qkv: scratch.qkv,
                a: scratch.a,
                b: scratch.b,
                z: scratch.z,
            },
        )));
        let compute_buffers = GDNComputeBuffers {
            qkv: scratch.qkv,
            a: scratch.a,
            b: scratch.b,
            z: scratch.z,
            conv_weight: weights.conv_weight,
            norm_weight: weights.norm_weight,
            a_log: weights.a_log,
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
            norm_gated_output: scratch.norm_gated_output,
        };
        if input.materialize_candidate_states {
            recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.invoke_with_candidate_state_update(
                compute_shape(shape),
                GDNComputeWithCandidateStateUpdateBuffers {
                    compute: compute_buffers,
                    flat_candidate_state_slots: batch_metadata.flat_candidate_state_slots(),
                },
            )));
        } else {
            recorder.record_with_barrier_before(ReplayOp::opaque(
                self.compute.invoke(compute_shape(shape), compute_buffers),
            ));
        }
        recorder.record_with_barrier_before(ReplayOp::opaque(self.output.invoke(
            shape.num_tokens.try_into().expect("GDN token count must fit i32"),
            next_hidden_state,
            0,
            scratch.norm_gated_output,
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

fn compute_config(core: &GDNCore, config: GDNMetalConfig) -> GDNComputeConfig {
    GDNComputeConfig {
        num_qk_heads: core.num_qk_heads.try_into().expect("GDN query/key heads must fit u32"),
        qk_head_dim: core.qk_head_dim.try_into().expect("GDN qk_head_dim must fit u32"),
        num_v_heads: core.num_v_heads.try_into().expect("GDN num_v_heads must fit u32"),
        v_head_dim: core.v_head_dim.try_into().expect("GDN v_head_dim must fit u32"),
        conv_kernel_size: core
            .conv_kernel_size
            .try_into()
            .expect("GDN conv_kernel_size must fit u32"),
        q_scale: core.q_scale,
        norm_eps: config.norm_eps,
    }
}

fn compute_shape(shape: GDNReplayShape) -> GDNComputeShape {
    GDNComputeShape {
        num_reqs: shape.num_reqs,
        num_tokens: shape.num_tokens,
    }
}

fn qkvabz_split_config(core: &GDNCore) -> GDNQKVABZSplitConfig {
    let qkv_dim = core.qkv_dim().try_into().expect("GDN qkv_dim must fit u32");
    let num_v_heads = core.num_v_heads.try_into().expect("GDN num_v_heads must fit u32");
    let v_dim = core.v_dim().try_into().expect("GDN v_dim must fit u32");
    GDNQKVABZSplitConfig::new(qkv_dim, num_v_heads, v_dim)
}

fn affine_config(
    n: usize,
    k: usize,
    input_dtype: Dtype,
    output_dtype: Dtype,
    scale_bias_dtype: Dtype,
    config: GDNMetalConfig,
) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig {
        n: n.try_into().expect("GDN affine n must fit i32"),
        k: k.try_into().expect("GDN affine k must fit i32"),
        group_size: config.group_size.try_into().expect("GDN group size must fit i32"),
        bits: config.bits.try_into().expect("GDN bits must fit i32"),
        input_dtype,
        output_dtype,
        scale_bias_dtype,
    }
}
