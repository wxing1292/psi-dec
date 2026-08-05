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
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::batch_metadata::GDNReplayBucketPolicy;
use crate::attn::gdn::scratch::GDNScratchBindings;
use crate::attn::gdn::state_table::GDNPreparedRequestState;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

pub const GDN_NUM_ACTIVE_REQUESTS: ReplayParameterKey = ReplayParameterKey::new("gdn.num_active_requests");
pub const GDN_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("gdn.num_active_tokens");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GDNReplayMode {
    Exact,
    Bucketed,
    BucketedWithTokenKey(ReplayParameterKey),
}

impl GDNReplayMode {
    fn active_tokens_key(self) -> Option<ReplayParameterKey> {
        match self {
            Self::Exact => None,
            Self::Bucketed => Some(GDN_NUM_ACTIVE_TOKENS),
            Self::BucketedWithTokenKey(key) => {
                assert_ne!(
                    key, GDN_NUM_ACTIVE_REQUESTS,
                    "GDN active-token key must differ from the private active-request key"
                );
                Some(key)
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GDNReplayTopology {
    pub materialize_candidate_states: bool,
    pub qkvabz_affine: AffineQuantizedMatmulKernelKind,
    pub output_affine: AffineQuantizedMatmulKernelKind,
}

pub fn add_gdn_replay_arguments(shape: GDNReplayShape, arguments: &mut ReplayArguments) {
    add_gdn_private_replay_arguments(shape, arguments);
    arguments.set_u32(GDN_NUM_ACTIVE_TOKENS, shape.num_tokens);
}

pub fn add_gdn_private_replay_arguments(shape: GDNReplayShape, arguments: &mut ReplayArguments) {
    shape.validate();
    arguments.set_u32(GDN_NUM_ACTIVE_REQUESTS, shape.num_reqs);
}

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
    pub replay_mode: GDNReplayMode,
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

    pub fn prepare_bucketed(
        &self,
        metadata: &GDNMetadataBuffers,
        cu_tokens: &[u32],
        state: &GDNPreparedRequestState,
        policy: &GDNReplayBucketPolicy,
    ) -> GDNReplayShape {
        metadata.update_bucketed(
            cu_tokens,
            &state.src_state_slots,
            &state.dst_state_slots,
            &state.flat_candidate_state_slots,
            policy,
        )
    }

    pub fn prepare_bucketed_with_token_capacity(
        &self,
        metadata: &GDNMetadataBuffers,
        cu_tokens: &[u32],
        state: &GDNPreparedRequestState,
        policy: &GDNReplayBucketPolicy,
        total_tokens: u32,
    ) -> GDNReplayShape {
        let num_tokens = cu_tokens.last().copied().unwrap_or_default();
        assert!(
            total_tokens <= policy.max_tokens(),
            "GDN caller-owned token capacity must not exceed the metadata capacity"
        );
        self.validate_token_capacity(num_tokens, total_tokens);
        metadata.update_bucketed_with_token_capacity(
            cu_tokens,
            &state.src_state_slots,
            &state.dst_state_slots,
            &state.flat_candidate_state_slots,
            policy,
            total_tokens,
        )
    }

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.qkvabz.topology_boundaries().into_vec();
        boundaries.extend(self.output.topology_boundaries());
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
    }

    pub fn replay_bucket_policy(&self, max_requests: u32, max_tokens: u32) -> GDNReplayBucketPolicy {
        let boundaries = self.replay_token_topology_boundaries();
        GDNReplayBucketPolicy::new(max_requests, max_tokens, &boundaries)
    }

    pub fn replay_topology(
        &self,
        batch_metadata: &GDNMetadataBuffers,
        materialize_candidate_states: bool,
    ) -> GDNReplayTopology {
        let shape = batch_metadata.replay_shape();
        shape.validate();
        self.replay_topology_for_token_capacity(shape.total_tokens, materialize_candidate_states)
    }

    fn replay_topology_for_token_capacity(
        &self,
        total_tokens: u32,
        materialize_candidate_states: bool,
    ) -> GDNReplayTopology {
        assert!(total_tokens > 0, "GDN replay topology requires token capacity");
        GDNReplayTopology {
            materialize_candidate_states,
            qkvabz_affine: self.qkvabz.topology(total_tokens),
            output_affine: self.output.topology(total_tokens),
        }
    }

    fn validate_token_capacity(&self, num_tokens: u32, total_tokens: u32) {
        assert!(num_tokens > 0, "GDN replay requires active tokens");
        assert!(
            num_tokens <= total_tokens,
            "GDN caller-owned token capacity must contain all active tokens"
        );
        let active_topology = self.replay_topology_for_token_capacity(num_tokens, true);
        let selected_topology = self.replay_topology_for_token_capacity(total_tokens, true);
        assert_eq!(
            active_topology.qkvabz_affine, selected_topology.qkvabz_affine,
            "GDN caller-owned token capacity must preserve the QKVABZ affine topology"
        );
        assert_eq!(
            active_topology.output_affine, selected_topology.output_affine,
            "GDN caller-owned token capacity must preserve the output affine topology"
        );
    }
}

impl ReplayLayer for GDN {
    type Input<'a> = GDNInput<'a>;
    type Output<'a> = GDNOutput<'a>;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shape = input.batch_metadata.replay_shape();
        shape.validate();
        match input.replay_mode {
            GDNReplayMode::Exact => {
                assert_eq!(shape.num_reqs, shape.total_reqs);
                assert_eq!(shape.num_tokens, shape.total_tokens);
            },
            GDNReplayMode::Bucketed | GDNReplayMode::BucketedWithTokenKey(_) => {
                self.validate_token_capacity(shape.num_tokens, shape.total_tokens);
            },
        }
        let num_active_tokens_key = input.replay_mode.active_tokens_key();
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let scratch = input.scratch;
        let batch_metadata = input.batch_metadata;
        let state = input.state;
        let weights = input.weights;
        let bucketed = num_active_tokens_key.is_some();
        let active_reqs = ReplayU32::Parameter(GDN_NUM_ACTIVE_REQUESTS);
        let active_tokens = ReplayU32::Parameter(num_active_tokens_key.unwrap_or(GDN_NUM_ACTIVE_TOKENS));
        let qkvabz = if bucketed {
            self.qkvabz.invoke_bucketed(
                shape.total_tokens,
                num_active_tokens_key.expect("bucketed GDN replay must have an active-token parameter"),
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
            )
        } else {
            self.qkvabz.invoke(
                shape.total_tokens.try_into().expect("GDN token count must fit i32"),
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
            )
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(qkvabz));
        let split_shape = GDNQKVABZSplitShape {
            num_tokens: shape.total_tokens,
        };
        let split_buffers = GDNQKVABZSplitBuffers {
            qkvabz: scratch.qkvabz,
            qkv: scratch.qkv,
            a: scratch.a,
            b: scratch.b,
            z: scratch.z,
        };
        let split = if bucketed {
            self.qkvabz_to_qkv_a_b_z
                .invoke_bucketed(split_shape, split_buffers, active_tokens)
        } else {
            self.qkvabz_to_qkv_a_b_z.invoke(split_shape, split_buffers)
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(split));
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
        let compute_shape = compute_shape(shape);
        if input.materialize_candidate_states {
            let buffers = GDNComputeWithCandidateStateUpdateBuffers {
                compute: compute_buffers,
                flat_candidate_state_slots: batch_metadata.flat_candidate_state_slots(),
            };
            let compute = if bucketed {
                self.compute.invoke_with_candidate_state_update_bucketed(
                    compute_shape,
                    buffers,
                    active_reqs,
                    active_tokens,
                )
            } else {
                self.compute.invoke_with_candidate_state_update(compute_shape, buffers)
            };
            recorder.record_with_barrier_before(ReplayOp::opaque(compute));
        } else {
            let compute = if bucketed {
                self.compute
                    .invoke_bucketed(compute_shape, compute_buffers, active_reqs, active_tokens)
            } else {
                self.compute.invoke(compute_shape, compute_buffers)
            };
            recorder.record_with_barrier_before(ReplayOp::opaque(compute));
        }
        let output = if bucketed {
            self.output.invoke_bucketed(
                shape.total_tokens,
                num_active_tokens_key.expect("bucketed GDN replay must have an active-token parameter"),
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
            )
        } else {
            self.output.invoke(
                shape.total_tokens.try_into().expect("GDN token count must fit i32"),
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
            )
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(output));
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
        num_reqs: shape.total_reqs,
        num_tokens: shape.total_tokens,
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

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;
    use inference_backend_metal::metal::ReplayArguments;
    use inference_backend_metal::metal::ReplayParameterKey;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::attn::GDNCore;

    use super::GDN;
    use super::GDN_NUM_ACTIVE_REQUESTS;
    use super::GDN_NUM_ACTIVE_TOKENS;
    use super::GDNInput;
    use super::GDNLayerStateBindings;
    use super::GDNMetalConfig;
    use super::GDNReplayMode;
    use super::GDNWeights;
    use super::add_gdn_private_replay_arguments;
    use super::add_gdn_replay_arguments;
    use super::affine_config;
    use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
    use crate::attn::gdn::scratch::GDNScratch;
    use crate::attn::gdn::state_table::GDNPreparedRequestState;
    use crate::def::layer::ReplayLayer;
    use crate::def::replay_op::MetalReplayRuntime;

    #[test]
    fn test_bucket_policy_preserves_both_affine_topologies() {
        let device = Device::system_default();
        let core = fixture_core();
        let backend = GDN::new(&device, core, fixture_metal_config());
        let metadata = GDNMetadataBuffers::new(&device, 1, 64);
        let policy = backend.replay_bucket_policy(1, 64);

        for num_tokens in 1..=64 {
            let state = GDNPreparedRequestState {
                src_state_slots: vec![0],
                dst_state_slots: vec![1],
                flat_candidate_state_slots: vec![u32::MAX; num_tokens as usize],
            };
            let shape = backend.prepare_bucketed(&metadata, &[0, num_tokens], &state, &policy);
            let topology = backend.replay_topology(&metadata, true);

            assert_eq!(topology.qkvabz_affine, backend.qkvabz.topology(num_tokens));
            assert_eq!(topology.qkvabz_affine, backend.qkvabz.topology(shape.total_tokens));
            assert_eq!(topology.output_affine, backend.output.topology(num_tokens));
            assert_eq!(topology.output_affine, backend.output.topology(shape.total_tokens));
        }
    }

    #[test]
    #[should_panic(expected = "GDN caller-owned token capacity must preserve")]
    fn test_caller_owned_token_capacity_rejects_topology_change() {
        let device = Device::system_default();
        let backend = GDN::new(&device, fixture_core(), fixture_metal_config());
        let metadata = GDNMetadataBuffers::new(&device, 1, 64);
        let policy = backend.replay_bucket_policy(1, 64);
        let topology_boundary = backend
            .replay_token_topology_boundaries()
            .iter()
            .copied()
            .find(|&boundary| boundary <= 64)
            .expect("test GDN affine topology must change within the token capacity");
        let num_tokens = topology_boundary - 1;
        let state = GDNPreparedRequestState {
            src_state_slots: vec![0],
            dst_state_slots: vec![1],
            flat_candidate_state_slots: vec![u32::MAX; num_tokens as usize],
        };

        backend.prepare_bucketed_with_token_capacity(&metadata, &[0, num_tokens], &state, &policy, topology_boundary);
    }

    #[test]
    fn test_replay_argument_helpers_separate_default_tokens_from_private_requests() {
        const STAGE_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gdn_stage.num_active_tokens");
        let shape = inference_executor_core::attn::GDNReplayShape::new(2, 4, 3, 6);

        let mut private_arguments = ReplayArguments::new();
        add_gdn_private_replay_arguments(shape, &mut private_arguments);
        assert_eq!(
            private_arguments,
            ReplayArguments::new().with_u32(GDN_NUM_ACTIVE_REQUESTS, 2)
        );

        let mut default_arguments = ReplayArguments::new();
        add_gdn_replay_arguments(shape, &mut default_arguments);
        assert_eq!(
            default_arguments,
            ReplayArguments::new()
                .with_u32(GDN_NUM_ACTIVE_REQUESTS, 2)
                .with_u32(GDN_NUM_ACTIVE_TOKENS, 3)
        );

        private_arguments.set_u32(STAGE_NUM_ACTIVE_TOKENS, shape.num_tokens);
        assert_eq!(
            private_arguments,
            ReplayArguments::new()
                .with_u32(GDN_NUM_ACTIVE_REQUESTS, 2)
                .with_u32(STAGE_NUM_ACTIVE_TOKENS, 3)
        );
    }

    #[test]
    #[should_panic(expected = "GDN active-token key must differ from the private active-request key")]
    fn test_caller_owned_token_key_rejects_private_request_key() {
        let _ = GDNReplayMode::BucketedWithTokenKey(GDN_NUM_ACTIVE_REQUESTS).active_tokens_key();
    }

    #[test]
    fn test_exact_default_and_caller_keyed_bucketed_candidate_program_parameter_counts() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let core = fixture_core();
        let metal = fixture_metal_config();
        let backend = GDN::new(&device, core.clone(), metal);
        let scratch = GDNScratch::new(&device, &core, 2);
        let metadata = GDNMetadataBuffers::new(&device, 2, 2);
        let state = GDNPreparedRequestState {
            src_state_slots: vec![0, 2],
            dst_state_slots: vec![1, 3],
            flat_candidate_state_slots: vec![u32::MAX; 2],
        };
        let hidden_state = Buffer::new_zeroed_elements(&device, 2 * core.hidden_dim, Dtype::Bfloat16);
        let next_hidden_state = Buffer::new_zeroed_elements(&device, 2 * core.hidden_dim, Dtype::Bfloat16);
        let conv_state_stride = core
            .qkv_dim()
            .checked_mul(core.conv_kernel_size - 1)
            .expect("test convolution state stride must fit usize");
        let recurrent_state_stride = core
            .num_v_heads
            .checked_mul(core.v_head_dim)
            .and_then(|value| value.checked_mul(core.qk_head_dim))
            .expect("test recurrent state stride must fit usize");
        let conv_state = Buffer::new_zeroed_elements(&device, 4 * conv_state_stride, Dtype::Float32);
        let recurrent_state = Buffer::new_zeroed_elements(&device, 4 * recurrent_state_stride, Dtype::Float32);
        let qkvabz_config = affine_config(
            core.qkvabz_dim(),
            core.hidden_dim,
            metal.input_dtype,
            Dtype::Float32,
            metal.qkvabz_scale_bias_dtype,
            metal,
        );
        let output_config = affine_config(
            core.hidden_dim,
            core.v_dim(),
            Dtype::Float32,
            metal.output_dtype,
            metal.output_scale_bias_dtype,
            metal,
        );
        let qkvabz_weight = Buffer::new_zeroed(&device, qkvabz_config.weight_bytes());
        let qkvabz_scales = Buffer::new_zeroed(&device, qkvabz_config.scale_or_bias_bytes());
        let qkvabz_biases = Buffer::new_zeroed(&device, qkvabz_config.scale_or_bias_bytes());
        let conv_weight = Buffer::new_zeroed(
            &device,
            core.qkv_dim() * core.conv_kernel_size * Dtype::Bfloat16.item_size(),
        );
        let norm_weight = Buffer::new_zeroed(&device, core.v_head_dim * Dtype::Bfloat16.item_size());
        let a_log = Buffer::new_zeroed(&device, core.num_v_heads * Dtype::Bfloat16.item_size());
        let dt_bias = Buffer::new_zeroed(&device, core.num_v_heads * Dtype::Bfloat16.item_size());
        let output_weight = Buffer::new_zeroed(&device, output_config.weight_bytes());
        let output_scales = Buffer::new_zeroed(&device, output_config.scale_or_bias_bytes());
        let output_biases = Buffer::new_zeroed(&device, output_config.scale_or_bias_bytes());
        let weights = GDNWeights {
            qkvabz_weight: &qkvabz_weight,
            qkvabz_scales: &qkvabz_scales,
            qkvabz_biases: &qkvabz_biases,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            output_weight: &output_weight,
            output_scales: &output_scales,
            output_biases: &output_biases,
        };
        let layer_state = GDNLayerStateBindings {
            conv_state: &conv_state,
            conv_state_offset_bytes: 0,
            next_conv_state: &conv_state,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &recurrent_state,
            recurrent_state_arena_offset_bytes: 0,
        };

        backend.prepare(&metadata, &[0, 1, 2], &state);
        let mut exact = runtime.create_recorder();
        let _ = <GDN as ReplayLayer>::record(
            &backend,
            &mut exact,
            GDNInput {
                hidden_state: &hidden_state,
                next_hidden_state: &next_hidden_state,
                scratch: scratch.bindings(),
                batch_metadata: &metadata,
                state: layer_state,
                materialize_candidate_states: true,
                weights,
                replay_mode: GDNReplayMode::Exact,
            },
        );
        assert_eq!(exact.build().stats().parameter_count, 0);

        let policy = backend.replay_bucket_policy(2, 2);
        backend.prepare_bucketed(&metadata, &[0, 1, 2], &state, &policy);
        let mut bucketed = runtime.create_recorder();
        let _ = <GDN as ReplayLayer>::record(
            &backend,
            &mut bucketed,
            GDNInput {
                hidden_state: &hidden_state,
                next_hidden_state: &next_hidden_state,
                scratch: scratch.bindings(),
                batch_metadata: &metadata,
                state: layer_state,
                materialize_candidate_states: true,
                weights,
                replay_mode: GDNReplayMode::Bucketed,
            },
        );
        assert_eq!(bucketed.build().stats().parameter_count, 2);

        const STAGE_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gdn_stage.num_active_tokens");
        backend.prepare_bucketed_with_token_capacity(&metadata, &[0, 1, 2], &state, &policy, 2);
        let mut caller_keyed = runtime.create_recorder();
        let _ = <GDN as ReplayLayer>::record(
            &backend,
            &mut caller_keyed,
            GDNInput {
                hidden_state: &hidden_state,
                next_hidden_state: &next_hidden_state,
                scratch: scratch.bindings(),
                batch_metadata: &metadata,
                state: layer_state,
                materialize_candidate_states: true,
                weights,
                replay_mode: GDNReplayMode::BucketedWithTokenKey(STAGE_NUM_ACTIVE_TOKENS),
            },
        );
        let caller_keyed = caller_keyed.build();
        assert_eq!(caller_keyed.stats().parameter_count, 2);
        let mut arguments = ReplayArguments::new();
        add_gdn_private_replay_arguments(metadata.replay_shape(), &mut arguments);
        arguments.set_u32(STAGE_NUM_ACTIVE_TOKENS, metadata.replay_shape().num_tokens);
        runtime.submit_replay_with_arguments(&caller_keyed, &arguments).wait();
    }

    fn fixture_core() -> GDNCore {
        GDNCore {
            model_layer_index: 0,
            hidden_dim: 32,
            num_qk_heads: 1,
            qk_head_dim: 32,
            num_v_heads: 1,
            v_head_dim: 32,
            conv_kernel_size: 3,
            q_scale: 1.0,
        }
    }

    fn fixture_metal_config() -> GDNMetalConfig {
        GDNMetalConfig {
            group_size: 32,
            bits: 4,
            norm_eps: 1.0e-6,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            qkvabz_scale_bias_dtype: Dtype::Bfloat16,
            output_scale_bias_dtype: Dtype::Bfloat16,
        }
    }
}
