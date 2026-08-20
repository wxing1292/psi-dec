use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const GDN_COMPUTE_SOURCE: &str = include_str!("metal/gdn_compute.metal");

const SHORT_CONV_REQUIRED_THREADS: u32 = 256;
const FINAL_STATE_RECURRENT_NUM_QK_DIM_THREADS: u32 = 32;
const CANDIDATE_STATE_RECURRENT_NUM_QK_DIM_THREADS: u32 = 32;
const CANDIDATE_STATE_RECURRENT_NUM_V_ROWS_PER_SIMDGROUP: u32 = 2;
const CANDIDATE_STATE_RECURRENT_NUM_SIMDGROUPS: u32 = 2;
const OUTPUT_NORM_GATE_REQUIRED_THREADS: u32 = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockSpecialization {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelSpecialization<T> {
    thread_block: T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNFinalStateRecurrentThreadBlockSpecialization {
    num_qk_dim_threads: u32,
    num_v_rows: u32,
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNCandidateStateRecurrentSimdgroupSpecialization {
    num_v_rows: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNCandidateStateRecurrentThreadBlockSpecialization {
    num_qk_dim_threads: u32,
    num_simdgroups: u32,
    simdgroup: GDNCandidateStateRecurrentSimdgroupSpecialization,
    required_threads: u32,
}

impl GDNCandidateStateRecurrentThreadBlockSpecialization {
    fn num_v_rows(self) -> u32 {
        self.num_simdgroups * self.simdgroup.num_v_rows
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNModelGeometry {
    num_qk_heads: u32,
    qk_head_dim: u32,
    num_v_heads: u32,
    v_head_dim: u32,
    conv_kernel_size: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNKernelSpecializations {
    short_conv: KernelSpecialization<ThreadBlockSpecialization>,
    candidate_conv_state: KernelSpecialization<ThreadBlockSpecialization>,
    final_state_recurrent: KernelSpecialization<GDNFinalStateRecurrentThreadBlockSpecialization>,
    candidate_state_recurrent: KernelSpecialization<GDNCandidateStateRecurrentThreadBlockSpecialization>,
    output_norm_gate: KernelSpecialization<ThreadBlockSpecialization>,
}

/// Compile-time model geometry and thread-block geometry for the GDN compute
/// kernel set.
///
/// GDN has one recurrent algorithm. The final-state and candidate-state
/// kernels implement different state-materialization contracts. They are not
/// runtime cost candidates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNComputeSpecialization {
    model: GDNModelGeometry,
    kernels: GDNKernelSpecializations,
}

/// Construction input for the generic GDN compute graph.
///
/// The core consumes projected `qkv`, `a`, `b`, and `z` tensors over flat token
/// axis `T`. Q/K use `[T, Hqk, Dqk]`; V and recurrent output use
/// `[T, Hv, Dv]`; recurrent state uses `[S, Hv, Dv, Dqk]`. `Cqkv` is the
/// concatenated channel width `2 * Hqk * Dqk + Hv * Dv` at projection and
/// short-convolution boundaries; it is unrelated to convolution-kernel width.
///
/// ```text
/// qkv + conv_state + conv_weight
///   -> causal depthwise short_conv -> SiLU -> conv_qkv
/// old conv_state + qkv
///   -> take final Ks inputs -> next_conv_state
/// ```
///
/// `conv_qkv` is the post-SiLU recurrent-core input, not the raw convolution
/// accumulation. `next_conv_state` contains the final `Ks = Kc - 1` inputs for
/// the next invocation. Model geometry selects `GDNComputeSpecialization` at
/// initialization. `q_scale` and `norm_eps` remain kernel arguments.
#[derive(Clone, Copy, Debug)]
pub struct GDNComputeConfig {
    pub num_qk_heads: u32,
    pub qk_head_dim: u32,
    pub num_v_heads: u32,
    pub v_head_dim: u32,
    pub conv_kernel_size: u32,
    pub q_scale: f32,
    pub norm_eps: f32,
}

impl GDNComputeConfig {
    fn model_geometry(self) -> GDNModelGeometry {
        GDNModelGeometry {
            num_qk_heads: self.num_qk_heads,
            qk_head_dim: self.qk_head_dim,
            num_v_heads: self.num_v_heads,
            v_head_dim: self.v_head_dim,
            conv_kernel_size: self.conv_kernel_size,
        }
    }

    pub fn qkv_dim(self) -> u32 {
        self.model_geometry().qkv_dim()
    }

    pub fn conv_state_len(self) -> u32 {
        self.model_geometry().conv_state_len()
    }

    pub fn recurrent_state_stride(self) -> usize {
        self.model_geometry().recurrent_state_stride()
    }

    pub fn num_recurrent_output_values(self, shape: GDNComputeShape) -> usize {
        self.model_geometry().num_recurrent_output_values(shape)
    }

    pub fn num_qkv_values(self, shape: GDNComputeShape) -> usize {
        self.model_geometry().num_qkv_values(shape)
    }

    pub fn num_conv_state_values(self, shape: GDNComputeShape) -> usize {
        self.model_geometry().num_conv_state_values(shape)
    }

    fn num_conv_weight_values(self) -> usize {
        self.model_geometry().num_conv_weight_values()
    }

    fn validate(self) {
        assert!(self.num_qk_heads > 0);
        assert!(self.qk_head_dim > 0);
        assert!(self.num_v_heads > 0);
        assert!(self.v_head_dim > 0);
        assert_eq!(self.num_v_heads % self.num_qk_heads, 0);
        assert!(self.conv_kernel_size > 1);
        assert!(self.q_scale.is_finite() && self.q_scale > 0.0);
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        assert_eq!(
            self.qk_head_dim % CANDIDATE_STATE_RECURRENT_NUM_QK_DIM_THREADS,
            0,
            "GDN candidate register-V requires Dqk divisible by the SIMDgroup width"
        );
        assert_eq!(
            self.v_head_dim % CANDIDATE_STATE_RECURRENT_NUM_V_ROWS_PER_SIMDGROUP,
            0,
            "GDN candidate recurrent specialization requires Dv divisible by its SIMDgroup V-row count"
        );
        assert_eq!(
            (self.v_head_dim / CANDIDATE_STATE_RECURRENT_NUM_V_ROWS_PER_SIMDGROUP)
                % CANDIDATE_STATE_RECURRENT_NUM_SIMDGROUPS,
            0,
            "GDN candidate recurrent thread-block V-row count must divide Dv"
        );
        let _ = self.qkv_dim();
        let _ = self.recurrent_state_stride();
        let _ = self.num_conv_weight_values();
    }
}

impl GDNComputeSpecialization {
    fn from_config(config: GDNComputeConfig) -> Self {
        config.validate();
        let final_state_recurrent_num_v_rows = [8, 4]
            .into_iter()
            .find(|num_v_rows| {
                config.v_head_dim.is_multiple_of(*num_v_rows)
                    && *num_v_rows * FINAL_STATE_RECURRENT_NUM_QK_DIM_THREADS <= 1024
            })
            .expect("GDN recurrent V-dimension specialization requires Dv divisible by four");
        let candidate_state_recurrent_thread_block = GDNCandidateStateRecurrentThreadBlockSpecialization {
            num_qk_dim_threads: CANDIDATE_STATE_RECURRENT_NUM_QK_DIM_THREADS,
            num_simdgroups: CANDIDATE_STATE_RECURRENT_NUM_SIMDGROUPS,
            simdgroup: GDNCandidateStateRecurrentSimdgroupSpecialization {
                num_v_rows: CANDIDATE_STATE_RECURRENT_NUM_V_ROWS_PER_SIMDGROUP,
            },
            required_threads: CANDIDATE_STATE_RECURRENT_NUM_QK_DIM_THREADS * CANDIDATE_STATE_RECURRENT_NUM_SIMDGROUPS,
        };
        let specialization = Self {
            model: config.model_geometry(),
            kernels: GDNKernelSpecializations {
                short_conv: KernelSpecialization {
                    thread_block: ThreadBlockSpecialization {
                        required_threads: SHORT_CONV_REQUIRED_THREADS,
                    },
                },
                candidate_conv_state: KernelSpecialization {
                    thread_block: ThreadBlockSpecialization {
                        required_threads: SHORT_CONV_REQUIRED_THREADS,
                    },
                },
                final_state_recurrent: KernelSpecialization {
                    thread_block: GDNFinalStateRecurrentThreadBlockSpecialization {
                        num_qk_dim_threads: FINAL_STATE_RECURRENT_NUM_QK_DIM_THREADS,
                        num_v_rows: final_state_recurrent_num_v_rows,
                        required_threads: FINAL_STATE_RECURRENT_NUM_QK_DIM_THREADS * final_state_recurrent_num_v_rows,
                    },
                },
                candidate_state_recurrent: KernelSpecialization {
                    thread_block: candidate_state_recurrent_thread_block,
                },
                output_norm_gate: KernelSpecialization {
                    thread_block: ThreadBlockSpecialization {
                        required_threads: OUTPUT_NORM_GATE_REQUIRED_THREADS,
                    },
                },
            },
        };
        specialization.validate();
        specialization
    }

    fn validate(self) {
        assert!(self.kernels.short_conv.thread_block.required_threads > 0);
        assert!(self.kernels.candidate_conv_state.thread_block.required_threads > 0);
        let final_state_recurrent = self.kernels.final_state_recurrent.thread_block;
        assert_eq!(
            final_state_recurrent.required_threads,
            final_state_recurrent.num_qk_dim_threads * final_state_recurrent.num_v_rows
        );
        assert_eq!(self.model.v_head_dim % final_state_recurrent.num_v_rows, 0);
        let candidate_state_recurrent = self.kernels.candidate_state_recurrent.thread_block;
        assert_eq!(
            candidate_state_recurrent.required_threads,
            candidate_state_recurrent.num_qk_dim_threads * candidate_state_recurrent.num_simdgroups
        );
        assert_eq!(
            candidate_state_recurrent.num_qk_dim_threads, final_state_recurrent.num_qk_dim_threads,
            "GDN recurrent kernels share one generated Q/K-dimension thread count"
        );
        assert_eq!(self.model.v_head_dim % candidate_state_recurrent.num_v_rows(), 0);
        assert!(self.kernels.output_norm_gate.thread_block.required_threads > 0);
    }
}

impl GDNModelGeometry {
    fn qkv_dim(self) -> u32 {
        self.num_qk_heads
            .checked_mul(self.qk_head_dim)
            .and_then(|dim| dim.checked_mul(2))
            .and_then(|dim| {
                self.num_v_heads
                    .checked_mul(self.v_head_dim)
                    .and_then(|v_dim| dim.checked_add(v_dim))
            })
            .expect("GDN concatenated Q/K/V dimension must fit u32")
    }

    fn conv_state_len(self) -> u32 {
        self.conv_kernel_size - 1
    }

    fn recurrent_state_stride(self) -> usize {
        checked_product(
            "GDN recurrent state stride",
            &[
                self.num_v_heads as usize,
                self.v_head_dim as usize,
                self.qk_head_dim as usize,
            ],
        )
    }

    fn num_recurrent_output_values(self, shape: GDNComputeShape) -> usize {
        checked_product(
            "GDN output element count",
            &[
                shape.num_total_tokens as usize,
                self.num_v_heads as usize,
                self.v_head_dim as usize,
            ],
        )
    }

    fn num_qkv_values(self, shape: GDNComputeShape) -> usize {
        checked_product(
            "GDN convolution element count",
            &[shape.num_total_tokens as usize, self.qkv_dim() as usize],
        )
    }

    fn num_conv_state_values(self, shape: GDNComputeShape) -> usize {
        checked_product(
            "GDN convolution state element count",
            &[
                shape.num_total_reqs as usize,
                self.qkv_dim() as usize,
                self.conv_state_len() as usize,
            ],
        )
    }

    fn num_candidate_conv_state_values(self, shape: GDNComputeShape) -> usize {
        checked_product(
            "GDN candidate convolution state element count",
            &[
                shape.num_total_tokens as usize,
                self.qkv_dim() as usize,
                self.conv_state_len() as usize,
            ],
        )
    }

    fn num_conv_weight_values(self) -> usize {
        checked_product(
            "GDN convolution weight element count",
            &[self.qkv_dim() as usize, self.conv_kernel_size as usize],
        )
    }
}

impl GDNComputeSpecialization {
    fn total_output_norm_gate_threads(self, shape: GDNComputeShape) -> usize {
        checked_product(
            "GDN output norm + gate thread count",
            &[
                shape.num_total_tokens as usize,
                self.model.num_v_heads as usize,
                self.kernels.output_norm_gate.thread_block.required_threads as usize,
            ],
        )
    }

    fn validate_shape(self, shape: GDNComputeShape) {
        shape.validate();
        for (name, num_elements) in [
            ("GDN convolution", self.model.num_qkv_values(shape)),
            ("GDN output", self.model.num_recurrent_output_values(shape)),
            ("GDN convolution state", self.model.num_conv_state_values(shape)),
            ("GDN convolution weights", self.model.num_conv_weight_values()),
            (
                "GDN output norm + gate threads",
                self.total_output_norm_gate_threads(shape),
            ),
            ("GDN recurrent state stride", self.model.recurrent_state_stride()),
        ] {
            assert_u32_count_domain(num_elements, name);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GDNComputeShape {
    pub num_total_reqs: u32,
    pub num_total_tokens: u32,
}

impl GDNComputeShape {
    fn validate(self) {
        assert!(self.num_total_reqs > 0);
        assert!(self.num_total_tokens > 0);
    }
}

fn gdn_compute_source(specialization: GDNComputeSpecialization) -> String {
    let model = specialization.model;
    let final_state_recurrent = specialization.kernels.final_state_recurrent.thread_block;
    let candidate_state_recurrent = specialization.kernels.candidate_state_recurrent.thread_block;
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_qk_heads = {num_qk_heads}u;\nconstant uint qk_head_dim = \
         {qk_head_dim}u;\nconstant uint num_v_heads = {num_v_heads}u;\nconstant uint v_head_dim = \
         {v_head_dim}u;\nconstant uint conv_kernel_size = {conv_kernel_size}u;\nconstant uint qkv_dim = \
         {qkv_dim}u;\nconstant uint conv_state_len = {conv_state_len}u;\nconstant uint \
         final_state_recurrent_num_v_rows = {final_state_recurrent_num_v_rows}u;\nconstant uint \
         final_state_recurrent_num_qk_dim_threads = {final_state_recurrent_num_qk_dim_threads}u;\nconstant uint \
         candidate_state_recurrent_num_qk_dim_threads = {candidate_state_recurrent_num_qk_dim_threads}u;\nconstant \
         uint candidate_state_recurrent_num_v_rows_per_simdgroup = \
         {candidate_state_recurrent_num_v_rows_per_simdgroup}u;\nconstant uint \
         candidate_state_recurrent_num_simdgroups = {candidate_state_recurrent_num_simdgroups}u;\nconstant uint \
         output_norm_gate_required_threads = {output_norm_gate_required_threads}u;",
        num_qk_heads = model.num_qk_heads,
        qk_head_dim = model.qk_head_dim,
        num_v_heads = model.num_v_heads,
        v_head_dim = model.v_head_dim,
        conv_kernel_size = model.conv_kernel_size,
        qkv_dim = model.qkv_dim(),
        conv_state_len = model.conv_state_len(),
        final_state_recurrent_num_v_rows = final_state_recurrent.num_v_rows,
        final_state_recurrent_num_qk_dim_threads = final_state_recurrent.num_qk_dim_threads,
        candidate_state_recurrent_num_qk_dim_threads = candidate_state_recurrent.num_qk_dim_threads,
        candidate_state_recurrent_num_v_rows_per_simdgroup = candidate_state_recurrent.simdgroup.num_v_rows,
        candidate_state_recurrent_num_simdgroups = candidate_state_recurrent.num_simdgroups,
        output_norm_gate_required_threads = specialization.kernels.output_norm_gate.thread_block.required_threads,
    );
    GDN_COMPUTE_SOURCE.replacen("using namespace metal;", &constants, 1)
}

#[derive(Clone, Copy)]
pub struct GDNComputeBuffers<'a> {
    pub qkv: &'a Buffer,
    pub a: &'a Buffer,
    pub b: &'a Buffer,
    pub z: &'a Buffer,
    pub conv_weight: &'a Buffer,
    pub norm_weight: &'a Buffer,
    pub a_log: &'a Buffer,
    pub dt_bias: &'a Buffer,
    pub cu_tokens: &'a Buffer,
    pub src_recurrent_state_slots: &'a Buffer,
    pub src_conv_state_slots: &'a Buffer,
    /// Persistent recurrent state slot for each forward row.
    ///
    /// `u32::MAX` keeps the row output but discards the row's recurrent state.
    pub flat_materialized_recurrent_state_slots: &'a Buffer,
    /// Persistent convolution state slot for each forward row.
    ///
    /// `u32::MAX` keeps the row output but discards the row's convolution state.
    pub flat_materialized_conv_state_slots: &'a Buffer,
    pub conv_state: &'a Buffer,
    pub conv_state_offset_bytes: u64,
    pub next_conv_state: &'a Buffer,
    pub next_conv_state_offset_bytes: u64,
    pub recurrent_state_arena: &'a Buffer,
    pub recurrent_state_arena_offset_bytes: u64,
    pub conv_qkv: &'a Buffer,
    pub recurrent_output: &'a Buffer,
    pub norm_gated_output: &'a Buffer,
}

pub struct GDNCompute {
    specialization: GDNComputeSpecialization,
    q_scale: f32,
    norm_eps: f32,
    short_conv: Kernel,
    candidate_conv_state: Kernel,
    final_state_recurrent: Kernel,
    candidate_state_recurrent: Kernel,
    output_norm_gate: Kernel,
}

impl GDNCompute {
    pub fn new(device: &Device, config: GDNComputeConfig) -> Self {
        let specialization = GDNComputeSpecialization::from_config(config);
        let source = gdn_compute_source(specialization);
        Self {
            specialization,
            q_scale: config.q_scale,
            norm_eps: config.norm_eps,
            short_conv: Kernel::new(device, &source, "gdn_compute_short_conv_f32"),
            candidate_conv_state: Kernel::new(device, &source, "gdn_compute_candidate_conv_state_f32"),
            final_state_recurrent: Kernel::new(device, &source, "gdn_compute_final_state_recurrent_f32"),
            candidate_state_recurrent: Kernel::new(device, &source, "gdn_compute_candidate_state_recurrent_f32"),
            output_norm_gate: Kernel::new(device, &source, "gdn_compute_output_norm_gate_f32"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: GDNComputeShape, buffers: GDNComputeBuffers<'a>) -> GDNComputeInvocation<'a> {
        assert!(
            shape.num_total_tokens >= shape.num_total_reqs,
            "GDN ragged recurrent requires at least one token per request"
        );
        GDNComputeInvocation {
            compute: self,
            shape,
            buffers,
            num_active_reqs: ReplayU32::Fixed(shape.num_total_reqs),
            num_active_tokens: ReplayU32::Fixed(shape.num_total_tokens),
        }
    }

    pub fn invoke_bucketed<'a>(
        &'a self,
        shape: GDNComputeShape,
        buffers: GDNComputeBuffers<'a>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
    ) -> GDNComputeInvocation<'a> {
        GDNComputeInvocation {
            compute: self,
            shape,
            buffers,
            num_active_reqs,
            num_active_tokens,
        }
    }

    pub fn invoke_with_candidate_state_update<'a>(
        &'a self,
        shape: GDNComputeShape,
        buffers: GDNComputeBuffers<'a>,
    ) -> GDNComputeWithCandidateStateUpdateInvocation<'a> {
        assert!(
            shape.num_total_tokens >= shape.num_total_reqs,
            "GDN ragged recurrent requires at least one token per request"
        );
        GDNComputeWithCandidateStateUpdateInvocation {
            compute: self,
            shape,
            buffers,
            num_active_reqs: ReplayU32::Fixed(shape.num_total_reqs),
            num_active_tokens: ReplayU32::Fixed(shape.num_total_tokens),
        }
    }

    pub fn invoke_with_candidate_state_update_bucketed<'a>(
        &'a self,
        shape: GDNComputeShape,
        buffers: GDNComputeBuffers<'a>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
    ) -> GDNComputeWithCandidateStateUpdateInvocation<'a> {
        GDNComputeWithCandidateStateUpdateInvocation {
            compute: self,
            shape,
            buffers,
            num_active_reqs,
            num_active_tokens,
        }
    }

    fn record_short_conv(
        &self,
        recorder: &CommandRecorder,
        shape: GDNComputeShape,
        buffers: &GDNComputeBuffers<'_>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
        write_final_state: bool,
    ) {
        recorder.set_kernel(&self.short_conv);
        recorder.set_buffer_write(0, buffers.conv_qkv, 0);
        recorder.set_buffer_write(1, buffers.next_conv_state, 0);
        recorder.set_buffer_read(2, buffers.qkv, 0);
        recorder.set_buffer_read(3, buffers.conv_state, 0);
        recorder.set_buffer_read(4, buffers.conv_weight, 0);
        recorder.set_buffer_read(5, buffers.src_conv_state_slots, 0);
        recorder.set_buffer_read(6, buffers.flat_materialized_conv_state_slots, 0);
        recorder.set_buffer_read(7, buffers.cu_tokens, 0);
        set_batch_args(recorder, shape, 8, num_active_reqs, num_active_tokens);
        recorder.set_u64(10, buffers.conv_state_offset_bytes);
        recorder.set_u64(11, buffers.next_conv_state_offset_bytes);
        recorder.set_u32(12, u32::from(write_final_state));
        let total_short_conv_threads = self
            .specialization
            .model
            .num_qkv_values(shape)
            .max(self.specialization.model.num_conv_state_values(shape));
        recorder.dispatch_1d(
            total_short_conv_threads,
            self.specialization.kernels.short_conv.thread_block.required_threads as usize,
        );
    }

    fn record_candidate_conv_state(
        &self,
        recorder: &CommandRecorder,
        shape: GDNComputeShape,
        buffers: &GDNComputeBuffers<'_>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
    ) {
        recorder.set_kernel(&self.candidate_conv_state);
        recorder.set_barrier_before();
        recorder.set_buffer_write(0, buffers.next_conv_state, 0);
        recorder.set_buffer_read(1, buffers.qkv, 0);
        recorder.set_buffer_read(2, buffers.conv_state, 0);
        recorder.set_buffer_read(3, buffers.src_conv_state_slots, 0);
        recorder.set_buffer_read(4, buffers.flat_materialized_conv_state_slots, 0);
        recorder.set_buffer_read(5, buffers.cu_tokens, 0);
        set_batch_args(recorder, shape, 6, num_active_reqs, num_active_tokens);
        recorder.set_u64(8, buffers.conv_state_offset_bytes);
        recorder.set_u64(9, buffers.next_conv_state_offset_bytes);
        recorder.dispatch_1d(
            self.specialization.model.num_candidate_conv_state_values(shape),
            self.specialization
                .kernels
                .candidate_conv_state
                .thread_block
                .required_threads as usize,
        );
    }

    /// Current final-state recurrent execution (`R = num_reqs`):
    ///
    /// ```text
    /// recurrent_state: [S, Hv, Dv, Dqk]  (Dqk contiguous)
    /// grid:             (Dv / num_v_rows, R * Hv, 1)
    /// threadblock:      (num_qk_dim_threads, num_v_rows, 1)
    /// GDNFinalStateRecurrentThreadBlockTask / threadblock
    ///   -> owns recurrent_state[slot, v_head_index, v_dim_indices, 0..Dqk]
    ///   -> advances it over flat_token_indices in order
    /// task from grid: request_index, v_head_index, v_dim_indices
    /// task from metadata: flat_token_indices
    /// parallel: requests, V heads, V-row ranges, Dqk lanes
    /// ordered:  tokens within one request
    /// produces: recurrent_output; updates: destination recurrent_state slice
    /// ```
    ///
    /// The kernel derives the task from its arguments, thread-block index, and
    /// specialization. It does not require a materialized task buffer.
    fn record_final_state_recurrent(
        &self,
        recorder: &CommandRecorder,
        shape: GDNComputeShape,
        buffers: &GDNComputeBuffers<'_>,
        num_active_reqs: ReplayU32,
    ) {
        recorder.set_kernel(&self.final_state_recurrent);
        recorder.set_barrier_before();
        recorder.set_buffer_write(0, buffers.recurrent_output, 0);
        recorder.set_buffer_read_write(1, buffers.recurrent_state_arena, 0);
        recorder.set_buffer_read(2, buffers.conv_qkv, 0);
        recorder.set_buffer_read(3, buffers.a, 0);
        recorder.set_buffer_read(4, buffers.b, 0);
        recorder.set_buffer_read(5, buffers.a_log, 0);
        recorder.set_buffer_read(6, buffers.dt_bias, 0);
        recorder.set_buffer_read(7, buffers.src_recurrent_state_slots, 0);
        recorder.set_buffer_read(8, buffers.flat_materialized_recurrent_state_slots, 0);
        recorder.set_buffer_read(9, buffers.cu_tokens, 0);
        recorder.set_f32(10, self.q_scale);
        set_replay_u32(
            recorder,
            11,
            num_active_reqs,
            shape.num_total_reqs,
            "GDN active request count",
        );
        recorder.set_u64(12, buffers.recurrent_state_arena_offset_bytes);
        let thread_block = self.specialization.kernels.final_state_recurrent.thread_block;
        let num_v_row_ranges = self.specialization.model.v_head_dim / thread_block.num_v_rows;
        recorder.dispatch_threadblocks(
            (
                num_v_row_ranges as usize,
                shape.num_total_reqs as usize * self.specialization.model.num_v_heads as usize,
                1,
            ),
            (
                thread_block.num_qk_dim_threads as usize,
                thread_block.num_v_rows as usize,
                1,
            ),
        );
    }

    /// Current candidate-state recurrent execution:
    ///
    /// ```text
    /// grid:        (Dv / num_v_rows, R * Hv, 1)
    /// threadblock: (num_qk_dim_threads, num_simdgroups, 1)
    /// GDNCandidateStateRecurrentThreadBlockTask / threadblock
    ///   -> owns recurrent_state[slot, v_head_index, v_dim_indices, 0..Dqk]
    ///   -> advances flat_token_indices in order
    ///   -> can materialize the state after each token
    /// ```
    ///
    /// Each SIMDgroup owns `simdgroup.num_v_rows` rows. The full thread block
    /// owns `thread_block.num_v_rows()` rows. The kernel derives the task from
    /// its arguments, thread-block index, and specialization.
    fn record_candidate_state_recurrent(
        &self,
        recorder: &CommandRecorder,
        shape: GDNComputeShape,
        buffers: &GDNComputeBuffers<'_>,
        num_active_reqs: ReplayU32,
    ) {
        recorder.set_kernel(&self.candidate_state_recurrent);
        recorder.set_barrier_before();
        recorder.set_buffer_write(0, buffers.recurrent_output, 0);
        recorder.set_buffer_read_write(1, buffers.recurrent_state_arena, 0);
        recorder.set_buffer_read(2, buffers.conv_qkv, 0);
        recorder.set_buffer_read(3, buffers.a, 0);
        recorder.set_buffer_read(4, buffers.b, 0);
        recorder.set_buffer_read(5, buffers.a_log, 0);
        recorder.set_buffer_read(6, buffers.dt_bias, 0);
        recorder.set_buffer_read(7, buffers.src_recurrent_state_slots, 0);
        recorder.set_buffer_read(8, buffers.flat_materialized_recurrent_state_slots, 0);
        recorder.set_buffer_read(9, buffers.cu_tokens, 0);
        recorder.set_f32(10, self.q_scale);
        set_replay_u32(
            recorder,
            11,
            num_active_reqs,
            shape.num_total_reqs,
            "GDN active request count",
        );
        recorder.set_u64(12, buffers.recurrent_state_arena_offset_bytes);
        let thread_block = self.specialization.kernels.candidate_state_recurrent.thread_block;
        let num_threadblocks = self.specialization.model.v_head_dim / thread_block.num_v_rows();
        recorder.dispatch_threadblocks(
            (
                num_threadblocks as usize,
                shape.num_total_reqs as usize * self.specialization.model.num_v_heads as usize,
                1,
            ),
            (
                thread_block.num_qk_dim_threads as usize,
                thread_block.num_simdgroups as usize,
                1,
            ),
        );
    }

    /// Output norm + gate execution:
    ///
    /// ```text
    /// recurrent_output [T, Hv, Dv] -> RMS norm * SiLU(z)
    ///   -> norm_gated_output [T, Hv, Dv]
    /// GDNOutputNormGateThreadBlockTask / threadblock:
    ///   { flat_token_index, v_head_index }
    /// grid: (T * Hv, 1, 1); threadblock: (128, 1, 1)
    /// reduce: Dv; produces: one normalized/gated [Dv] vector
    /// ```
    ///
    /// Both task fields are grid-derived. The kernel does not require a
    /// materialized task buffer.
    fn record_output_norm_gate(
        &self,
        recorder: &CommandRecorder,
        shape: GDNComputeShape,
        buffers: &GDNComputeBuffers<'_>,
        num_active_tokens: ReplayU32,
    ) {
        recorder.set_kernel(&self.output_norm_gate);
        recorder.set_barrier_before();
        recorder.set_buffer_write(0, buffers.norm_gated_output, 0);
        recorder.set_buffer_read(1, buffers.recurrent_output, 0);
        recorder.set_buffer_read(2, buffers.z, 0);
        recorder.set_buffer_read(3, buffers.norm_weight, 0);
        recorder.set_f32(4, self.norm_eps);
        set_replay_u32(
            recorder,
            5,
            num_active_tokens,
            shape.num_total_tokens,
            "GDN active token count",
        );
        recorder.dispatch_1d(
            self.specialization.total_output_norm_gate_threads(shape),
            self.specialization
                .kernels
                .output_norm_gate
                .thread_block
                .required_threads as usize,
        );
    }
}

pub struct GDNComputeInvocation<'a> {
    compute: &'a GDNCompute,
    shape: GDNComputeShape,
    buffers: GDNComputeBuffers<'a>,
    num_active_reqs: ReplayU32,
    num_active_tokens: ReplayU32,
}

impl Operator for GDNComputeInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.compute.specialization.validate_shape(self.shape);
        validate_buffers(self.compute.specialization, self.shape, &self.buffers);
        self.compute.record_short_conv(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
            true,
        );
        self.compute
            .record_final_state_recurrent(recorder, self.shape, &self.buffers, self.num_active_reqs);
        self.compute
            .record_output_norm_gate(recorder, self.shape, &self.buffers, self.num_active_tokens);
    }
}

pub struct GDNComputeWithCandidateStateUpdateInvocation<'a> {
    compute: &'a GDNCompute,
    shape: GDNComputeShape,
    buffers: GDNComputeBuffers<'a>,
    num_active_reqs: ReplayU32,
    num_active_tokens: ReplayU32,
}

impl Operator for GDNComputeWithCandidateStateUpdateInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.compute.specialization.validate_shape(self.shape);
        assert_u32_count_domain(
            self.compute
                .specialization
                .model
                .num_candidate_conv_state_values(self.shape),
            "GDN candidate convolution state",
        );
        validate_buffers(self.compute.specialization, self.shape, &self.buffers);
        self.compute.record_short_conv(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
            false,
        );
        self.compute.record_candidate_conv_state(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
        );
        self.compute
            .record_candidate_state_recurrent(recorder, self.shape, &self.buffers, self.num_active_reqs);
        self.compute
            .record_output_norm_gate(recorder, self.shape, &self.buffers, self.num_active_tokens);
    }
}

fn set_batch_args(
    recorder: &CommandRecorder,
    shape: GDNComputeShape,
    start_index: usize,
    num_active_reqs: ReplayU32,
    num_active_tokens: ReplayU32,
) {
    set_replay_u32(
        recorder,
        start_index,
        num_active_reqs,
        shape.num_total_reqs,
        "GDN active request count",
    );
    set_replay_u32(
        recorder,
        start_index + 1,
        num_active_tokens,
        shape.num_total_tokens,
        "GDN active token count",
    );
}

fn set_replay_u32(recorder: &CommandRecorder<'_>, index: usize, value: ReplayU32, max_value: u32, name: &str) {
    match value {
        ReplayU32::Fixed(value) => {
            assert!(value > 0, "{name} must be positive");
            assert!(value <= max_value, "{name} exceeds recorded capacity");
            recorder.set_u32(index, value);
        },
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, max_value),
    }
}

fn validate_buffers(specialization: GDNComputeSpecialization, shape: GDNComputeShape, buffers: &GDNComputeBuffers<'_>) {
    let model = specialization.model;
    let f32_bytes = size_of::<f32>() as u64;
    for (name, offset_bytes) in [
        ("conv_state", buffers.conv_state_offset_bytes),
        ("next_conv_state", buffers.next_conv_state_offset_bytes),
        ("recurrent_state", buffers.recurrent_state_arena_offset_bytes),
    ] {
        assert_eq!(
            offset_bytes % f32_bytes,
            0,
            "GDN {name} byte offset must be f32-aligned"
        );
    }
    assert!(
        buffers.qkv.len_bytes() >= model.num_qkv_values(shape) * size_of::<f32>(),
        "GDN qkv buffer is too small"
    );
    assert!(
        buffers.a.len_bytes() >= shape.num_total_tokens as usize * model.num_v_heads as usize * size_of::<f32>(),
        "GDN a buffer is too small"
    );
    assert!(
        buffers.b.len_bytes() >= shape.num_total_tokens as usize * model.num_v_heads as usize * size_of::<f32>(),
        "GDN b buffer is too small"
    );
    assert!(
        buffers.z.len_bytes() >= model.num_recurrent_output_values(shape) * size_of::<f32>(),
        "GDN z buffer is too small"
    );
    assert_eq!(
        buffers.conv_weight.len_bytes(),
        model.num_conv_weight_values() * size_of::<u16>()
    );
    assert_eq!(
        buffers.norm_weight.len_bytes(),
        model.v_head_dim as usize * size_of::<u16>()
    );
    assert_eq!(buffers.a_log.len_bytes(), model.num_v_heads as usize * size_of::<u16>());
    assert_eq!(
        buffers.dt_bias.len_bytes(),
        model.num_v_heads as usize * size_of::<u16>()
    );
    assert!(
        buffers.cu_tokens.len_bytes_u64()
            >= (shape.num_total_reqs as u64 + 1)
                .checked_mul(size_of::<u32>() as u64)
                .expect("GDN cumulative-token byte length must fit u64"),
        "GDN cu_tokens buffer is too small"
    );
    assert!(
        buffers.src_recurrent_state_slots.len_bytes() >= shape.num_total_reqs as usize * size_of::<u32>(),
        "GDN src_recurrent_state_slots buffer is too small"
    );
    assert!(
        buffers.src_conv_state_slots.len_bytes() >= shape.num_total_reqs as usize * size_of::<u32>(),
        "GDN src_conv_state_slots buffer is too small"
    );
    assert!(
        buffers.flat_materialized_recurrent_state_slots.len_bytes()
            >= shape.num_total_tokens as usize * size_of::<u32>(),
        "GDN flat_materialized_recurrent_state_slots buffer is too small"
    );
    assert!(
        buffers.flat_materialized_conv_state_slots.len_bytes() >= shape.num_total_tokens as usize * size_of::<u32>(),
        "GDN flat_materialized_conv_state_slots buffer is too small"
    );
    let conv_state_region_bytes = (model.num_conv_state_values(shape) as u64)
        .checked_mul(f32_bytes)
        .expect("GDN convolution state region bytes must fit u64");
    let recurrent_state_region_bytes = (model.recurrent_state_stride() as u64)
        .checked_mul(f32_bytes)
        .expect("GDN recurrent state region bytes must fit u64");
    assert!(
        buffers.conv_state.len_bytes_u64()
            >= buffers
                .conv_state_offset_bytes
                .checked_add(conv_state_region_bytes)
                .expect("GDN conv_state region size overflow"),
        "GDN conv_state buffer is too small"
    );
    assert!(
        buffers.next_conv_state.len_bytes_u64()
            >= buffers
                .next_conv_state_offset_bytes
                .checked_add(conv_state_region_bytes)
                .expect("GDN next_conv_state region size overflow"),
        "GDN next_conv_state buffer is too small"
    );
    assert!(
        buffers.recurrent_state_arena.len_bytes_u64()
            >= buffers
                .recurrent_state_arena_offset_bytes
                .checked_add(recurrent_state_region_bytes)
                .expect("GDN recurrent state region size overflow")
    );
    assert!(
        buffers.conv_qkv.len_bytes() >= model.num_qkv_values(shape) * size_of::<f32>(),
        "GDN conv_qkv buffer is too small"
    );
    assert!(
        buffers.recurrent_output.len_bytes() >= model.num_recurrent_output_values(shape) * size_of::<f32>(),
        "GDN recurrent_output buffer is too small"
    );
    assert!(
        buffers.norm_gated_output.len_bytes() >= model.num_recurrent_output_values(shape) * size_of::<f32>(),
        "GDN norm_gated_output buffer is too small"
    );
}

#[cfg(test)]
#[path = "gdn_compute_test.rs"]
mod tests;
