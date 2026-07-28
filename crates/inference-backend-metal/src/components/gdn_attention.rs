use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Kernel;
use crate::metal::Operator;

const GDN_CORE_SOURCE: &str = include_str!("metal/gdn_core.metal");

const SHORT_CONV_NUM_THREADS_PER_THREADBLOCK: usize = 256;
const RAGGED_RECURRENT_NUM_QK_DIM_THREADS: usize = 32;
const OUTPUT_NORM_GATE_NUM_THREADS_PER_THREADBLOCK: usize = 128;

/// Static geometry for the generic GDN core.
///
/// The core consumes projected `qkv`, `a`, `b`, and `z` tensors over flat token
/// axis `T`. Q/K use `[T, Hqk, Dqk]`; V and recurrent output use
/// `[T, Hv, Dv]`; recurrent state uses `[S, Hv, Dv, Dqk]`. `Cqkv` is the
/// concatenated channel width `2 * Hqk * Dqk + Hv * Dv` at projection and
/// short-convolution boundaries; it is unrelated to convolution-kernel width.
///
/// ```text
/// projected_qkv + conv_state + conv_weight
///   -> causal depthwise short_conv -> SiLU -> conv_qkv
/// old conv_state + projected_qkv
///   -> take final Ks inputs -> next_conv_state
/// ```
///
/// `conv_qkv` is the post-SiLU recurrent-core input, not the raw convolution
/// accumulation. `next_conv_state` contains the final `Ks = Kc - 1` inputs for
/// the next invocation.
#[derive(Clone, Copy, Debug)]
pub struct GDNCoreConfig {
    pub num_qk_heads: u32,
    pub qk_head_dim: u32,
    pub num_v_heads: u32,
    pub v_head_dim: u32,
    pub conv_kernel_size: u32,
    pub v_dim_tile_size: u32,
}

impl GDNCoreConfig {
    pub fn qkv_dim(self) -> u32 {
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

    pub fn conv_state_len(self) -> u32 {
        self.conv_kernel_size - 1
    }

    pub fn recurrent_state_stride(self) -> usize {
        checked_product(
            "GDN recurrent state stride",
            &[
                self.num_v_heads as usize,
                self.v_head_dim as usize,
                self.qk_head_dim as usize,
            ],
        )
    }

    pub fn num_recurrent_output_values(self, shape: GDNCoreShape) -> usize {
        checked_product(
            "GDN output element count",
            &[
                shape.num_tokens as usize,
                self.num_v_heads as usize,
                self.v_head_dim as usize,
            ],
        )
    }

    pub fn num_qkv_values(self, shape: GDNCoreShape) -> usize {
        checked_product(
            "GDN convolution element count",
            &[shape.num_tokens as usize, self.qkv_dim() as usize],
        )
    }

    pub fn num_conv_state_values(self, shape: GDNCoreShape) -> usize {
        checked_product(
            "GDN convolution state element count",
            &[
                shape.num_reqs as usize,
                self.qkv_dim() as usize,
                self.conv_state_len() as usize,
            ],
        )
    }

    fn num_candidate_conv_state_values(self, shape: GDNCoreShape) -> usize {
        checked_product(
            "GDN candidate convolution state element count",
            &[
                shape.num_tokens as usize,
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

    fn total_output_norm_gate_threads(self, shape: GDNCoreShape) -> usize {
        checked_product(
            "GDN output norm + gate thread count",
            &[
                shape.num_tokens as usize,
                self.num_v_heads as usize,
                OUTPUT_NORM_GATE_NUM_THREADS_PER_THREADBLOCK,
            ],
        )
    }

    fn validate(self) {
        assert!(self.num_qk_heads > 0);
        assert!(self.qk_head_dim > 0);
        assert!(self.num_v_heads > 0);
        assert!(self.v_head_dim > 0);
        assert_eq!(self.num_v_heads % self.num_qk_heads, 0);
        assert!(self.conv_kernel_size > 1);
        assert!(self.v_dim_tile_size > 0);
        assert_eq!(self.v_head_dim % self.v_dim_tile_size, 0);
        assert!(self.v_dim_tile_size as usize * RAGGED_RECURRENT_NUM_QK_DIM_THREADS <= 1024);
        let _ = self.qkv_dim();
        let _ = self.recurrent_state_stride();
        let _ = self.num_conv_weight_values();
    }

    fn validate_shape(self, shape: GDNCoreShape) {
        self.validate();
        shape.validate();
        for (name, num_elements) in [
            ("GDN convolution", self.num_qkv_values(shape)),
            ("GDN output", self.num_recurrent_output_values(shape)),
            ("GDN convolution state", self.num_conv_state_values(shape)),
            ("GDN convolution weights", self.num_conv_weight_values()),
            (
                "GDN output norm + gate threads",
                self.total_output_norm_gate_threads(shape),
            ),
            ("GDN recurrent state stride", self.recurrent_state_stride()),
        ] {
            assert_u32_count_domain(num_elements, name);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GDNCoreShape {
    pub num_reqs: u32,
    pub num_tokens: u32,
}

impl GDNCoreShape {
    fn validate(self) {
        assert!(self.num_reqs > 0);
        assert!(self.num_tokens > 0);
    }
}

fn gdn_core_source(config: GDNCoreConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_qk_heads = {num_qk_heads}u;\nconstant uint qk_head_dim = \
         {qk_head_dim}u;\nconstant uint num_v_heads = {num_v_heads}u;\nconstant uint v_head_dim = \
         {v_head_dim}u;\nconstant uint conv_kernel_size = {conv_kernel_size}u;\nconstant uint qkv_dim = \
         {qkv_dim}u;\nconstant uint conv_state_len = {conv_state_len}u;\nconstant uint v_dim_tile_size = \
         {v_dim_tile_size}u;",
        num_qk_heads = config.num_qk_heads,
        qk_head_dim = config.qk_head_dim,
        num_v_heads = config.num_v_heads,
        v_head_dim = config.v_head_dim,
        conv_kernel_size = config.conv_kernel_size,
        qkv_dim = config.qkv_dim(),
        conv_state_len = config.conv_state_len(),
        v_dim_tile_size = config.v_dim_tile_size,
    );
    GDN_CORE_SOURCE.replacen("using namespace metal;", &constants, 1)
}

#[derive(Clone, Copy)]
pub struct GDNCoreBuffers<'a> {
    pub projected_qkv: &'a Buffer,
    pub a: &'a Buffer,
    pub b: &'a Buffer,
    pub z: &'a Buffer,
    pub conv_weight: &'a Buffer,
    pub norm_weight: &'a Buffer,
    pub a_log: &'a Buffer,
    pub dt_bias: &'a Buffer,
    pub cu_tokens: &'a Buffer,
    pub src_state_slots: &'a Buffer,
    pub dst_state_slots: &'a Buffer,
    pub conv_state: &'a Buffer,
    pub conv_state_offset_bytes: u64,
    pub next_conv_state: &'a Buffer,
    pub next_conv_state_offset_bytes: u64,
    pub recurrent_state_arena: &'a Buffer,
    pub recurrent_state_arena_offset_bytes: u64,
    pub conv_qkv: &'a Buffer,
    pub recurrent_output: &'a Buffer,
    pub pre_output_hidden_states: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct GDNCoreForwardCandidateStateUpdateBuffers<'a> {
    pub core: GDNCoreBuffers<'a>,
    pub flat_candidate_state_slots: &'a Buffer,
}

pub struct GDNCoreKernels {
    config: GDNCoreConfig,
    short_conv: Kernel,
    forward_conv_candidate_state: Kernel,
    ragged_recurrent: Kernel,
    ragged_recurrent_forward_candidate_state: Kernel,
    output_norm_gate: Kernel,
}

impl GDNCoreKernels {
    pub fn new(device: &Device, config: GDNCoreConfig) -> Self {
        config.validate();
        let source = gdn_core_source(config);
        Self {
            config,
            short_conv: Kernel::new(device, &source, "gdn_core_short_conv_f32"),
            forward_conv_candidate_state: Kernel::new(device, &source, "gdn_core_forward_conv_candidate_state_f32"),
            ragged_recurrent: Kernel::new(device, &source, "gdn_core_ragged_recurrent_f32"),
            ragged_recurrent_forward_candidate_state: Kernel::new(
                device,
                &source,
                "gdn_core_ragged_recurrent_forward_candidate_state_f32",
            ),
            output_norm_gate: Kernel::new(device, &source, "gdn_core_output_norm_gate_f32"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GDNCoreShape,
        buffers: GDNCoreBuffers<'a>,
        q_scale: f32,
        eps: f32,
    ) -> GDNCoreInvocation<'a> {
        GDNCoreInvocation {
            kernels: self,
            shape,
            buffers,
            q_scale,
            eps,
        }
    }

    pub fn invoke_forward_candidate_state_update<'a>(
        &'a self,
        shape: GDNCoreShape,
        buffers: GDNCoreForwardCandidateStateUpdateBuffers<'a>,
        q_scale: f32,
        eps: f32,
    ) -> GDNCoreForwardCandidateStateUpdateInvocation<'a> {
        GDNCoreForwardCandidateStateUpdateInvocation {
            kernels: self,
            shape,
            buffers,
            q_scale,
            eps,
        }
    }

    fn record_short_conv(&self, builder: &CommandRecorder, shape: GDNCoreShape, buffers: &GDNCoreBuffers<'_>) {
        builder.set_kernel(&self.short_conv);
        builder.set_buffer_write(0, buffers.conv_qkv, 0);
        builder.set_buffer_write(1, buffers.next_conv_state, 0);
        builder.set_buffer_read(2, buffers.projected_qkv, 0);
        builder.set_buffer_read(3, buffers.conv_state, 0);
        builder.set_buffer_read(4, buffers.conv_weight, 0);
        builder.set_buffer_read(5, buffers.src_state_slots, 0);
        builder.set_buffer_read(6, buffers.dst_state_slots, 0);
        builder.set_buffer_read(7, buffers.cu_tokens, 0);
        set_batch_args(builder, shape, 8);
        builder.set_u64(10, buffers.conv_state_offset_bytes);
        builder.set_u64(11, buffers.next_conv_state_offset_bytes);
        let total_short_conv_threads = self
            .config
            .num_qkv_values(shape)
            .max(self.config.num_conv_state_values(shape));
        builder.dispatch_1d(total_short_conv_threads, SHORT_CONV_NUM_THREADS_PER_THREADBLOCK);
    }

    fn record_forward_conv_candidate_state(
        &self,
        builder: &CommandRecorder,
        shape: GDNCoreShape,
        buffers: &GDNCoreForwardCandidateStateUpdateBuffers<'_>,
    ) {
        let core = &buffers.core;
        builder.set_kernel(&self.forward_conv_candidate_state);
        builder.set_barrier_before();
        builder.set_buffer_write(0, core.next_conv_state, 0);
        builder.set_buffer_read(1, core.projected_qkv, 0);
        builder.set_buffer_read(2, core.conv_state, 0);
        builder.set_buffer_read(3, core.src_state_slots, 0);
        builder.set_buffer_read(4, buffers.flat_candidate_state_slots, 0);
        builder.set_buffer_read(5, core.cu_tokens, 0);
        set_batch_args(builder, shape, 6);
        builder.set_u64(8, core.conv_state_offset_bytes);
        builder.set_u64(9, core.next_conv_state_offset_bytes);
        builder.dispatch_1d(
            self.config.num_candidate_conv_state_values(shape),
            SHORT_CONV_NUM_THREADS_PER_THREADBLOCK,
        );
    }

    /// Current ragged recurrent execution path (`R = num_reqs`):
    ///
    /// ```text
    /// recurrent_state: [S, Hv, Dv, Dqk]  (Dqk contiguous)
    /// grid:             (Dv / Dv_tile, R * Hv, 1)
    /// threadblock:      (32, Dv_tile, 1)
    /// GDNRaggedRecurrentTask / threadblock
    ///   -> owns one GDNRecurrentStateTile [Dv_tile, Dqk]
    ///   -> advances it over cu_tokens[req_index]..cu_tokens[req_index + 1]
    /// task from grid: req_index, v_head_index, v_dim_tile_index
    /// task from metadata: flat_token_begin, flat_token_end
    /// parallel: requests, V heads, V-dimension tiles, Dqk lanes
    /// ordered:  tokens within one request
    /// produces: recurrent_output; updates: destination recurrent_state tile
    /// ```
    ///
    /// No Task value, TaskTemplate, or ABI buffer is materialized.
    fn record_ragged_recurrent(
        &self,
        builder: &CommandRecorder,
        shape: GDNCoreShape,
        buffers: &GDNCoreBuffers<'_>,
        q_scale: f32,
    ) {
        builder.set_kernel(&self.ragged_recurrent);
        builder.set_barrier_before();
        builder.set_buffer_write(0, buffers.recurrent_output, 0);
        builder.set_buffer_read_write(1, buffers.recurrent_state_arena, 0);
        builder.set_buffer_read(2, buffers.conv_qkv, 0);
        builder.set_buffer_read(3, buffers.a, 0);
        builder.set_buffer_read(4, buffers.b, 0);
        builder.set_buffer_read(5, buffers.a_log, 0);
        builder.set_buffer_read(6, buffers.dt_bias, 0);
        builder.set_buffer_read(7, buffers.src_state_slots, 0);
        builder.set_buffer_read(8, buffers.dst_state_slots, 0);
        builder.set_buffer_read(9, buffers.cu_tokens, 0);
        builder.set_f32(10, q_scale);
        set_batch_args(builder, shape, 11);
        builder.set_u64(13, buffers.recurrent_state_arena_offset_bytes);
        let v_dim_tile_size = self.config.v_dim_tile_size as usize;
        let num_v_dim_tiles = self.config.v_head_dim as usize / v_dim_tile_size;
        builder.dispatch_threadblocks(
            (
                num_v_dim_tiles,
                shape.num_reqs as usize * self.config.num_v_heads as usize,
                1,
            ),
            (RAGGED_RECURRENT_NUM_QK_DIM_THREADS, v_dim_tile_size, 1),
        );
    }

    fn record_ragged_recurrent_forward_candidate_state(
        &self,
        builder: &CommandRecorder,
        shape: GDNCoreShape,
        buffers: &GDNCoreForwardCandidateStateUpdateBuffers<'_>,
        q_scale: f32,
    ) {
        let core = &buffers.core;
        builder.set_kernel(&self.ragged_recurrent_forward_candidate_state);
        builder.set_barrier_before();
        builder.set_buffer_write(0, core.recurrent_output, 0);
        builder.set_buffer_read_write(1, core.recurrent_state_arena, 0);
        builder.set_buffer_read(2, core.conv_qkv, 0);
        builder.set_buffer_read(3, core.a, 0);
        builder.set_buffer_read(4, core.b, 0);
        builder.set_buffer_read(5, core.a_log, 0);
        builder.set_buffer_read(6, core.dt_bias, 0);
        builder.set_buffer_read(7, core.src_state_slots, 0);
        builder.set_buffer_read(8, core.dst_state_slots, 0);
        builder.set_buffer_read(9, buffers.flat_candidate_state_slots, 0);
        builder.set_buffer_read(10, core.cu_tokens, 0);
        builder.set_f32(11, q_scale);
        set_batch_args(builder, shape, 12);
        builder.set_u64(14, core.recurrent_state_arena_offset_bytes);
        let v_dim_tile_size = self.config.v_dim_tile_size as usize;
        let num_v_dim_tiles = self.config.v_head_dim as usize / v_dim_tile_size;
        builder.dispatch_threadblocks(
            (
                num_v_dim_tiles,
                shape.num_reqs as usize * self.config.num_v_heads as usize,
                1,
            ),
            (RAGGED_RECURRENT_NUM_QK_DIM_THREADS, v_dim_tile_size, 1),
        );
    }

    /// Output norm + gate execution:
    ///
    /// ```text
    /// recurrent_output [T, Hv, Dv] -> RMS norm * SiLU(z)
    ///   -> pre_output_hidden_states [T, Hv, Dv]
    /// GDNOutputNormGateTask / threadblock: { flat_token_index, v_head_index }
    /// grid: (T * Hv, 1, 1); threadblock: (128, 1, 1)
    /// reduce: Dv; produces: one normalized/gated [Dv] vector
    /// ```
    ///
    /// Both Task fields are grid-derived, so no Task value, TaskTemplate, or
    /// ABI buffer is materialized.
    fn record_output_norm_gate(
        &self,
        builder: &CommandRecorder,
        shape: GDNCoreShape,
        buffers: &GDNCoreBuffers<'_>,
        eps: f32,
    ) {
        builder.set_kernel(&self.output_norm_gate);
        builder.set_barrier_before();
        builder.set_buffer_write(0, buffers.pre_output_hidden_states, 0);
        builder.set_buffer_read(1, buffers.recurrent_output, 0);
        builder.set_buffer_read(2, buffers.z, 0);
        builder.set_buffer_read(3, buffers.norm_weight, 0);
        builder.set_f32(4, eps);
        set_batch_args(builder, shape, 5);
        builder.dispatch_1d(
            self.config.total_output_norm_gate_threads(shape),
            OUTPUT_NORM_GATE_NUM_THREADS_PER_THREADBLOCK,
        );
    }
}

pub struct GDNCoreInvocation<'a> {
    kernels: &'a GDNCoreKernels,
    shape: GDNCoreShape,
    buffers: GDNCoreBuffers<'a>,
    q_scale: f32,
    eps: f32,
}

impl Operator for GDNCoreInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.kernels.config.validate_shape(self.shape);
        validate_buffers(self.kernels.config, self.shape, &self.buffers);
        self.kernels.record_short_conv(builder, self.shape, &self.buffers);
        assert!(
            self.shape.num_tokens >= self.shape.num_reqs,
            "GDN ragged recurrent requires at least one token per request"
        );
        self.kernels
            .record_ragged_recurrent(builder, self.shape, &self.buffers, self.q_scale);
        self.kernels
            .record_output_norm_gate(builder, self.shape, &self.buffers, self.eps);
    }
}

pub struct GDNCoreForwardCandidateStateUpdateInvocation<'a> {
    kernels: &'a GDNCoreKernels,
    shape: GDNCoreShape,
    buffers: GDNCoreForwardCandidateStateUpdateBuffers<'a>,
    q_scale: f32,
    eps: f32,
}

impl Operator for GDNCoreForwardCandidateStateUpdateInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.kernels.config.validate_shape(self.shape);
        assert_u32_count_domain(
            self.kernels.config.num_candidate_conv_state_values(self.shape),
            "GDN candidate convolution state",
        );
        validate_buffers(self.kernels.config, self.shape, &self.buffers.core);
        assert!(
            self.buffers.flat_candidate_state_slots.len_bytes() >= self.shape.num_tokens as usize * size_of::<u32>(),
            "GDN flat_candidate_state_slots buffer is too small"
        );
        self.kernels.record_short_conv(builder, self.shape, &self.buffers.core);
        self.kernels
            .record_forward_conv_candidate_state(builder, self.shape, &self.buffers);
        self.kernels
            .record_ragged_recurrent_forward_candidate_state(builder, self.shape, &self.buffers, self.q_scale);
        self.kernels
            .record_output_norm_gate(builder, self.shape, &self.buffers.core, self.eps);
    }
}

fn set_batch_args(builder: &CommandRecorder, shape: GDNCoreShape, start_index: usize) {
    builder.set_u32(start_index, shape.num_reqs);
    builder.set_u32(start_index + 1, shape.num_tokens);
}

fn validate_buffers(config: GDNCoreConfig, shape: GDNCoreShape, buffers: &GDNCoreBuffers<'_>) {
    let f32_bytes = u64::try_from(size_of::<f32>()).expect("f32 item size must fit u64");
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
        buffers.projected_qkv.len_bytes() >= config.num_qkv_values(shape) * size_of::<f32>(),
        "GDN projected_qkv buffer is too small"
    );
    assert!(
        buffers.a.len_bytes() >= shape.num_tokens as usize * config.num_v_heads as usize * size_of::<f32>(),
        "GDN a buffer is too small"
    );
    assert!(
        buffers.b.len_bytes() >= shape.num_tokens as usize * config.num_v_heads as usize * size_of::<f32>(),
        "GDN b buffer is too small"
    );
    assert!(
        buffers.z.len_bytes() >= config.num_recurrent_output_values(shape) * size_of::<f32>(),
        "GDN z buffer is too small"
    );
    assert_eq!(
        buffers.conv_weight.len_bytes(),
        config.num_conv_weight_values() * size_of::<u16>()
    );
    assert_eq!(
        buffers.norm_weight.len_bytes(),
        config.v_head_dim as usize * size_of::<u16>()
    );
    assert_eq!(
        buffers.a_log.len_bytes(),
        config.num_v_heads as usize * size_of::<u16>()
    );
    assert_eq!(
        buffers.dt_bias.len_bytes(),
        config.num_v_heads as usize * size_of::<u16>()
    );
    assert!(
        buffers.cu_tokens.len_bytes_u64()
            >= (u64::from(shape.num_reqs) + 1)
                .checked_mul(size_of::<u32>().try_into().expect("u32 item size must fit u64"))
                .expect("GDN cumulative-token byte length must fit u64"),
        "GDN cu_tokens buffer is too small"
    );
    assert!(
        buffers.src_state_slots.len_bytes() >= shape.num_reqs as usize * size_of::<u32>(),
        "GDN src_state_slots buffer is too small"
    );
    assert!(
        buffers.dst_state_slots.len_bytes() >= shape.num_reqs as usize * size_of::<u32>(),
        "GDN dst_state_slots buffer is too small"
    );
    let conv_state_region_bytes = u64::try_from(config.num_conv_state_values(shape))
        .expect("GDN convolution state element count must fit u64")
        .checked_mul(f32_bytes)
        .expect("GDN convolution state region bytes must fit u64");
    let recurrent_state_region_bytes = u64::try_from(config.recurrent_state_stride())
        .expect("GDN recurrent state stride must fit u64")
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
        buffers.conv_qkv.len_bytes() >= config.num_qkv_values(shape) * size_of::<f32>(),
        "GDN conv_qkv buffer is too small"
    );
    assert!(
        buffers.recurrent_output.len_bytes() >= config.num_recurrent_output_values(shape) * size_of::<f32>(),
        "GDN recurrent_output buffer is too small"
    );
    assert!(
        buffers.pre_output_hidden_states.len_bytes() >= config.num_recurrent_output_values(shape) * size_of::<f32>(),
        "GDN pre_output_hidden_states buffer is too small"
    );
}

#[cfg(test)]
#[path = "gdn_attention_test.rs"]
mod tests;
