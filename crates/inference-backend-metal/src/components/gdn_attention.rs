use super::assert_u32_count_domain;
use super::checked_product;
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
    pub a_log_decay: &'a Buffer,
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
        builder.set_buffer_read(5, buffers.a_log_decay, 0);
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
        builder.set_buffer_read(5, core.a_log_decay, 0);
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
        config.num_conv_weight_values() * size_of::<f32>()
    );
    assert_eq!(
        buffers.norm_weight.len_bytes(),
        config.v_head_dim as usize * size_of::<f32>()
    );
    assert_eq!(
        buffers.a_log_decay.len_bytes(),
        config.num_v_heads as usize * size_of::<f32>()
    );
    assert_eq!(
        buffers.dt_bias.len_bytes(),
        config.num_v_heads as usize * size_of::<f32>()
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
mod tests {
    use inference_executor_core::attn::gdn::GDNCore;
    use inference_executor_core::attn::gdn::reference::GDNRecurrentReferenceInput;
    use inference_executor_core::attn::gdn::reference::gdn_recurrent_reference;
    use inference_executor_core::attn::gdn::reference::gdn_short_conv_reference;

    use super::*;
    use crate::metal::Dtype;
    use crate::metal::Stream;

    #[test]
    #[should_panic(expected = "GDN convolution exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        GDNCoreConfig {
            num_qk_heads: 1,
            qk_head_dim: 1,
            num_v_heads: 1,
            v_head_dim: 2,
            conv_kernel_size: 2,
            v_dim_tile_size: 1,
        }
        .validate_shape(GDNCoreShape {
            num_reqs: 1,
            num_tokens: 1 << 30,
        });
    }

    #[test]
    fn test_ragged_recurrent_fixed() {
        let shape = fixture_shape(1, 1);
        let cu_tokens = vec![0_u32, 1];
        let src_state_slots = vec![0_u32];
        let dst_slot_ids = vec![0_u32];
        let projected_qkv = fixture_values(fixture_config().num_qkv_values(shape), 0.03125, 3);
        let conv_state = fixture_values(fixture_config().num_conv_state_values(shape), 0.015625, 7);
        let recurrent_state = fixture_values(fixture_config().recurrent_state_stride(), 0.0078125, 11);
        let conv_weight = fixture_values(
            fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
            0.00390625,
            13,
        );
        let a = fixture_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            0.0625,
            17,
        );
        let b = fixture_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            0.0625,
            19,
        );
        let z = fixture_values(fixture_config().num_recurrent_output_values(shape), 0.03125, 23);
        let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
        let a_log_decay = vec![-0.25_f32; fixture_config().num_v_heads as usize];
        let dt_bias = vec![0.125_f32; fixture_config().num_v_heads as usize];

        let actual = run_gdn_core(
            shape,
            &projected_qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &a,
            &b,
            &z,
            &norm_weight,
            &a_log_decay,
            &dt_bias,
            &cu_tokens,
            &src_state_slots,
            &dst_slot_ids,
        );
        assert_gdn_reference_matches(
            shape,
            &cu_tokens,
            &projected_qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &a,
            &b,
            &a_log_decay,
            &dt_bias,
            &actual,
            2.0e-5,
        );
    }

    #[test]
    fn test_ragged_recurrent_random() {
        let random_seed = 0x729B_40D6;
        let shape = fixture_shape(1, 3);
        let cu_tokens = vec![0_u32, 3];
        let src_state_slots = vec![0_u32];
        let dst_slot_ids = vec![0_u32];
        let projected_qkv = generated_values(fixture_config().num_qkv_values(shape), random_seed);
        let conv_state = generated_values(
            fixture_config().num_conv_state_values(shape),
            random_seed.wrapping_add(1),
        );
        let recurrent_state = generated_values(fixture_config().recurrent_state_stride(), random_seed.wrapping_add(2));
        let conv_weight = generated_values(
            fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
            random_seed.wrapping_add(3),
        )
        .into_iter()
        .map(|value| value * 0.125)
        .collect::<Vec<_>>();
        let a = generated_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            random_seed.wrapping_add(4),
        );
        let b = generated_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            random_seed.wrapping_add(5),
        );
        let z = generated_values(
            fixture_config().num_recurrent_output_values(shape),
            random_seed.wrapping_add(6),
        );
        let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
        let a_log_decay = vec![-0.125_f32; fixture_config().num_v_heads as usize];
        let dt_bias = vec![0.0625_f32; fixture_config().num_v_heads as usize];

        let actual = run_gdn_core(
            shape,
            &projected_qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &a,
            &b,
            &z,
            &norm_weight,
            &a_log_decay,
            &dt_bias,
            &cu_tokens,
            &src_state_slots,
            &dst_slot_ids,
        );
        assert_gdn_reference_matches(
            shape,
            &cu_tokens,
            &projected_qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &a,
            &b,
            &a_log_decay,
            &dt_bias,
            &actual,
            4.0e-5,
        );
    }

    #[test]
    fn test_ragged_multi_random() {
        let random_seed = 0xE13C_58A4;
        let shape = fixture_shape(2, 4);
        let cu_tokens = vec![0_u32, 1, 4];
        let src_state_slots = vec![0_u32, 1];
        let dst_slot_ids = vec![0_u32, 1];
        let projected_qkv = generated_values(fixture_config().num_qkv_values(shape), random_seed);
        let conv_state = generated_values(
            fixture_config().num_conv_state_values(shape),
            random_seed.wrapping_add(1),
        );
        let recurrent_state = generated_values(
            shape.num_reqs as usize * fixture_config().recurrent_state_stride(),
            random_seed.wrapping_add(2),
        );
        let conv_weight = generated_values(
            fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
            random_seed.wrapping_add(3),
        )
        .into_iter()
        .map(|value| value * 0.125)
        .collect::<Vec<_>>();
        let a = generated_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            random_seed.wrapping_add(4),
        );
        let b = generated_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            random_seed.wrapping_add(5),
        );
        let z = generated_values(
            fixture_config().num_recurrent_output_values(shape),
            random_seed.wrapping_add(6),
        );
        let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
        let a_log_decay = vec![-0.125_f32; fixture_config().num_v_heads as usize];
        let dt_bias = vec![0.0625_f32; fixture_config().num_v_heads as usize];

        let actual = run_gdn_core(
            shape,
            &projected_qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &a,
            &b,
            &z,
            &norm_weight,
            &a_log_decay,
            &dt_bias,
            &cu_tokens,
            &src_state_slots,
            &dst_slot_ids,
        );
        assert_gdn_reference_matches(
            shape,
            &cu_tokens,
            &projected_qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &a,
            &b,
            &a_log_decay,
            &dt_bias,
            &actual,
            4.0e-5,
        );
    }

    #[test]
    fn test_candidate_state_prefixes() {
        let shape = fixture_shape(1, 3);
        let cu_tokens = vec![0_u32, 3];
        let src_state_slots = vec![0_u32];
        let dst_slot_ids = vec![4_u32];
        let candidate_dst_slot_ids = vec![1_u32, 2, 3];
        let projected_qkv = fixture_values(fixture_config().num_qkv_values(shape), 0.03125, 29);
        let conv_state = fixture_values(fixture_config().num_conv_state_values(shape), 0.015625, 31);
        let recurrent_state = fixture_values(fixture_config().recurrent_state_stride(), 0.0078125, 37);
        let conv_weight = fixture_values(
            fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
            0.00390625,
            41,
        );
        let a = fixture_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            0.0625,
            43,
        );
        let b = fixture_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            0.0625,
            47,
        );
        let z = fixture_values(fixture_config().num_recurrent_output_values(shape), 0.03125, 53);
        let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
        let a_log_decay = vec![-0.25_f32; fixture_config().num_v_heads as usize];
        let dt_bias = vec![0.125_f32; fixture_config().num_v_heads as usize];

        let actual = run_gdn_forward_candidate_state(
            shape,
            &projected_qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &a,
            &b,
            &z,
            &norm_weight,
            &a_log_decay,
            &dt_bias,
            &cu_tokens,
            &src_state_slots,
            &dst_slot_ids,
            &candidate_dst_slot_ids,
            5,
        );

        assert_gdn_reference_matches(
            shape,
            &cu_tokens,
            &projected_qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &a,
            &b,
            &a_log_decay,
            &dt_bias,
            &actual.full,
            2.0e-5,
        );
        let core = fixture_core(shape);
        for verified_tokens in 1..=shape.num_tokens as usize {
            let prefix_cu_tokens = [0_u32, verified_tokens as u32];
            let conv_reference = gdn_short_conv_reference(
                &core,
                &prefix_cu_tokens,
                &conv_state,
                &projected_qkv[..verified_tokens * fixture_config().qkv_dim() as usize],
                &conv_weight,
            );
            let recurrent_reference = gdn_recurrent_reference(
                &core,
                GDNRecurrentReferenceInput {
                    cu_tokens: &prefix_cu_tokens,
                    source_recurrent_state: &recurrent_state,
                    conv_qkv: &conv_reference.conv_qkv,
                    a: &a[..verified_tokens * fixture_config().num_v_heads as usize],
                    b: &b[..verified_tokens * fixture_config().num_v_heads as usize],
                    a_log_decay: &a_log_decay,
                    dt_bias: &dt_bias,
                },
            );
            let slot = candidate_dst_slot_ids[verified_tokens - 1] as usize;
            assert_close(
                conv_state_slot(&actual.next_conv_state_arena, shape, slot),
                &conv_reference.next_conv_state,
                2.0e-5,
            );
            assert_close(
                recurrent_state_slot(&actual.recurrent_state_arena, shape, slot),
                &recurrent_reference.next_recurrent_state,
                2.0e-5,
            );
        }
    }

    #[test]
    fn test_candidate_states_above_u32_byte_offset() {
        let shape = fixture_shape(1, 3);
        let num_state_slots = 5usize;
        let cu_tokens_values = [0_u32, 3];
        let src_state_slot_values = [0_u32];
        let dst_slot_id_values = [4_u32];
        let candidate_dst_slot_id_values = [1_u32, 2, 3];
        let projected_qkv_values = fixture_values(fixture_config().num_qkv_values(shape), 0.03125, 29);
        let conv_state_values = fixture_values(fixture_config().num_conv_state_values(shape), 0.015625, 31);
        let recurrent_state_values = fixture_values(fixture_config().recurrent_state_stride(), 0.0078125, 37);
        let conv_weight_values = fixture_values(
            fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
            0.00390625,
            41,
        );
        let a_values = fixture_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            0.0625,
            43,
        );
        let b_values = fixture_values(
            shape.num_tokens as usize * fixture_config().num_v_heads as usize,
            0.0625,
            47,
        );
        let z_values = fixture_values(fixture_config().num_recurrent_output_values(shape), 0.03125, 53);
        let norm_weight_values = vec![1.0_f32; fixture_config().v_head_dim as usize];
        let a_log_decay_values = vec![-0.25_f32; fixture_config().num_v_heads as usize];
        let dt_bias_values = vec![0.125_f32; fixture_config().num_v_heads as usize];

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernels = GDNCoreKernels::new(&device, fixture_config());
        let projected_qkv = Buffer::from_slice(&device, &projected_qkv_values);
        let a = Buffer::from_slice(&device, &a_values);
        let b = Buffer::from_slice(&device, &b_values);
        let z = Buffer::from_slice(&device, &z_values);
        let conv_weight = Buffer::from_slice(&device, &conv_weight_values);
        let norm_weight = Buffer::from_slice(&device, &norm_weight_values);
        let a_log_decay = Buffer::from_slice(&device, &a_log_decay_values);
        let dt_bias = Buffer::from_slice(&device, &dt_bias_values);
        let cu_tokens = Buffer::from_slice(&device, &cu_tokens_values);
        let src_state_slots = Buffer::from_slice(&device, &src_state_slot_values);
        let dst_slot_ids = Buffer::from_slice(&device, &dst_slot_id_values);
        let candidate_dst_slot_ids = Buffer::from_slice(&device, &candidate_dst_slot_id_values);

        let high_base = u64::from(u32::MAX) + 1;
        let conv_state_offset_bytes = high_base;
        let next_conv_state_offset_bytes = high_base + 4096;
        let recurrent_state_offset_bytes = high_base + 8192;
        let recurrent_arena_bytes = u64::try_from(num_state_slots)
            .expect("test state-slot count must fit u64")
            .checked_mul(
                u64::try_from(
                    fixture_config()
                        .recurrent_state_stride()
                        .checked_mul(size_of::<f32>())
                        .expect("test recurrent-state byte stride must fit usize"),
                )
                .expect("test recurrent-state byte stride must fit u64"),
            )
            .expect("test recurrent-state arena byte length must fit u64");
        let state_arena = Buffer::new_uninit(
            &device,
            recurrent_state_offset_bytes
                .checked_add(recurrent_arena_bytes)
                .expect("test state arena byte length must fit u64"),
        );
        state_arena.write_typed(
            usize::try_from(conv_state_offset_bytes / size_of::<f32>() as u64)
                .expect("test convolution state offset must fit usize"),
            &conv_state_values,
        );
        state_arena.write_typed(
            usize::try_from(recurrent_state_offset_bytes / size_of::<f32>() as u64)
                .expect("test recurrent state offset must fit usize"),
            &recurrent_state_values,
        );

        let conv_qkv = Buffer::new_zeroed_elements(&device, fixture_config().num_qkv_values(shape), Dtype::Float32);
        let recurrent_output = Buffer::new_zeroed_elements(
            &device,
            fixture_config().num_recurrent_output_values(shape),
            Dtype::Float32,
        );
        let pre_output_hidden_states = Buffer::new_zeroed_elements(
            &device,
            fixture_config().num_recurrent_output_values(shape),
            Dtype::Float32,
        );
        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke_forward_candidate_state_update(
            shape,
            GDNCoreForwardCandidateStateUpdateBuffers {
                core: GDNCoreBuffers {
                    projected_qkv: &projected_qkv,
                    a: &a,
                    b: &b,
                    z: &z,
                    conv_weight: &conv_weight,
                    norm_weight: &norm_weight,
                    a_log_decay: &a_log_decay,
                    dt_bias: &dt_bias,
                    cu_tokens: &cu_tokens,
                    src_state_slots: &src_state_slots,
                    dst_state_slots: &dst_slot_ids,
                    conv_state: &state_arena,
                    conv_state_offset_bytes,
                    next_conv_state: &state_arena,
                    next_conv_state_offset_bytes,
                    recurrent_state_arena: &state_arena,
                    recurrent_state_arena_offset_bytes: recurrent_state_offset_bytes,
                    conv_qkv: &conv_qkv,
                    recurrent_output: &recurrent_output,
                    pre_output_hidden_states: &pre_output_hidden_states,
                },
                flat_candidate_state_slots: &candidate_dst_slot_ids,
            },
            1.0,
            1.0e-6,
        ));
        stream.submit_replay(&builder.build()).wait();

        let next_conv_state_arena = state_arena.read_typed::<f32>(
            usize::try_from(next_conv_state_offset_bytes / size_of::<f32>() as u64)
                .expect("test next convolution state offset must fit usize"),
            num_state_slots * fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize,
        );
        let recurrent_state_arena = state_arena.read_typed::<f32>(
            usize::try_from(recurrent_state_offset_bytes / size_of::<f32>() as u64)
                .expect("test recurrent state offset must fit usize"),
            num_state_slots * fixture_config().recurrent_state_stride(),
        );
        let core = fixture_core(shape);
        for verified_tokens in 1..=shape.num_tokens as usize {
            let prefix_cu_tokens = [0_u32, verified_tokens as u32];
            let conv_reference = gdn_short_conv_reference(
                &core,
                &prefix_cu_tokens,
                &conv_state_values,
                &projected_qkv_values[..verified_tokens * fixture_config().qkv_dim() as usize],
                &conv_weight_values,
            );
            let recurrent_reference = gdn_recurrent_reference(
                &core,
                GDNRecurrentReferenceInput {
                    cu_tokens: &prefix_cu_tokens,
                    source_recurrent_state: &recurrent_state_values,
                    conv_qkv: &conv_reference.conv_qkv,
                    a: &a_values[..verified_tokens * fixture_config().num_v_heads as usize],
                    b: &b_values[..verified_tokens * fixture_config().num_v_heads as usize],
                    a_log_decay: &a_log_decay_values,
                    dt_bias: &dt_bias_values,
                },
            );
            let slot = candidate_dst_slot_id_values[verified_tokens - 1] as usize;
            assert_close(
                conv_state_slot(&next_conv_state_arena, shape, slot),
                &conv_reference.next_conv_state,
                2.0e-5,
            );
            assert_close(
                recurrent_state_slot(&recurrent_state_arena, shape, slot),
                &recurrent_reference.next_recurrent_state,
                2.0e-5,
            );
        }
    }

    struct GDNCoreOutputs {
        conv_qkv: Vec<f32>,
        next_conv_state: Vec<f32>,
        recurrent_output: Vec<f32>,
        recurrent_state: Vec<f32>,
    }

    struct GDNForwardCandidateStateOutputs {
        full: GDNCoreOutputs,
        next_conv_state_arena: Vec<f32>,
        recurrent_state_arena: Vec<f32>,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gdn_core(
        shape: GDNCoreShape,
        projected_qkv_values: &[f32],
        conv_state_values: &[f32],
        recurrent_state_values: &[f32],
        conv_weight_values: &[f32],
        a_values: &[f32],
        b_values: &[f32],
        z_values: &[f32],
        norm_weight_values: &[f32],
        a_log_decay_values: &[f32],
        dt_bias_values: &[f32],
        cu_tokens_values: &[u32],
        src_state_slot_values: &[u32],
        dst_slot_id_values: &[u32],
    ) -> GDNCoreOutputs {
        const STATE_PREFIX_VALUES: usize = 7;
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernels = GDNCoreKernels::new(&device, fixture_config());
        let projected_qkv = Buffer::from_slice(&device, projected_qkv_values);
        let a = Buffer::from_slice(&device, a_values);
        let b = Buffer::from_slice(&device, b_values);
        let z = Buffer::from_slice(&device, z_values);
        let conv_weight = Buffer::from_slice(&device, conv_weight_values);
        let norm_weight = Buffer::from_slice(&device, norm_weight_values);
        let a_log_decay = Buffer::from_slice(&device, a_log_decay_values);
        let dt_bias = Buffer::from_slice(&device, dt_bias_values);
        let cu_tokens = Buffer::from_slice(&device, cu_tokens_values);
        let src_state_slots = Buffer::from_slice(&device, src_state_slot_values);
        let dst_slot_ids = Buffer::from_slice(&device, dst_slot_id_values);
        let mut conv_state_values_with_prefix = vec![-1.0; STATE_PREFIX_VALUES];
        conv_state_values_with_prefix.extend_from_slice(conv_state_values);
        let conv_state = Buffer::from_slice(&device, &conv_state_values_with_prefix);
        let next_conv_state = Buffer::new_zeroed(
            &device,
            (STATE_PREFIX_VALUES + fixture_config().num_conv_state_values(shape)) * size_of::<f32>(),
        );
        let mut recurrent_state_values_with_prefix = vec![-1.0; STATE_PREFIX_VALUES];
        recurrent_state_values_with_prefix.extend_from_slice(recurrent_state_values);
        let recurrent_state_arena = Buffer::from_slice(&device, &recurrent_state_values_with_prefix);
        let state_offset_bytes =
            u64::try_from(STATE_PREFIX_VALUES * size_of::<f32>()).expect("test GDN state offset must fit u64");
        let conv_qkv = Buffer::new_zeroed(&device, fixture_config().num_qkv_values(shape) * size_of::<f32>());
        let recurrent_output = Buffer::new_zeroed(
            &device,
            fixture_config().num_recurrent_output_values(shape) * size_of::<f32>(),
        );
        let pre_output_hidden_states = Buffer::new_zeroed(
            &device,
            fixture_config().num_recurrent_output_values(shape) * size_of::<f32>(),
        );
        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke(
            shape,
            GDNCoreBuffers {
                projected_qkv: &projected_qkv,
                a: &a,
                b: &b,
                z: &z,
                conv_weight: &conv_weight,
                norm_weight: &norm_weight,
                a_log_decay: &a_log_decay,
                dt_bias: &dt_bias,
                cu_tokens: &cu_tokens,
                src_state_slots: &src_state_slots,
                dst_state_slots: &dst_slot_ids,
                conv_state: &conv_state,
                conv_state_offset_bytes: state_offset_bytes,
                next_conv_state: &next_conv_state,
                next_conv_state_offset_bytes: state_offset_bytes,
                recurrent_state_arena: &recurrent_state_arena,
                recurrent_state_arena_offset_bytes: state_offset_bytes,
                conv_qkv: &conv_qkv,
                recurrent_output: &recurrent_output,
                pre_output_hidden_states: &pre_output_hidden_states,
            },
            1.0,
            1.0e-6,
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        GDNCoreOutputs {
            conv_qkv: conv_qkv.read_typed::<f32>(0, fixture_config().num_qkv_values(shape)),
            next_conv_state: next_conv_state
                .read_typed::<f32>(STATE_PREFIX_VALUES, fixture_config().num_conv_state_values(shape)),
            recurrent_output: recurrent_output
                .read_typed::<f32>(0, fixture_config().num_recurrent_output_values(shape)),
            recurrent_state: recurrent_state_arena.read_typed::<f32>(
                STATE_PREFIX_VALUES,
                shape.num_reqs as usize * fixture_config().recurrent_state_stride(),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gdn_forward_candidate_state(
        shape: GDNCoreShape,
        projected_qkv_values: &[f32],
        conv_state_values: &[f32],
        recurrent_state_values: &[f32],
        conv_weight_values: &[f32],
        a_values: &[f32],
        b_values: &[f32],
        z_values: &[f32],
        norm_weight_values: &[f32],
        a_log_decay_values: &[f32],
        dt_bias_values: &[f32],
        cu_tokens_values: &[u32],
        src_state_slot_values: &[u32],
        dst_slot_id_values: &[u32],
        candidate_dst_slot_id_values: &[u32],
        num_state_slots: usize,
    ) -> GDNForwardCandidateStateOutputs {
        const STATE_PREFIX_VALUES: usize = 7;
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernels = GDNCoreKernels::new(&device, fixture_config());
        let projected_qkv = Buffer::from_slice(&device, projected_qkv_values);
        let a = Buffer::from_slice(&device, a_values);
        let b = Buffer::from_slice(&device, b_values);
        let z = Buffer::from_slice(&device, z_values);
        let conv_weight = Buffer::from_slice(&device, conv_weight_values);
        let norm_weight = Buffer::from_slice(&device, norm_weight_values);
        let a_log_decay = Buffer::from_slice(&device, a_log_decay_values);
        let dt_bias = Buffer::from_slice(&device, dt_bias_values);
        let cu_tokens = Buffer::from_slice(&device, cu_tokens_values);
        let src_state_slots = Buffer::from_slice(&device, src_state_slot_values);
        let dst_slot_ids = Buffer::from_slice(&device, dst_slot_id_values);
        let candidate_dst_slot_ids = Buffer::from_slice(&device, candidate_dst_slot_id_values);
        let mut conv_state_values_with_prefix = vec![-1.0; STATE_PREFIX_VALUES];
        conv_state_values_with_prefix.extend_from_slice(conv_state_values);
        let conv_state = Buffer::from_slice(&device, &conv_state_values_with_prefix);
        let next_conv_state = Buffer::new_zeroed(
            &device,
            (STATE_PREFIX_VALUES
                + num_state_slots * fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize)
                * size_of::<f32>(),
        );
        let mut recurrent_state_arena_values =
            vec![0.0_f32; STATE_PREFIX_VALUES + num_state_slots * fixture_config().recurrent_state_stride()];
        recurrent_state_arena_values
            [STATE_PREFIX_VALUES..STATE_PREFIX_VALUES + fixture_config().recurrent_state_stride()]
            .copy_from_slice(recurrent_state_values);
        let recurrent_state_arena = Buffer::from_slice(&device, &recurrent_state_arena_values);
        let state_offset_bytes =
            u64::try_from(STATE_PREFIX_VALUES * size_of::<f32>()).expect("test GDN state offset must fit u64");
        let conv_qkv = Buffer::new_zeroed(&device, fixture_config().num_qkv_values(shape) * size_of::<f32>());
        let recurrent_output = Buffer::new_zeroed(
            &device,
            fixture_config().num_recurrent_output_values(shape) * size_of::<f32>(),
        );
        let pre_output_hidden_states = Buffer::new_zeroed(
            &device,
            fixture_config().num_recurrent_output_values(shape) * size_of::<f32>(),
        );

        let mut builder = stream.create_replay_program();
        let core = GDNCoreBuffers {
            projected_qkv: &projected_qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log_decay: &a_log_decay,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_state_slots: &src_state_slots,
            dst_state_slots: &dst_slot_ids,
            conv_state: &conv_state,
            conv_state_offset_bytes: state_offset_bytes,
            next_conv_state: &next_conv_state,
            next_conv_state_offset_bytes: state_offset_bytes,
            recurrent_state_arena: &recurrent_state_arena,
            recurrent_state_arena_offset_bytes: state_offset_bytes,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            pre_output_hidden_states: &pre_output_hidden_states,
        };
        builder.record(kernels.invoke_forward_candidate_state_update(
            shape,
            GDNCoreForwardCandidateStateUpdateBuffers {
                core,
                flat_candidate_state_slots: &candidate_dst_slot_ids,
            },
            1.0,
            1.0e-6,
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        GDNForwardCandidateStateOutputs {
            full: GDNCoreOutputs {
                conv_qkv: conv_qkv.read_typed::<f32>(0, fixture_config().num_qkv_values(shape)),
                next_conv_state: next_conv_state.read_typed::<f32>(
                    STATE_PREFIX_VALUES
                        + dst_slot_id_values[0] as usize
                            * fixture_config().qkv_dim() as usize
                            * fixture_config().conv_state_len() as usize,
                    fixture_config().num_conv_state_values(shape),
                ),
                recurrent_output: recurrent_output
                    .read_typed::<f32>(0, fixture_config().num_recurrent_output_values(shape)),
                recurrent_state: recurrent_state_arena.read_typed::<f32>(
                    STATE_PREFIX_VALUES + dst_slot_id_values[0] as usize * fixture_config().recurrent_state_stride(),
                    fixture_config().recurrent_state_stride(),
                ),
            },
            next_conv_state_arena: next_conv_state.read_typed::<f32>(
                STATE_PREFIX_VALUES,
                num_state_slots * fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize,
            ),
            recurrent_state_arena: recurrent_state_arena.read_typed::<f32>(
                STATE_PREFIX_VALUES,
                num_state_slots * fixture_config().recurrent_state_stride(),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_gdn_reference_matches(
        shape: GDNCoreShape,
        cu_tokens: &[u32],
        projected_qkv: &[f32],
        conv_state: &[f32],
        recurrent_state: &[f32],
        conv_weight: &[f32],
        a: &[f32],
        b: &[f32],
        a_log_decay: &[f32],
        dt_bias: &[f32],
        actual: &GDNCoreOutputs,
        tolerance: f32,
    ) {
        let core = fixture_core(shape);
        let conv_reference = gdn_short_conv_reference(&core, cu_tokens, conv_state, projected_qkv, conv_weight);
        let recurrent_reference = gdn_recurrent_reference(
            &core,
            GDNRecurrentReferenceInput {
                cu_tokens,
                source_recurrent_state: recurrent_state,
                conv_qkv: &conv_reference.conv_qkv,
                a,
                b,
                a_log_decay,
                dt_bias,
            },
        );

        assert_close(&actual.conv_qkv, &conv_reference.conv_qkv, tolerance);
        assert_close(&actual.next_conv_state, &conv_reference.next_conv_state, tolerance);
        assert_close(
            &actual.recurrent_output,
            &recurrent_reference.recurrent_output,
            tolerance,
        );
        assert_close(
            &actual.recurrent_state,
            &recurrent_reference.next_recurrent_state,
            tolerance,
        );
    }

    fn fixture_shape(num_reqs: u32, num_tokens: u32) -> GDNCoreShape {
        GDNCoreShape { num_reqs, num_tokens }
    }

    fn fixture_config() -> GDNCoreConfig {
        GDNCoreConfig {
            num_qk_heads: 1,
            qk_head_dim: 4,
            num_v_heads: 1,
            v_head_dim: 8,
            conv_kernel_size: 3,
            v_dim_tile_size: 8,
        }
    }

    fn fixture_core(_shape: GDNCoreShape) -> GDNCore {
        GDNCore {
            model_layer_index: 0,
            hidden_dim: fixture_config().num_v_heads as usize * fixture_config().v_head_dim as usize,
            num_qk_heads: fixture_config().num_qk_heads as usize,
            qk_head_dim: fixture_config().qk_head_dim as usize,
            num_v_heads: fixture_config().num_v_heads as usize,
            v_head_dim: fixture_config().v_head_dim as usize,
            conv_kernel_size: fixture_config().conv_kernel_size as usize,
            q_scale: 1.0,
        }
    }

    fn conv_state_slot(arena: &[f32], _shape: GDNCoreShape, state_slot: usize) -> &[f32] {
        let conv_state_stride = fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize;
        &arena[state_slot * conv_state_stride..(state_slot + 1) * conv_state_stride]
    }

    fn recurrent_state_slot(arena: &[f32], _shape: GDNCoreShape, state_slot: usize) -> &[f32] {
        let recurrent_state_stride = fixture_config().recurrent_state_stride();
        &arena[state_slot * recurrent_state_stride..(state_slot + 1) * recurrent_state_stride]
    }

    fn fixture_values(count: usize, scale: f32, pattern_offset: usize) -> Vec<f32> {
        (0..count)
            .map(|index| ((index * 17 + pattern_offset) % 29) as f32 * scale - 14.0 * scale)
            .collect()
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

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual_value - expected_value).abs();
            assert!(
                diff <= tolerance,
                "GDN reference mismatch at {index}: expected={expected_value} actual={actual_value} diff={diff} \
                 tolerance={tolerance}"
            );
        }
    }
}
