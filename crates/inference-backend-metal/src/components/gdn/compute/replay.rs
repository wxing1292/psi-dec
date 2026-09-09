use super::Buffers;
use super::Compute;
use super::Shape;
use super::Variant;
use super::VariantConstants;
use super::set_chunkwise_count;
use super::set_replay_u32;
use super::validate_buffers;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../../metal/gdn_compute_replay.metal");

/// Per-layer inputs that survive forward until the accepted state is committed.
///
/// Alpha, normalized K, and U use F32. Raw pre-convolution QKV uses BF16.
/// `token_offset` selects this layer in the all-layer log.
/// `num_total_tokens` is the log capacity, independent of the full forward shape.
/// The caller must validate that the active replay suffix fits this capacity.
#[derive(Clone, Copy)]
pub struct ReplayBuffers<'a> {
    pub alpha: &'a Buffer,
    pub k: &'a Buffer,
    pub u: &'a Buffer,
    pub qkv: &'a Buffer,
    pub token_offset: u64,
    pub num_total_tokens: u32,
}

pub fn source(constants: VariantConstants) -> String {
    let thread_block = constants.kernels.candidate_recurrent_state.thread_block;
    format!(
        "constant uint replay_num_qk_dim_threads = {qk_threads}u;\nconstant uint replay_num_v_rows_per_simdgroup = \
         {v_rows}u;\nconstant uint replay_num_simdgroups = {simdgroups}u;\n{SOURCE}",
        qk_threads = thread_block.num_qk_dim_threads,
        v_rows = thread_block.simdgroup.num_v_rows,
        simdgroups = thread_block.num_simdgroups,
    )
}

impl Compute {
    /// Chunkwise writes final/boundary states. The suffix writes a replay log.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_replay<'a>(
        &'a self,
        shape: Shape,
        buffers: Buffers<'a>,
        replay: ReplayBuffers<'a>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
        num_active_chunkwise_requests: ReplayU32,
    ) -> ReplayInvocation<'a> {
        let (_, variant) = self.select(shape);
        ReplayInvocation {
            variant,
            shape,
            buffers,
            replay,
            num_active_reqs,
            num_active_tokens,
            num_active_chunkwise_requests,
        }
    }
}

pub struct ReplayInvocation<'a> {
    variant: &'a Variant,
    shape: Shape,
    buffers: Buffers<'a>,
    replay: ReplayBuffers<'a>,
    num_active_reqs: ReplayU32,
    num_active_tokens: ReplayU32,
    num_active_chunkwise_requests: ReplayU32,
}

impl Operator for ReplayInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let variant = self.variant;
        variant.constants.validate_shape(self.shape);
        validate_buffers(variant.constants, self.shape, &self.buffers);
        assert!(self.replay.num_total_tokens > 0);
        let model = variant.constants.model;
        for (buffer, row_bytes) in [
            (self.replay.alpha, model.num_v_heads as u64 * size_of::<f32>() as u64),
            (
                self.replay.k,
                model.num_qk_heads as u64 * model.qk_head_dim as u64 * size_of::<f32>() as u64,
            ),
            (
                self.replay.u,
                model.num_v_heads as u64 * model.v_head_dim as u64 * size_of::<f32>() as u64,
            ),
            (self.replay.qkv, model.qkv_dim() as u64 * size_of::<u16>() as u64),
        ] {
            validate_log_range(
                buffer.len_bytes_u64(),
                row_bytes,
                self.replay.token_offset,
                self.replay.num_total_tokens,
            );
        }
        if let (ReplayU32::Fixed(num_active_reqs), ReplayU32::Fixed(num_active_chunkwise_requests)) =
            (self.num_active_reqs, self.num_active_chunkwise_requests)
        {
            assert!(num_active_chunkwise_requests <= num_active_reqs);
        }
        recorder.set_kernel(&variant.short_conv_replay);
        variant.record_short_conv_bindings(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
            false,
        );
        recorder.set_buffer_write(13, self.replay.qkv, 0);
        recorder.set_u64(14, self.replay.token_offset);
        set_chunkwise_count(
            recorder,
            15,
            self.num_active_chunkwise_requests,
            self.shape.num_total_reqs,
        );
        variant.dispatch_short_conv(recorder, self.shape);
        variant.record_candidate_conv_state(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
        );
        // Chunkwise and replay requests have disjoint state slots and output rows.
        recorder.record_disjoint_buffers(
            &[self.buffers.recurrent_output, self.buffers.recurrent_state_arena],
            || {
                variant.record_chunkwise_state(
                    recorder,
                    self.shape,
                    &self.buffers,
                    self.num_active_chunkwise_requests,
                    true,
                );
                self.record_replay(recorder);
            },
        );
        variant.record_output_norm_gate(recorder, self.shape, &self.buffers, self.num_active_tokens);
    }
}

fn validate_log_range(buffer_bytes: u64, row_bytes: u64, token_offset: u64, num_tokens: u32) {
    // Row widths are positive and bounded by validated model geometry.
    // Compare row counts so neither an offset sum nor a byte product can wrap.
    let num_rows = buffer_bytes / row_bytes;
    assert!(
        token_offset <= num_rows,
        "GDN replay log offset exceeds buffer capacity"
    );
    assert!(
        num_tokens as u64 <= num_rows - token_offset,
        "GDN replay log buffer is too small"
    );
}

impl ReplayInvocation<'_> {
    fn record_replay(&self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.variant.replay);
        recorder.set_buffer_write(0, self.buffers.recurrent_output, 0);
        recorder.set_buffer_read(1, self.buffers.recurrent_state_arena, 0);
        recorder.set_buffer_read(2, self.buffers.conv_qkv, 0);
        recorder.set_buffer_read(3, self.buffers.a, 0);
        recorder.set_buffer_read(4, self.buffers.b, 0);
        recorder.set_buffer_read(5, self.buffers.a_log, 0);
        recorder.set_buffer_read(6, self.buffers.dt_bias, 0);
        recorder.set_buffer_read(7, self.buffers.src_recurrent_state_slots, 0);
        recorder.set_buffer_write(8, self.replay.u, 0);
        recorder.set_buffer_read(9, self.buffers.cu_tokens, 0);
        recorder.set_f32(10, self.variant.q_scale);
        set_replay_u32(
            recorder,
            11,
            self.num_active_reqs,
            self.shape.num_total_reqs,
            "GDN active request count",
        );
        recorder.set_u64(12, self.buffers.recurrent_state_arena_offset_bytes);
        set_chunkwise_count(
            recorder,
            13,
            self.num_active_chunkwise_requests,
            self.shape.num_total_reqs,
        );
        recorder.set_buffer_write(14, self.replay.k, 0);
        recorder.set_buffer_write(15, self.replay.alpha, 0);
        recorder.set_u64(16, self.replay.token_offset);
        let thread_block = self.variant.constants.kernels.candidate_recurrent_state.thread_block;
        recorder.dispatch_threadblocks(
            (
                (self.variant.constants.model.v_head_dim / thread_block.num_v_rows()) as usize,
                self.shape.num_total_reqs as usize * self.variant.constants.model.num_v_heads as usize,
                1,
            ),
            (
                thread_block.num_qk_dim_threads as usize,
                thread_block.num_simdgroups as usize,
                1,
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::validate_log_range;

    #[test]
    fn test_log_range() {
        validate_log_range(64, 4, 12, 4);
        for (bytes, row_bytes, offset, tokens) in [(64, 4, 13, 4), (64, 4, u64::MAX, 1), (64, 4, 1 << 62, 1)] {
            assert!(std::panic::catch_unwind(|| validate_log_range(bytes, row_bytes, offset, tokens)).is_err());
        }
    }
}
