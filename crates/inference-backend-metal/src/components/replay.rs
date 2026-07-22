use crate::components::RMSNormInvocation;
use crate::components::ResidualInvocation;
use crate::components::residual::DuplicateResidualOutput;
use crate::components::residual::DuplicateResidualReplayOp;
use crate::components::residual::ResidualReplayOp;
use crate::components::residual_rms_norm::DuplicateResidualRMSNormReplayInvocation;
use crate::components::residual_rms_norm::ResidualRMSNormReplayInvocation;
use crate::components::rms_norm::RMSNormReplayOp;
use crate::metal::Operator;
use crate::metal::ReplayProgram;
use crate::metal::ReplayProgramBuilder;

pub struct ReplayRecorder {
    inner: ReplayProgramBuilder,
    pending: Vec<PendingReplayOp>,
}

impl ReplayRecorder {
    pub fn new(inner: ReplayProgramBuilder) -> Self {
        Self {
            inner,
            pending: Vec::with_capacity(4),
        }
    }

    pub fn record(&mut self, operator: ReplayOp<'_>) {
        self.record_inner(operator, false);
    }

    pub fn record_with_barrier_before(&mut self, operator: ReplayOp<'_>) {
        self.record_inner(operator, true);
    }

    fn record_inner(&mut self, operator: ReplayOp<'_>, barrier_before: bool) {
        match operator.kind {
            ReplayOpKind::Opaque(op) => {
                self.flush_pending();
                op.record_into(&mut self.inner, barrier_before);
            },
            ReplayOpKind::ResidualAdd(op) => self.push_pending(PendingReplayOp::ResidualAdd { op, barrier_before }),
            ReplayOpKind::ResidualAddWithDuplicate(op) => {
                self.push_pending(PendingReplayOp::ResidualAddWithDuplicate { op, barrier_before });
            },
            ReplayOpKind::RMSNorm(op) => {
                if let Some((previous, residual_barrier_before)) = self.pop_duplicate_residual_add() {
                    self.push_pending(PendingReplayOp::DuplicateResidualAddRMSNorm {
                        op: previous.fuse_rms_norm(op),
                        barrier_before: residual_barrier_before,
                    });
                    return;
                }
                if let Some((previous, residual_barrier_before)) = self.pop_residual_add() {
                    self.push_pending(PendingReplayOp::ResidualAddRMSNorm {
                        op: previous.fuse_rms_norm(op),
                        barrier_before: residual_barrier_before,
                    });
                    return;
                }
                self.push_pending(PendingReplayOp::RMSNorm { op, barrier_before });
            },
        }
    }

    pub fn build(mut self) -> ReplayProgram {
        self.flush_pending();
        self.inner.build()
    }

    fn push_pending(&mut self, operator: PendingReplayOp) {
        self.pending.push(operator);
    }

    fn pop(&mut self) -> Option<PendingReplayOp> {
        self.pending.pop()
    }

    fn pop_residual_add(&mut self) -> Option<(ResidualReplayOp, bool)> {
        if !matches!(self.pending.last(), Some(PendingReplayOp::ResidualAdd { .. })) {
            return None;
        }
        let Some(PendingReplayOp::ResidualAdd { op, barrier_before }) = self.pop() else {
            panic!("pending replay op changed after residual check");
        };
        Some((op, barrier_before))
    }

    fn pop_duplicate_residual_add(&mut self) -> Option<(DuplicateResidualReplayOp, bool)> {
        if !matches!(
            self.pending.last(),
            Some(PendingReplayOp::ResidualAddWithDuplicate { .. })
        ) {
            return None;
        }
        let Some(PendingReplayOp::ResidualAddWithDuplicate { op, barrier_before }) = self.pop() else {
            panic!("pending replay op changed after duplicate residual check");
        };
        Some((op, barrier_before))
    }

    fn flush_pending(&mut self) {
        for operator in self.pending.drain(..) {
            match operator {
                PendingReplayOp::ResidualAdd { op, barrier_before } => {
                    record_pending(&mut self.inner, op.into_replay(), barrier_before);
                },
                PendingReplayOp::ResidualAddWithDuplicate { .. } => {
                    panic!("duplicate residual add must be immediately fused with RMSNorm");
                },
                PendingReplayOp::RMSNorm { op, barrier_before } => {
                    record_pending(&mut self.inner, op.into_replay(), barrier_before);
                },
                PendingReplayOp::ResidualAddRMSNorm { op, barrier_before } => {
                    record_pending(&mut self.inner, op, barrier_before);
                },
                PendingReplayOp::DuplicateResidualAddRMSNorm { op, barrier_before } => {
                    record_pending(&mut self.inner, op, barrier_before);
                },
            }
        }
    }
}

fn record_pending<I: Operator>(builder: &mut ReplayProgramBuilder, operator: I, barrier_before: bool) {
    if barrier_before {
        builder.record_with_barrier_before(operator);
    } else {
        builder.record(operator);
    }
}

pub struct ReplayOp<'a> {
    kind: ReplayOpKind<'a>,
}

enum ReplayOpKind<'a> {
    Opaque(OpaqueReplayOp<'a>),
    ResidualAdd(ResidualReplayOp),
    ResidualAddWithDuplicate(DuplicateResidualReplayOp),
    RMSNorm(RMSNormReplayOp),
}

impl<'a> ReplayOp<'a> {
    pub fn opaque<I>(operator: I) -> Self
    where
        I: Operator + 'a,
    {
        Self {
            kind: ReplayOpKind::Opaque(OpaqueReplayOp::new(operator)),
        }
    }

    pub fn residual_add(invocation: ResidualInvocation<'a>) -> Self {
        Self {
            kind: ReplayOpKind::ResidualAdd(invocation.into_replay_op()),
        }
    }

    pub fn residual_add_with_duplicate_output(
        invocation: ResidualInvocation<'a>,
        duplicate_output: DuplicateResidualOutput<'a>,
    ) -> Self {
        Self {
            kind: ReplayOpKind::ResidualAddWithDuplicate(invocation.into_duplicate_replay_op(duplicate_output)),
        }
    }

    pub fn rms_norm(invocation: RMSNormInvocation<'a>) -> Self {
        Self {
            kind: ReplayOpKind::RMSNorm(invocation.into_replay_op()),
        }
    }
}

enum PendingReplayOp {
    ResidualAdd {
        op: ResidualReplayOp,
        barrier_before: bool,
    },
    ResidualAddWithDuplicate {
        op: DuplicateResidualReplayOp,
        barrier_before: bool,
    },
    RMSNorm {
        op: RMSNormReplayOp,
        barrier_before: bool,
    },
    ResidualAddRMSNorm {
        op: ResidualRMSNormReplayInvocation,
        barrier_before: bool,
    },
    DuplicateResidualAddRMSNorm {
        op: DuplicateResidualRMSNormReplayInvocation,
        barrier_before: bool,
    },
}

struct OpaqueReplayOp<'a> {
    record: Box<OpaqueReplayRecord<'a>>,
}

type OpaqueReplayRecord<'a> = dyn FnOnce(&mut ReplayProgramBuilder, bool) + 'a;

impl<'a> OpaqueReplayOp<'a> {
    fn new<I>(operator: I) -> Self
    where
        I: Operator + 'a,
    {
        Self {
            record: Box::new(move |builder, barrier_before| {
                if barrier_before {
                    builder.record_with_barrier_before(operator);
                } else {
                    builder.record(operator);
                }
            }),
        }
    }

    fn record_into(self, builder: &mut ReplayProgramBuilder, barrier_before: bool) {
        (self.record)(builder, barrier_before);
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use half::bf16;

    use super::ReplayOp;
    use super::ReplayRecorder;
    use crate::components::DuplicateResidualOutput;
    use crate::components::RMSNormBuffers;
    use crate::components::RMSNormKernel;
    use crate::components::RMSNormShape;
    use crate::components::ResidualBuffers;
    use crate::components::ResidualKernel;
    use crate::components::ResidualShape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.fused.num_active_tokens");

    #[test]
    fn test_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let residual = ResidualKernel::new(&device);
        let rms_norm = RMSNormKernel::new(&device);
        let tokens = 2;
        let hidden_dim = 8;
        let num_values = tokens * hidden_dim;
        let bytes = num_values * size_of::<u16>();
        let lhs = Buffer::new_zeroed(&device, bytes);
        let rhs = Buffer::new_zeroed(&device, bytes);
        let residual_output = Buffer::new_zeroed(&device, bytes);
        let norm_output = Buffer::new_zeroed(&device, bytes);
        let weight = Buffer::new_zeroed(&device, hidden_dim * size_of::<u16>());
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());

        recorder.record(ReplayOp::residual_add(residual.invoke(
            ResidualShape::bf16(num_values as u32),
            ResidualBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &residual_output,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::rms_norm(rms_norm.invoke(
            RMSNormShape::bf16(tokens as u32, hidden_dim as u32),
            RMSNormBuffers {
                input: &residual_output,
                weight: &weight,
                output: &norm_output,
            },
            1.0e-6,
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 1);
        let stats = replay.stats();
        assert_eq!(stats.retained_pipeline_count, 1);
        assert_eq!(stats.retained_buffer_count, 5);
    }

    #[test]
    fn test_bucketed_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let residual = ResidualKernel::new(&device);
        let rms_norm = RMSNormKernel::new(&device);
        let num_active_tokens = 2_u32;
        let token_capacity = 4_u32;
        let hidden_dim = 8_u32;
        let capacity_values = (token_capacity * hidden_dim) as usize;
        let active_values = (num_active_tokens * hidden_dim) as usize;
        let lhs = Buffer::from_slice(&device, &vec![1.0_f32; capacity_values]);
        let rhs = Buffer::from_slice(&device, &vec![2.0_f32; capacity_values]);
        let weight = Buffer::from_slice(&device, &vec![1.0_f32; hidden_dim as usize]);
        let sentinel = -321.0_f32;
        let residual_output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let norm_output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());

        recorder.record(ReplayOp::residual_add(residual.invoke(
            ResidualShape::f32(capacity_values as u32),
            ResidualBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &residual_output,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::rms_norm(rms_norm.invoke_bucketed(
            RMSNormShape::f32(token_capacity, hidden_dim),
            NUM_ACTIVE_TOKENS,
            RMSNormBuffers {
                input: &residual_output,
                weight: &weight,
                output: &norm_output,
            },
            1.0e-6,
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 1);
        assert_eq!(replay.stats().parameter_count, 1);
        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
            )
            .wait();

        assert_eq!(
            residual_output.read_typed::<f32>(0, active_values),
            vec![3.0; active_values]
        );
        assert_eq!(
            residual_output.read_typed::<f32>(active_values, capacity_values - active_values),
            vec![sentinel; capacity_values - active_values]
        );
        assert_eq!(
            norm_output.read_typed::<f32>(active_values, capacity_values - active_values),
            vec![sentinel; capacity_values - active_values]
        );
    }

    #[test]
    fn test_bucketed_duplicate_residual_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let residual = ResidualKernel::new(&device);
        let rms_norm = RMSNormKernel::new(&device);
        let num_active_tokens = 2_u32;
        let token_capacity = 4_u32;
        let hidden_dim = 8_u32;
        let duplicate_row_stride = hidden_dim * 3;
        let duplicate_column_offset = hidden_dim;
        let capacity_values = (token_capacity * hidden_dim) as usize;
        let duplicate_values = (token_capacity * duplicate_row_stride) as usize;
        let lhs_values = (0..capacity_values)
            .map(|index| bf16::from_f32(index as f32 * 0.03125 - 0.5).to_bits())
            .collect::<Vec<_>>();
        let rhs_values = (0..capacity_values)
            .map(|index| bf16::from_f32(index as f32 * -0.015625 + 0.25).to_bits())
            .collect::<Vec<_>>();
        let lhs = Buffer::from_slice(&device, &lhs_values);
        let rhs = Buffer::from_slice(&device, &rhs_values);
        let weight = Buffer::from_slice(&device, &vec![bf16::from_f32(1.0).to_bits(); hidden_dim as usize]);
        let sentinel = bf16::from_f32(-321.0).to_bits();
        let residual_output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let duplicate_output = Buffer::from_slice(&device, &vec![sentinel; duplicate_values]);
        let norm_output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());

        recorder.record(ReplayOp::residual_add_with_duplicate_output(
            residual.invoke(
                ResidualShape::bf16(capacity_values as u32),
                ResidualBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    output: &residual_output,
                },
            ),
            DuplicateResidualOutput {
                buffer: &duplicate_output,
                row_stride: duplicate_row_stride,
                column_offset: duplicate_column_offset,
            },
        ));
        recorder.record_with_barrier_before(ReplayOp::rms_norm(rms_norm.invoke_bucketed(
            RMSNormShape::bf16(token_capacity, hidden_dim),
            NUM_ACTIVE_TOKENS,
            RMSNormBuffers {
                input: &residual_output,
                weight: &weight,
                output: &norm_output,
            },
            1.0e-6,
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 1);
        assert_eq!(replay.stats().retained_buffer_count, 6);
        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
            )
            .wait();

        let residual_values = residual_output.read_typed::<u16>(0, capacity_values);
        let duplicate_values = duplicate_output.read_typed::<u16>(0, duplicate_values);
        for row in 0..token_capacity as usize {
            for column in 0..duplicate_row_stride as usize {
                let actual = duplicate_values[row * duplicate_row_stride as usize + column];
                let in_duplicate_slice = column >= duplicate_column_offset as usize
                    && column < (duplicate_column_offset + hidden_dim) as usize;
                if row < num_active_tokens as usize && in_duplicate_slice {
                    assert_eq!(
                        actual,
                        residual_values[row * hidden_dim as usize + column - duplicate_column_offset as usize]
                    );
                } else {
                    assert_eq!(actual, sentinel);
                }
            }
        }
        assert!(
            residual_values[(num_active_tokens * hidden_dim) as usize..]
                .iter()
                .all(|&value| value == sentinel)
        );
    }

    #[test]
    fn test_scalar_duplicate_residual_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let residual = ResidualKernel::new(&device);
        let rms_norm = RMSNormKernel::new(&device);
        let tokens = 2_u32;
        let hidden_dim = 6_u32;
        let row_stride = 13_u32;
        let column_offset = 2_u32;
        let num_values = (tokens * hidden_dim) as usize;
        let lhs = Buffer::from_slice(
            &device,
            &(0..num_values)
                .map(|index| bf16::from_f32(index as f32 * 0.125 - 0.75).to_bits())
                .collect::<Vec<_>>(),
        );
        let rhs = Buffer::from_slice(
            &device,
            &(0..num_values)
                .map(|index| bf16::from_f32(index as f32 * -0.0625 + 0.5).to_bits())
                .collect::<Vec<_>>(),
        );
        let weight = Buffer::from_slice(&device, &vec![bf16::from_f32(1.0).to_bits(); hidden_dim as usize]);
        let residual_output = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let norm_output = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let sentinel = bf16::from_f32(-123.0).to_bits();
        let duplicate_output = Buffer::from_slice(&device, &vec![sentinel; (tokens * row_stride) as usize]);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::residual_add_with_duplicate_output(
            residual.invoke(
                ResidualShape::bf16(num_values as u32),
                ResidualBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    output: &residual_output,
                },
            ),
            DuplicateResidualOutput {
                buffer: &duplicate_output,
                row_stride,
                column_offset,
            },
        ));
        recorder.record(ReplayOp::rms_norm(rms_norm.invoke(
            RMSNormShape::bf16(tokens, hidden_dim),
            RMSNormBuffers {
                input: &residual_output,
                weight: &weight,
                output: &norm_output,
            },
            1.0e-6,
        )));

        stream.submit_replay(&recorder.build()).wait();

        let residual_values = residual_output.read_typed::<u16>(0, num_values);
        let duplicate_values = duplicate_output.read_typed::<u16>(0, (tokens * row_stride) as usize);
        for row in 0..tokens as usize {
            assert_eq!(
                &duplicate_values[row * row_stride as usize + column_offset as usize
                    ..row * row_stride as usize + column_offset as usize + hidden_dim as usize],
                &residual_values[row * hidden_dim as usize..(row + 1) * hidden_dim as usize]
            );
        }
    }

    #[test]
    #[should_panic(expected = "duplicate residual add must be immediately fused with RMSNorm")]
    fn test_duplicate_residual_requires_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let residual = ResidualKernel::new(&device);
        let lhs = Buffer::new_zeroed(&device, 16);
        let rhs = Buffer::new_zeroed(&device, 16);
        let output = Buffer::new_zeroed(&device, 16);
        let duplicate_output = Buffer::new_zeroed(&device, 32);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::residual_add_with_duplicate_output(
            residual.invoke(
                ResidualShape::bf16(8),
                ResidualBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    output: &output,
                },
            ),
            DuplicateResidualOutput {
                buffer: &duplicate_output,
                row_stride: 8,
                column_offset: 0,
            },
        ));

        recorder.build();
    }

    #[test]
    fn test_pending_non_residual() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let rms_norm = RMSNormKernel::new(&device);
        let tokens = 2;
        let hidden_dim = 8;
        let bytes = tokens * hidden_dim * size_of::<u16>();
        let first_input = Buffer::new_zeroed(&device, bytes);
        let first_output = Buffer::new_zeroed(&device, bytes);
        let second_input = Buffer::new_zeroed(&device, bytes);
        let second_output = Buffer::new_zeroed(&device, bytes);
        let weight = Buffer::new_zeroed(&device, hidden_dim * size_of::<u16>());
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());

        recorder.record(ReplayOp::rms_norm(rms_norm.invoke(
            RMSNormShape::bf16(tokens as u32, hidden_dim as u32),
            RMSNormBuffers {
                input: &first_input,
                weight: &weight,
                output: &first_output,
            },
            1.0e-6,
        )));
        recorder.record(ReplayOp::rms_norm(rms_norm.invoke(
            RMSNormShape::bf16(tokens as u32, hidden_dim as u32),
            RMSNormBuffers {
                input: &second_input,
                weight: &weight,
                output: &second_output,
            },
            1.0e-6,
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 2);
        assert_eq!(replay.stats().retained_pipeline_count, 1);
    }
}
