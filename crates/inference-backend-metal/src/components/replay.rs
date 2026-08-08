use crate::components::RMSNormInvocation;
use crate::components::ResidualAddInvocation;
use crate::components::residual_add::ResidualAddCaptureReplayOp;
use crate::components::residual_add::ResidualAddCaptureTarget;
use crate::components::residual_add::ResidualAddReplayOp;
use crate::components::residual_add_rms_norm::ResidualAddCaptureRMSNormReplayInvocation;
use crate::components::residual_add_rms_norm::ResidualAddRMSNormReplayInvocation;
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
            ReplayOpKind::ResidualAddWithCapture(op) => {
                self.push_pending(PendingReplayOp::ResidualAddWithCapture { op, barrier_before });
            },
            ReplayOpKind::RMSNorm(op) => {
                if let Some((previous, residual_barrier_before)) = self.pop_residual_add_with_capture(&op) {
                    self.push_pending(PendingReplayOp::ResidualAddCaptureRMSNorm {
                        op: ResidualAddCaptureRMSNormReplayInvocation::fuse_residual_add_capture_rms_norm(previous, op),
                        barrier_before: residual_barrier_before || barrier_before,
                    });
                    return;
                }
                if let Some((previous, residual_barrier_before)) = self.pop_residual_add(&op) {
                    self.push_pending(PendingReplayOp::ResidualAddRMSNorm {
                        op: ResidualAddRMSNormReplayInvocation::fuse_residual_add_rms_norm(previous, op),
                        barrier_before: residual_barrier_before || barrier_before,
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

    fn pop_residual_add(&mut self, rms_norm: &RMSNormReplayOp) -> Option<(ResidualAddReplayOp, bool)> {
        let Some(PendingReplayOp::ResidualAdd { op, .. }) = self.pending.last() else {
            return None;
        };
        if !ResidualAddRMSNormReplayInvocation::is_residual_add_rms_norm_fusion_compatible(op, rms_norm) {
            return None;
        }
        let Some(PendingReplayOp::ResidualAdd { op, barrier_before }) = self.pop() else {
            panic!("pending replay op changed after residual check");
        };
        Some((op, barrier_before))
    }

    fn pop_residual_add_with_capture(
        &mut self,
        rms_norm: &RMSNormReplayOp,
    ) -> Option<(ResidualAddCaptureReplayOp, bool)> {
        let Some(PendingReplayOp::ResidualAddWithCapture { op, .. }) = self.pending.last() else {
            return None;
        };
        if !ResidualAddCaptureRMSNormReplayInvocation::is_residual_add_capture_rms_norm_fusion_compatible(op, rms_norm)
        {
            return None;
        }
        let Some(PendingReplayOp::ResidualAddWithCapture { op, barrier_before }) = self.pop() else {
            panic!("pending replay op changed after residual-add capture check");
        };
        Some((op, barrier_before))
    }

    fn flush_pending(&mut self) {
        for operator in self.pending.drain(..) {
            match operator {
                PendingReplayOp::ResidualAdd { op, barrier_before } => {
                    record_pending(&mut self.inner, op.into_replay(), barrier_before);
                },
                PendingReplayOp::ResidualAddWithCapture { op, barrier_before } => {
                    record_pending(&mut self.inner, op.into_replay(), barrier_before);
                },
                PendingReplayOp::RMSNorm { op, barrier_before } => {
                    record_pending(&mut self.inner, op.into_replay(), barrier_before);
                },
                PendingReplayOp::ResidualAddRMSNorm { op, barrier_before } => {
                    record_pending(&mut self.inner, op, barrier_before);
                },
                PendingReplayOp::ResidualAddCaptureRMSNorm { op, barrier_before } => {
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
    ResidualAdd(ResidualAddReplayOp),
    ResidualAddWithCapture(ResidualAddCaptureReplayOp),
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

    pub fn residual_add(invocation: ResidualAddInvocation<'a>) -> Self {
        Self {
            kind: ReplayOpKind::ResidualAdd(invocation.into_replay_op()),
        }
    }

    /// Records a BF16 residual add that also captures every complete output row.
    ///
    /// An adjacent compatible RMSNorm is fused opportunistically. Otherwise,
    /// replay records the capture as an independent padding-safe operation.
    pub fn residual_add_with_capture(
        invocation: ResidualAddInvocation<'a>,
        capture: ResidualAddCaptureTarget<'a>,
    ) -> Self {
        Self {
            kind: ReplayOpKind::ResidualAddWithCapture(invocation.into_capture_replay_op(capture)),
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
        op: ResidualAddReplayOp,
        barrier_before: bool,
    },
    ResidualAddWithCapture {
        op: ResidualAddCaptureReplayOp,
        barrier_before: bool,
    },
    RMSNorm {
        op: RMSNormReplayOp,
        barrier_before: bool,
    },
    ResidualAddRMSNorm {
        op: ResidualAddRMSNormReplayInvocation,
        barrier_before: bool,
    },
    ResidualAddCaptureRMSNorm {
        op: ResidualAddCaptureRMSNormReplayInvocation,
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
    use crate::components::RMSNormBuffers;
    use crate::components::RMSNormConfig;
    use crate::components::RMSNormKernel;
    use crate::components::RMSNormShape;
    use crate::components::ResidualAddBuffers;
    use crate::components::ResidualAddCaptureTarget;
    use crate::components::ResidualAddConfig;
    use crate::components::ResidualAddKernel;
    use crate::components::ResidualAddRowShape;
    use crate::components::ResidualAddShape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.fused.num_active_tokens");
    const OTHER_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.fused.other_num_active_tokens");

    #[test]
    fn test_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let tokens = 2;
        let hidden_dim = 8;
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let rms_norm = RMSNormKernel::new(&device, RMSNormConfig::bf16(hidden_dim as u32, 1.0e-6));
        let num_values = tokens * hidden_dim;
        let bytes = num_values * size_of::<u16>();
        let lhs = Buffer::new_zeroed(&device, bytes);
        let rhs = Buffer::new_zeroed(&device, bytes);
        let residual_output = Buffer::new_zeroed(&device, bytes);
        let norm_output = Buffer::new_zeroed(&device, bytes);
        let weight = Buffer::new_zeroed(&device, hidden_dim * size_of::<u16>());
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());

        recorder.record(ReplayOp::residual_add(residual_add.invoke(
            ResidualAddShape {
                num_values: num_values as u32,
            },
            ResidualAddBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &residual_output,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::rms_norm(rms_norm.invoke(
            RMSNormShape {
                num_total_tokens: tokens as u32,
            },
            RMSNormBuffers {
                input: &residual_output,
                weight: &weight,
                output: &norm_output,
            },
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 1);
        let stats = replay.stats();
        assert_eq!(stats.retained_pipeline_count, 1);
        assert_eq!(stats.retained_buffer_count, 5);
    }

    #[test]
    fn test_unrelated_residual_add_and_rms_norm_are_not_fused() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let tokens = 2;
        let hidden_dim = 8;
        let num_values = tokens * hidden_dim;
        let bytes = num_values * size_of::<u16>();
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let rms_norm = RMSNormKernel::new(&device, RMSNormConfig::bf16(hidden_dim as u32, 1.0e-6));
        let lhs = Buffer::new_zeroed(&device, bytes);
        let rhs = Buffer::new_zeroed(&device, bytes);
        let residual_output = Buffer::new_zeroed(&device, bytes);
        let unrelated_norm_input = Buffer::new_zeroed(&device, bytes);
        let norm_output = Buffer::new_zeroed(&device, bytes);
        let weight = Buffer::new_zeroed(&device, hidden_dim * size_of::<u16>());
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());

        recorder.record(ReplayOp::residual_add(residual_add.invoke(
            ResidualAddShape {
                num_values: num_values as u32,
            },
            ResidualAddBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &residual_output,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::rms_norm(rms_norm.invoke(
            RMSNormShape {
                num_total_tokens: tokens as u32,
            },
            RMSNormBuffers {
                input: &unrelated_norm_input,
                weight: &weight,
                output: &norm_output,
            },
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 2);
    }

    #[test]
    fn test_bucketed_fusion_preserves_tails_across_grow_and_shrink() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let token_capacity = 2_u32;
        let hidden_dim = 8_u32;
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let rms_norm = RMSNormKernel::new(&device, RMSNormConfig::bf16(hidden_dim, 1.0e-6));
        let capacity_values = (token_capacity * hidden_dim) as usize;
        let row_values = hidden_dim as usize;
        let lhs = Buffer::from_slice(
            &device,
            &[
                vec![bf16::from_f32(1.0).to_bits(); row_values],
                vec![bf16::from_f32(4.0).to_bits(); row_values],
            ]
            .concat(),
        );
        let rhs = Buffer::from_slice(
            &device,
            &[
                vec![bf16::from_f32(2.0).to_bits(); row_values],
                vec![bf16::from_f32(5.0).to_bits(); row_values],
            ]
            .concat(),
        );
        let weight = Buffer::from_slice(&device, &vec![bf16::ONE.to_bits(); row_values]);
        let sentinel = bf16::from_f32(-321.0).to_bits();
        let residual_output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let norm_output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());

        recorder.record(ReplayOp::residual_add(residual_add.invoke_bucketed(
            ResidualAddRowShape {
                num_total_rows: token_capacity,
                num_columns: hidden_dim,
            },
            NUM_ACTIVE_TOKENS,
            ResidualAddBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &residual_output,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::rms_norm(rms_norm.invoke_bucketed(
            RMSNormShape {
                num_total_tokens: token_capacity,
            },
            NUM_ACTIVE_TOKENS,
            RMSNormBuffers {
                input: &residual_output,
                weight: &weight,
                output: &norm_output,
            },
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 1);
        assert_eq!(replay.stats().parameter_count, 1);

        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, 1))
            .wait();
        assert_eq!(
            residual_output.read_typed::<u16>(0, row_values),
            vec![bf16::from_f32(3.0).to_bits(); row_values]
        );
        assert_eq!(
            norm_output.read_typed::<u16>(0, row_values),
            vec![bf16::ONE.to_bits(); row_values]
        );
        assert_eq!(
            residual_output.read_typed::<u16>(row_values, row_values),
            vec![sentinel; row_values]
        );
        assert_eq!(
            norm_output.read_typed::<u16>(row_values, row_values),
            vec![sentinel; row_values]
        );

        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, 2))
            .wait();
        assert_eq!(
            residual_output.read_typed::<u16>(row_values, row_values),
            vec![bf16::from_f32(9.0).to_bits(); row_values]
        );
        assert_eq!(
            norm_output.read_typed::<u16>(row_values, row_values),
            vec![bf16::ONE.to_bits(); row_values]
        );

        lhs.write_typed(row_values, &vec![0x7fc1_u16; row_values]);
        rhs.write_typed(row_values, &vec![0x7fc1_u16; row_values]);
        residual_output.write_typed(row_values, &vec![sentinel; row_values]);
        norm_output.write_typed(row_values, &vec![sentinel; row_values]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, 1))
            .wait();
        assert_eq!(
            residual_output.read_typed::<u16>(row_values, row_values),
            vec![sentinel; row_values]
        );
        assert_eq!(
            norm_output.read_typed::<u16>(row_values, row_values),
            vec![sentinel; row_values]
        );
    }

    #[test]
    fn test_bucketed_residual_capture_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let token_capacity = 2_u32;
        let hidden_dim = 8_u32;
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let rms_norm = RMSNormKernel::new(&device, RMSNormConfig::bf16(hidden_dim, 1.0e-6));
        let capture_num_columns = hidden_dim * 3;
        let capture_column_start = hidden_dim;
        let capture_column_end = capture_column_start + hidden_dim;
        let capacity_values = (token_capacity * hidden_dim) as usize;
        let capture_capacity_values = (token_capacity * capture_num_columns) as usize;
        let input_poison = 0x7fc1_u16;
        let lhs = Buffer::from_slice(&device, &vec![input_poison; capacity_values]);
        let rhs = Buffer::from_slice(&device, &vec![input_poison; capacity_values]);
        let weight = Buffer::from_slice(&device, &vec![bf16::from_f32(1.0).to_bits(); hidden_dim as usize]);
        let sentinel = bf16::from_f32(-321.0).to_bits();
        let residual_output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let capture_output = Buffer::from_slice(&device, &vec![sentinel; capture_capacity_values]);
        let norm_output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());

        recorder.record(ReplayOp::residual_add_with_capture(
            residual_add.invoke_bucketed(
                ResidualAddRowShape {
                    num_total_rows: token_capacity,
                    num_columns: hidden_dim,
                },
                NUM_ACTIVE_TOKENS,
                ResidualAddBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    output: &residual_output,
                },
            ),
            ResidualAddCaptureTarget::columns(
                &capture_output,
                capture_num_columns,
                capture_column_start..capture_column_end,
            ),
        ));
        recorder.record_with_barrier_before(ReplayOp::rms_norm(rms_norm.invoke_bucketed(
            RMSNormShape {
                num_total_tokens: token_capacity,
            },
            NUM_ACTIVE_TOKENS,
            RMSNormBuffers {
                input: &residual_output,
                weight: &weight,
                output: &norm_output,
            },
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 1);
        assert_eq!(replay.stats().retained_buffer_count, 6);
        assert_eq!(replay.stats().parameter_count, 1);

        let run_round = |rows: &[(f32, f32)]| {
            let num_active_tokens = u32::try_from(rows.len()).unwrap();
            assert!(num_active_tokens > 0);
            assert!(num_active_tokens <= token_capacity);

            let mut lhs_values = vec![input_poison; capacity_values];
            let mut rhs_values = vec![input_poison; capacity_values];
            for (row, &(lhs_value, rhs_value)) in rows.iter().enumerate() {
                let row_start = row * hidden_dim as usize;
                lhs_values[row_start..row_start + hidden_dim as usize].fill(bf16::from_f32(lhs_value).to_bits());
                rhs_values[row_start..row_start + hidden_dim as usize].fill(bf16::from_f32(rhs_value).to_bits());
            }
            lhs.write_typed(0, &lhs_values);
            rhs.write_typed(0, &rhs_values);
            residual_output.write_typed(0, &vec![sentinel; capacity_values]);
            capture_output.write_typed(0, &vec![sentinel; capture_capacity_values]);
            norm_output.write_typed(0, &vec![sentinel; capacity_values]);

            stream
                .submit_replay_with_arguments(
                    &replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
                )
                .wait();

            let residual_values = residual_output.read_typed::<u16>(0, capacity_values);
            let capture_values = capture_output.read_typed::<u16>(0, capture_capacity_values);
            let norm_values = norm_output.read_typed::<u16>(0, capacity_values);
            for row in 0..token_capacity as usize {
                let expected_residual = rows
                    .get(row)
                    .map(|&(lhs_value, rhs_value)| bf16::from_f32(lhs_value + rhs_value).to_bits());
                for column in 0..hidden_dim as usize {
                    let index = row * hidden_dim as usize + column;
                    if let Some(expected_residual) = expected_residual {
                        assert_eq!(residual_values[index], expected_residual);
                        assert_eq!(norm_values[index], bf16::ONE.to_bits());
                    } else {
                        assert_eq!(residual_values[index], sentinel);
                        assert_eq!(norm_values[index], sentinel);
                    }
                }
                for column in 0..capture_num_columns as usize {
                    let actual = capture_values[row * capture_num_columns as usize + column];
                    let in_capture_columns =
                        column >= capture_column_start as usize && column < capture_column_end as usize;
                    if let (Some(expected_residual), true) = (expected_residual, in_capture_columns) {
                        assert_eq!(actual, expected_residual);
                    } else {
                        assert_eq!(actual, sentinel);
                    }
                }
            }
        };

        run_round(&[(1.0, 2.0)]);
        run_round(&[(4.0, 5.0), (6.0, 7.0)]);
        run_round(&[(8.0, 9.0)]);
    }

    #[test]
    #[should_panic(expected = "BF16 capture width must be divisible by four")]
    fn test_non_vector_width_capture_is_unsupported() {
        let device = Device::system_default();
        let capture_output = Buffer::new_zeroed(&device, 24 * size_of::<u16>());
        let _ = ResidualAddCaptureTarget::columns(&capture_output, 12, 0..6);
    }

    #[test]
    #[should_panic(expected = "unsupported residual-add capture layout")]
    fn test_unaligned_vec4_capture_is_unsupported() {
        let device = Device::system_default();
        let capture_output = Buffer::new_zeroed(&device, 26 * size_of::<u16>());
        let _ = ResidualAddCaptureTarget::columns(&capture_output, 13, 2..10);
    }

    #[test]
    fn test_residual_capture_records_without_rms_norm() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let lhs = Buffer::from_slice(&device, &vec![bf16::from_f32(1.0).to_bits(); 8]);
        let rhs = Buffer::from_slice(&device, &vec![bf16::from_f32(2.0).to_bits(); 8]);
        let output = Buffer::from_slice(&device, &vec![bf16::ZERO.to_bits(); 8]);
        let capture_output = Buffer::from_slice(&device, &vec![bf16::ZERO.to_bits(); 16]);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::residual_add_with_capture(
            residual_add.invoke_rows(
                ResidualAddRowShape {
                    num_total_rows: 1,
                    num_columns: 8,
                },
                ResidualAddBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    output: &output,
                },
            ),
            ResidualAddCaptureTarget::columns(&capture_output, 8, 0..8),
        ));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 1);
        stream.submit_replay(&replay).wait();
        let expected = vec![bf16::from_f32(3.0).to_bits(); 8];
        assert_eq!(output.read_typed::<u16>(0, 8), expected);
        assert_eq!(capture_output.read_typed::<u16>(0, 8), expected);
        assert_eq!(capture_output.read_typed::<u16>(8, 8), vec![bf16::ZERO.to_bits(); 8]);
    }

    #[test]
    #[should_panic(expected = "residual-add capture column count must match the residual source column count")]
    fn test_residual_capture_rejects_mismatched_source_column_count() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let lhs = Buffer::new_zeroed(&device, 16);
        let rhs = Buffer::new_zeroed(&device, 16);
        let output = Buffer::new_zeroed(&device, 16);
        let capture_output = Buffer::new_zeroed(&device, 16);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::residual_add_with_capture(
            residual_add.invoke_rows(
                ResidualAddRowShape {
                    num_total_rows: 1,
                    num_columns: 8,
                },
                ResidualAddBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    output: &output,
                },
            ),
            ResidualAddCaptureTarget::columns(&capture_output, 8, 0..4),
        ));

        recorder.build();
    }

    #[test]
    fn test_bucketed_residual_records_without_rms_norm() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let token_capacity = 2_u32;
        let hidden_dim = 8_u32;
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let row_values = hidden_dim as usize;
        let capacity_values = token_capacity as usize * row_values;
        let poison = 0x7fc1_u16;
        let sentinel = bf16::from_f32(-321.0).to_bits();
        let lhs = Buffer::from_slice(&device, &vec![poison; capacity_values]);
        let rhs = Buffer::from_slice(&device, &vec![poison; capacity_values]);
        let output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::residual_add(residual_add.invoke_bucketed(
            ResidualAddRowShape {
                num_total_rows: token_capacity,
                num_columns: hidden_dim,
            },
            NUM_ACTIVE_TOKENS,
            ResidualAddBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &output,
            },
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 1);
        assert_eq!(replay.stats().parameter_count, 1);
        for (num_active_tokens, lhs_value, rhs_value) in [(1, 1.0, 2.0), (2, 4.0, 5.0), (1, 6.0, 7.0)] {
            lhs.write_typed(0, &vec![poison; capacity_values]);
            rhs.write_typed(0, &vec![poison; capacity_values]);
            output.write_typed(0, &vec![sentinel; capacity_values]);
            let active_values = num_active_tokens as usize * row_values;
            lhs.write_typed(0, &vec![bf16::from_f32(lhs_value).to_bits(); active_values]);
            rhs.write_typed(0, &vec![bf16::from_f32(rhs_value).to_bits(); active_values]);
            stream
                .submit_replay_with_arguments(
                    &replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
                )
                .wait();
            assert_eq!(
                output.read_typed::<u16>(0, active_values),
                vec![bf16::from_f32(lhs_value + rhs_value).to_bits(); active_values]
            );
            if active_values < capacity_values {
                assert_eq!(
                    output.read_typed::<u16>(active_values, capacity_values - active_values),
                    vec![sentinel; capacity_values - active_values]
                );
            }
        }
    }

    #[test]
    fn test_bucketed_residual_capture_records_without_rms_norm() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let token_capacity = 2_u32;
        let hidden_dim = 8_u32;
        let capture_num_columns = 16_u32;
        let capture_column_start = 4_u32;
        let capacity_values = (token_capacity * hidden_dim) as usize;
        let capture_capacity_values = (token_capacity * capture_num_columns) as usize;
        let poison = 0x7fc1_u16;
        let sentinel = bf16::from_f32(-321.0).to_bits();
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let lhs = Buffer::from_slice(&device, &vec![poison; capacity_values]);
        let rhs = Buffer::from_slice(&device, &vec![poison; capacity_values]);
        let output = Buffer::from_slice(&device, &vec![sentinel; capacity_values]);
        let capture_output = Buffer::from_slice(&device, &vec![sentinel; capture_capacity_values]);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::residual_add_with_capture(
            residual_add.invoke_bucketed(
                ResidualAddRowShape {
                    num_total_rows: token_capacity,
                    num_columns: hidden_dim,
                },
                NUM_ACTIVE_TOKENS,
                ResidualAddBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    output: &output,
                },
            ),
            ResidualAddCaptureTarget::columns(
                &capture_output,
                capture_num_columns,
                capture_column_start..capture_column_start + hidden_dim,
            ),
        ));
        let replay = recorder.build();
        assert_eq!(replay.command_count(), 1);

        for (num_active_tokens, lhs_value, rhs_value) in [(1, 1.0, 2.0), (2, 4.0, 5.0), (1, 6.0, 7.0)] {
            lhs.write_typed(0, &vec![poison; capacity_values]);
            rhs.write_typed(0, &vec![poison; capacity_values]);
            output.write_typed(0, &vec![sentinel; capacity_values]);
            capture_output.write_typed(0, &vec![sentinel; capture_capacity_values]);
            let active_values = num_active_tokens as usize * hidden_dim as usize;
            let expected = bf16::from_f32(lhs_value + rhs_value).to_bits();
            lhs.write_typed(0, &vec![bf16::from_f32(lhs_value).to_bits(); active_values]);
            rhs.write_typed(0, &vec![bf16::from_f32(rhs_value).to_bits(); active_values]);
            stream
                .submit_replay_with_arguments(
                    &replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
                )
                .wait();

            let residual_values = output.read_typed::<u16>(0, capacity_values);
            let capture_values = capture_output.read_typed::<u16>(0, capture_capacity_values);
            for row in 0..token_capacity as usize {
                for column in 0..hidden_dim as usize {
                    let expected_value = if row < num_active_tokens as usize {
                        expected
                    } else {
                        sentinel
                    };
                    assert_eq!(residual_values[row * hidden_dim as usize + column], expected_value);
                }
                for column in 0..capture_num_columns as usize {
                    let is_captured = row < num_active_tokens as usize
                        && column >= capture_column_start as usize
                        && column < (capture_column_start + hidden_dim) as usize;
                    assert_eq!(
                        capture_values[row * capture_num_columns as usize + column],
                        if is_captured { expected } else { sentinel }
                    );
                }
            }
        }
    }

    #[test]
    fn test_mismatched_replay_parameter_disables_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let token_capacity = 2_u32;
        let hidden_dim = 8_u32;
        let capacity_values = (token_capacity * hidden_dim) as usize;
        let bytes = capacity_values * size_of::<u16>();
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let rms_norm = RMSNormKernel::new(&device, RMSNormConfig::bf16(hidden_dim, 1.0e-6));
        let lhs = Buffer::new_zeroed(&device, bytes);
        let rhs = Buffer::new_zeroed(&device, bytes);
        let residual_output = Buffer::new_zeroed(&device, bytes);
        let norm_output = Buffer::new_zeroed(&device, bytes);
        let weight = Buffer::new_zeroed(&device, hidden_dim as usize * size_of::<u16>());
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::residual_add(residual_add.invoke_bucketed(
            ResidualAddRowShape {
                num_total_rows: token_capacity,
                num_columns: hidden_dim,
            },
            NUM_ACTIVE_TOKENS,
            ResidualAddBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &residual_output,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::rms_norm(rms_norm.invoke_bucketed(
            RMSNormShape {
                num_total_tokens: token_capacity,
            },
            OTHER_NUM_ACTIVE_TOKENS,
            RMSNormBuffers {
                input: &residual_output,
                weight: &weight,
                output: &norm_output,
            },
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 2);
        assert_eq!(replay.stats().parameter_count, 2);
    }

    #[test]
    fn test_intervening_opaque_op_disables_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let hidden_dim = 8_u32;
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let rms_norm = RMSNormKernel::new(&device, RMSNormConfig::bf16(hidden_dim, 1.0e-6));
        let lhs = Buffer::new_zeroed(&device, 16);
        let rhs = Buffer::new_zeroed(&device, 16);
        let residual_output = Buffer::new_zeroed(&device, 16);
        let norm_output = Buffer::new_zeroed(&device, 16);
        let weight = Buffer::new_zeroed(&device, 16);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::residual_add(residual_add.invoke(
            ResidualAddShape { num_values: hidden_dim },
            ResidualAddBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &residual_output,
            },
        )));

        recorder.record(ReplayOp::opaque(rms_norm.invoke(
            RMSNormShape { num_total_tokens: 1 },
            RMSNormBuffers {
                input: &residual_output,
                weight: &weight,
                output: &norm_output,
            },
        )));
        assert_eq!(recorder.build().command_count(), 2);
    }

    #[test]
    fn test_unrelated_rms_norm_input_disables_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let hidden_dim = 8_u32;
        let residual_add = ResidualAddKernel::new(&device, ResidualAddConfig::bf16());
        let rms_norm = RMSNormKernel::new(&device, RMSNormConfig::bf16(hidden_dim, 1.0e-6));
        let lhs = Buffer::new_zeroed(&device, 16);
        let rhs = Buffer::new_zeroed(&device, 16);
        let residual_output = Buffer::new_zeroed(&device, 16);
        let unrelated_norm_input = Buffer::new_zeroed(&device, 16);
        let norm_output = Buffer::new_zeroed(&device, 16);
        let weight = Buffer::new_zeroed(&device, 16);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::residual_add(residual_add.invoke(
            ResidualAddShape { num_values: hidden_dim },
            ResidualAddBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &residual_output,
            },
        )));

        recorder.record(ReplayOp::rms_norm(rms_norm.invoke(
            RMSNormShape { num_total_tokens: 1 },
            RMSNormBuffers {
                input: &unrelated_norm_input,
                weight: &weight,
                output: &norm_output,
            },
        )));
        assert_eq!(recorder.build().command_count(), 2);
    }

    #[test]
    fn test_pending_non_residual() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let tokens = 2;
        let hidden_dim = 8;
        let rms_norm = RMSNormKernel::new(&device, RMSNormConfig::bf16(hidden_dim as u32, 1.0e-6));
        let bytes = tokens * hidden_dim * size_of::<u16>();
        let first_input = Buffer::new_zeroed(&device, bytes);
        let first_output = Buffer::new_zeroed(&device, bytes);
        let second_input = Buffer::new_zeroed(&device, bytes);
        let second_output = Buffer::new_zeroed(&device, bytes);
        let weight = Buffer::new_zeroed(&device, hidden_dim * size_of::<u16>());
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());

        recorder.record(ReplayOp::rms_norm(rms_norm.invoke(
            RMSNormShape {
                num_total_tokens: tokens as u32,
            },
            RMSNormBuffers {
                input: &first_input,
                weight: &weight,
                output: &first_output,
            },
        )));
        recorder.record(ReplayOp::rms_norm(rms_norm.invoke(
            RMSNormShape {
                num_total_tokens: tokens as u32,
            },
            RMSNormBuffers {
                input: &second_input,
                weight: &weight,
                output: &second_output,
            },
        )));

        let replay = recorder.build();
        assert_eq!(replay.command_count(), 2);
        assert_eq!(replay.stats().retained_pipeline_count, 1);
    }
}
