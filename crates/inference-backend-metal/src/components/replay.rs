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
#[path = "replay_test.rs"]
mod tests;
