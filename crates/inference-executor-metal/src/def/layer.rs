use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;

/// Replay-recording abstraction for semantic model layers/components.
///
/// A `ReplayLayer` owns model-level meaning: typed inputs, typed outputs, and
/// any request/state/page metadata at that semantic boundary. Its first
/// consumer command may carry the component-entry barrier; backend kernels,
/// resource usage markers, and internal phase barriers stay behind the
/// lower-level Metal operator invocations that the implementation records.
pub trait ReplayLayer {
    type Input<'a>
    where
        Self: 'a;
    type Output<'a>
    where
        Self: 'a;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>;
}
