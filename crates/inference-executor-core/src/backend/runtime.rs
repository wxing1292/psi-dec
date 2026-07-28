use inference_runtime_core::compute::ExecutionSubmission;

use crate::backend::recorder::Recorder;

/// Backend runtime boundary for replay creation and submission.
///
/// This is intentionally limited to replay orchestration. Tensor, weight, and
/// buffer abstractions remain backend-specific until a later design pass.
pub trait Runtime {
    type Replay;
    type Submission: ExecutionSubmission;
    type Recorder<'a>: Recorder<'a, Replay = Self::Replay>
    where
        Self: 'a;

    fn create_recorder(&self) -> Self::Recorder<'_>;
    fn submit_replay(&self, replay: &Self::Replay) -> Self::Submission;
}
