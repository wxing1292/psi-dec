use crate::def::ModelExecutorError;
use crate::model::ExecutionSubmission;

/// A replay-based encoder model without decoder request state.
pub trait ReplayableEncoderModel {
    type Input;
    type PreparedInput;
    type Output;
    type ModelOpsRecorder;
    type Submission: ExecutionSubmission;

    fn model_name(&self) -> &str;

    fn unload_weights(&mut self);
    fn load_weights(&mut self) -> Result<(), ModelExecutorError>;

    fn prepare(&mut self, input: Self::Input) -> Self::PreparedInput;
    fn record(&mut self, input: &Self::PreparedInput) -> Self::ModelOpsRecorder;
    fn submit(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission;
    fn complete(
        &mut self,
        input: Self::PreparedInput,
        recorder: Self::ModelOpsRecorder,
        submission: Self::Submission,
    ) -> Self::Output;

    fn execute(&mut self, input: Self::Input) -> Self::Output {
        let input = self.prepare(input);
        let recorder = self.record(&input);
        let submission = self.submit(&recorder);
        self.complete(input, recorder, submission)
    }
}

/// The service lifecycle of a standalone encoder executor.
pub trait EncoderExecutorLifecycle: Send + Sync + 'static {
    fn model_name(&self) -> &str;
    fn start(&self) -> Result<(), ModelExecutorError>;
    fn stop(&self);
}
