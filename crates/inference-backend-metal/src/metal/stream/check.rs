use std::fmt::Debug;

/// A component-owned invariant checked at the submission boundary.
///
/// Replay retains the check and resets it before encoding. The component owns
/// its GPU record, bindings, and diagnostic format. The record must remain
/// sticky across repeated executions in one submission.
pub trait SubmissionCheck: Debug {
    fn reset(&self);

    /// Called only after successful GPU completion, before outputs are read.
    fn assert_success(&self);
}
