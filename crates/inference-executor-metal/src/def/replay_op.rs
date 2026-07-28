use inference_backend_metal::components::RMSNormInvocation;
use inference_backend_metal::components::ResidualCaptureTarget;
use inference_backend_metal::components::ResidualInvocation;
use inference_backend_metal::metal::Operator;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplaySubmission;
use inference_backend_metal::metal::Stream;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::backend::runtime::Runtime;
use inference_runtime_core::compute::ExecutionSubmission;

pub struct MetalReplayRuntime<'a> {
    stream: &'a Stream,
}

impl<'a> MetalReplayRuntime<'a> {
    pub fn new(stream: &'a Stream) -> Self {
        Self { stream }
    }

    pub fn create_recorder(&self) -> ReplayRecorder {
        ReplayRecorder::new(self.stream.create_replay_program())
    }

    pub fn submit_replay(&self, replay: &ReplayProgram) -> MetalReplaySubmission {
        MetalReplaySubmission::new(self.stream.submit_replay(replay))
    }

    pub fn submit_replay_with_arguments(
        &self,
        replay: &ReplayProgram,
        arguments: &ReplayArguments,
    ) -> MetalReplaySubmission {
        MetalReplaySubmission::new(self.stream.submit_replay_with_arguments(replay, arguments))
    }

    pub fn submit_replay_sequence(&self, executions: &[ReplayExecution<'_>]) -> MetalReplaySubmission {
        MetalReplaySubmission::new(self.stream.submit_replay_sequence(executions))
    }
}

impl Runtime for MetalReplayRuntime<'_> {
    type Replay = ReplayProgram;
    type Submission = MetalReplaySubmission;
    type Recorder<'a>
        = ReplayRecorder
    where
        Self: 'a;

    fn create_recorder(&self) -> Self::Recorder<'_> {
        MetalReplayRuntime::create_recorder(self)
    }

    fn submit_replay(&self, replay: &Self::Replay) -> Self::Submission {
        MetalReplayRuntime::submit_replay(self, replay)
    }
}

pub struct ReplayRecorder {
    inner: inference_backend_metal::components::ReplayRecorder,
}

impl ReplayRecorder {
    fn new(inner: inference_backend_metal::metal::ReplayProgramBuilder) -> Self {
        Self {
            inner: inference_backend_metal::components::ReplayRecorder::new(inner),
        }
    }

    pub fn build(self) -> ReplayProgram {
        self.inner.build()
    }
}

impl<'a> Recorder<'a> for ReplayRecorder {
    type Operator = ReplayOp<'a>;
    type Replay = ReplayProgram;

    fn record(&mut self, operator: Self::Operator) {
        self.inner.record(operator.into_inner());
    }

    fn record_with_barrier_before(&mut self, operator: Self::Operator) {
        self.inner.record_with_barrier_before(operator.into_inner());
    }

    fn build(self) -> Self::Replay {
        ReplayRecorder::build(self)
    }
}

pub struct MetalReplaySubmission {
    inner: ReplaySubmission,
}

impl MetalReplaySubmission {
    fn new(inner: ReplaySubmission) -> Self {
        Self { inner }
    }

    pub fn wait(&self) {
        self.inner.wait();
    }
}

impl ExecutionSubmission for MetalReplaySubmission {
    fn wait(&self) {
        MetalReplaySubmission::wait(self);
    }
}

pub struct ReplayOp<'a> {
    inner: inference_backend_metal::components::ReplayOp<'a>,
}

impl<'a> ReplayOp<'a> {
    pub fn opaque<I>(operator: I) -> Self
    where
        I: Operator + 'a,
    {
        Self {
            inner: inference_backend_metal::components::ReplayOp::opaque(operator),
        }
    }

    pub fn residual_add(invocation: ResidualInvocation<'a>) -> Self {
        Self {
            inner: inference_backend_metal::components::ReplayOp::residual_add(invocation),
        }
    }

    /// Records a BF16 residual add that captures every complete output row.
    ///
    /// The next recorded operator must be the RMSNorm consuming this residual
    /// output. Backend replay fusion then asserts that the destination range
    /// width equals the RMSNorm hidden dimension, has sufficient capacity and
    /// alignment, and does not alias any fused residual/RMSNorm buffer. Those
    /// checks are delayed until fusion because this wrapper does not own the
    /// RMSNorm shape or fused buffers. Unsupported vec4 destination alignment
    /// panics instead of falling back to a more generic kernel.
    pub fn residual_add_with_capture(invocation: ResidualInvocation<'a>, capture: ResidualCaptureTarget<'a>) -> Self {
        Self {
            inner: inference_backend_metal::components::ReplayOp::residual_add_with_capture(invocation, capture),
        }
    }

    pub fn rms_norm(invocation: RMSNormInvocation<'a>) -> Self {
        Self {
            inner: inference_backend_metal::components::ReplayOp::rms_norm(invocation),
        }
    }

    fn into_inner(self) -> inference_backend_metal::components::ReplayOp<'a> {
        self.inner
    }
}
