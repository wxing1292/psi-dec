use std::cell::Cell;
use std::ptr::NonNull;
use std::rc::Rc;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTL4CommandAllocator;
use objc2_metal::MTL4CommandBuffer;
use objc2_metal::MTL4CommandEncoder;
use objc2_metal::MTL4CommandQueue;
use objc2_metal::MTL4VisibilityOptions;
use objc2_metal::MTLDevice;
use objc2_metal::MTLStages;

use crate::metal::stream::CommitCompletion;
use crate::metal::stream::ReplayArguments;
use crate::metal::stream::ReplayProgram;
use crate::metal::stream::Stream;
use crate::metal::stream::replay::ReplayResources;
use crate::metal::stream::replay::assert_replay_submission_queue;
use crate::metal::stream::replay::encode_replay;
use crate::metal::stream::replay::validate_replay_arguments;

#[derive(Clone, Copy, Debug)]
pub struct ReplayExecution<'a> {
    program: &'a ReplayProgram,
    arguments: &'a ReplayArguments,
}

impl<'a> ReplayExecution<'a> {
    pub fn new(program: &'a ReplayProgram, arguments: &'a ReplayArguments) -> Self {
        Self { program, arguments }
    }
}

#[derive(Debug)]
pub struct ReplaySubmission {
    allocator_in_flight: Rc<Cell<bool>>,
    completion: Rc<CommitCompletion>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    _resources: Vec<Rc<ReplayResources>>,
    _queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    _command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    waited: Cell<bool>,
}

impl ReplaySubmission {
    fn submit(
        stream: &Stream,
        command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        resources: Vec<Rc<ReplayResources>>,
    ) -> Self {
        command_buffer.endCommandBuffer();
        let mut command_buffer_ptr = NonNull::new(Retained::as_ptr(&command_buffer).cast_mut())
            .expect("MTL4CommandBuffer pointer must not be null");
        unsafe {
            // Metal consumes this registration with the commit. CommitCompletion retains the block and options.
            stream
                .completion
                .options
                .addFeedbackHandler(RcBlock::as_ptr(&stream.completion.handler));
            stream
                .queue
                .commit_count_options(NonNull::from(&mut command_buffer_ptr), 1, &stream.completion.options);
        }

        Self {
            allocator_in_flight: stream.allocator_in_flight.clone(),
            completion: stream.completion.clone(),
            allocator: stream.allocator.clone(),
            _resources: resources,
            _queue: stream.queue.clone(),
            _command_buffer: command_buffer,
            waited: Cell::new(false),
        }
    }

    pub fn wait(&self) {
        if self.waited.replace(true) {
            return;
        }
        self.completion.wait();
        self.allocator.reset();
        self.allocator_in_flight.set(false);
    }
}

impl Drop for ReplaySubmission {
    fn drop(&mut self) {
        self.wait();
    }
}

pub fn submit_replay_sequence(stream: &Stream, executions: &[ReplayExecution<'_>]) -> ReplaySubmission {
    assert!(
        !executions.is_empty(),
        "Metal replay sequence requires at least one execution"
    );
    for (index, execution) in executions.iter().enumerate() {
        assert_replay_submission_queue(execution.program, &stream.queue);
        validate_replay_arguments(execution.program, execution.arguments);
        for previous in &executions[..index] {
            if std::ptr::eq(previous.program, execution.program) {
                assert_eq!(
                    previous.arguments, execution.arguments,
                    "Metal replay sequence cannot execute the same program with conflicting arguments because replay \
                     arguments are stored in the program-owned parameter buffer"
                );
            }
        }
    }

    assert!(
        !stream.allocator_in_flight.replace(true),
        "Metal stream command allocator already has an in-flight submission; wait before submitting again"
    );
    let command_buffer = stream
        .device
        .newCommandBuffer()
        .expect("MTL4CommandBuffer allocation failed");
    command_buffer.beginCommandBufferWithAllocator(&stream.allocator);
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("MTL4ComputeCommandEncoder allocation failed");

    let mut resources = Vec::with_capacity(executions.len());
    for (index, execution) in executions.iter().enumerate() {
        if index > 0 {
            encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::None,
            );
        }
        resources.push(encode_replay(execution.program, &encoder, execution.arguments));
    }
    encoder.endEncoding();
    ReplaySubmission::submit(stream, command_buffer, resources)
}
