use std::cell::Cell;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::sync_channel;
use std::time::Duration;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTL4CommandAllocator;
use objc2_metal::MTL4CommandQueue;
use objc2_metal::MTL4CommitFeedback;
use objc2_metal::MTL4CommitOptions;
use objc2_metal::MTLDevice;

use crate::metal::Device;
use crate::metal::GpuAllocationSite;
use crate::metal::record_gpu_buffer_free;

mod operation;
pub use operation::CommandRecorder;
pub use operation::Operator;

mod dependency;

mod parameter;
pub use parameter::ReplayArguments;
pub use parameter::ReplayParameterKey;

mod residency;
use residency::ResidencySet;

mod replay;
pub use replay::ReplayProgram;
pub use replay::ReplayProgramBuilder;
pub use replay::ReplayProgramStats;

mod submission;
pub use submission::ReplayExecution;
pub use submission::ReplaySubmission;

const MAX_BUFFER_BINDINGS: usize = 31;
const PARAMETER_BUFFER_ALIGNMENT: usize = 8;

type CommitFeedbackBlock = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTL4CommitFeedback>>)>;

#[derive(Debug)]
struct CommitCompletion {
    options: Retained<MTL4CommitOptions>,
    handler: CommitFeedbackBlock,
    feedback: Receiver<Option<String>>,
}

impl CommitCompletion {
    fn new() -> Rc<Self> {
        let (feedback_tx, feedback) = sync_channel(1);
        let handler = RcBlock::new(move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
            let feedback = unsafe { feedback.as_ref() };
            let error = feedback.error().map(|error| format!("{error:?}"));
            let _ = feedback_tx.send(error);
        });
        Rc::new(Self {
            options: MTL4CommitOptions::new(),
            handler,
            feedback,
        })
    }

    fn wait(&self) {
        let error = self
            .feedback
            .recv_timeout(Duration::from_secs(60))
            .expect("timed out waiting for Metal commit feedback");
        if let Some(error) = error {
            panic!("Metal replay submission failed: {error}");
        }
    }
}

#[derive(Debug)]
pub struct Stream {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    allocator_in_flight: Rc<Cell<bool>>,
    completion: Rc<CommitCompletion>,
    residency_set: Rc<ResidencySet>,
}

impl Stream {
    pub fn new(device: &Device) -> Self {
        let queue = device
            .as_raw()
            .newMTL4CommandQueue()
            .expect("MTL4CommandQueue allocation failed");
        let allocator = device
            .as_raw()
            .newCommandAllocator()
            .expect("MTL4CommandAllocator allocation failed");
        let completion = CommitCompletion::new();
        let residency_set = ResidencySet::new(device.as_raw(), queue.clone());
        Self {
            device: device.as_raw_retained(),
            queue,
            allocator,
            allocator_in_flight: Rc::new(Cell::new(false)),
            completion,
            residency_set,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        "metal"
    }

    pub fn create_replay_program(&self) -> ReplayProgramBuilder {
        ReplayProgramBuilder::new(self)
    }

    pub fn submit_replay(&self, program: &ReplayProgram) -> ReplaySubmission {
        self.submit_replay_with_arguments(program, &ReplayArguments::new())
    }

    pub fn submit_replay_with_arguments(
        &self,
        program: &ReplayProgram,
        arguments: &ReplayArguments,
    ) -> ReplaySubmission {
        self.submit_replay_sequence(&[ReplayExecution::new(program, arguments)])
    }

    pub fn submit_replay_sequence(&self, executions: &[ReplayExecution<'_>]) -> ReplaySubmission {
        submission::submit_replay_sequence(self, executions)
    }
}

#[derive(Debug)]
pub struct TrackedGpuAllocation {
    site: GpuAllocationSite,
    len_bytes: usize,
}

impl TrackedGpuAllocation {
    pub fn new(site: GpuAllocationSite, len_bytes: usize) -> Self {
        Self { site, len_bytes }
    }
}

impl Drop for TrackedGpuAllocation {
    fn drop(&mut self) {
        record_gpu_buffer_free(self.site, self.len_bytes);
    }
}

#[cfg(test)]
mod tests {
    use crate::metal::Buffer;
    use crate::metal::CommandRecorder;
    use crate::metal::Device;
    use crate::metal::Kernel;
    use crate::metal::Operator;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayExecution;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const ADD_ONE_SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void add_one(
            device float* values [[buffer(0)]],
            constant uint& len [[buffer(1)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid < len) {
                values[gid] += 1.0f;
            }
        }
    "#;

    struct AddOneInvocation<'a> {
        kernel: &'a Kernel,
        values: &'a Buffer,
        len: u32,
    }

    struct AddOneReplayInvocation<'a> {
        kernel: &'a Kernel,
        values: &'a Buffer,
        num_active_threads_key: ReplayParameterKey,
        min_num_active_threads: u32,
        num_total_threads: u32,
        num_threads_per_threadblock: u32,
    }

    impl Operator for AddOneInvocation<'_> {
        fn record(self, builder: &CommandRecorder<'_>) {
            builder.set_kernel(self.kernel);
            builder.set_buffer_read_write(0, self.values, 0);
            builder.set_u32(1, self.len);
            builder.dispatch_1d(self.len as usize, 2);
        }
    }

    impl Operator for AddOneReplayInvocation<'_> {
        fn record(self, builder: &CommandRecorder<'_>) {
            builder.set_kernel(self.kernel);
            builder.set_buffer_read_write(0, self.values, 0);
            assert!(self.min_num_active_threads <= self.num_total_threads);
            builder.bind_u32(
                1,
                self.num_active_threads_key,
                self.min_num_active_threads,
                self.num_total_threads,
            );
            builder.dispatch_1d(
                self.num_total_threads as usize,
                self.num_threads_per_threadblock as usize,
            );
        }
    }

    #[test]
    fn test_submission_drop() {
        const NUM_ACTIVE_THREADS: ReplayParameterKey =
            ReplayParameterKey::new("test.drop_submission.num_active_threads");

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let values = Buffer::from_slice(&device, &[1.0_f32, 2.0, 3.0, 4.0]);

        let mut builder = stream.create_replay_program();
        builder.record(AddOneReplayInvocation {
            kernel: &kernel,
            values: &values,
            num_active_threads_key: NUM_ACTIVE_THREADS,
            min_num_active_threads: 1,
            num_total_threads: 4,
            num_threads_per_threadblock: 2,
        });
        let program = builder.build();
        let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_THREADS, 3);

        drop(stream.submit_replay_with_arguments(&program, &arguments));
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        assert_eq!(values.read_typed::<f32>(0, 4), vec![3.0, 4.0, 5.0, 4.0]);
    }

    #[test]
    fn test_consumer_barriers() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let replay_values = Buffer::from_slice(&device, &[1.0_f32, 2.0, 3.0]);

        let mut replay = stream.create_replay_program();
        replay.record_with_barrier_before(AddOneInvocation {
            kernel: &kernel,
            values: &replay_values,
            len: 3,
        });
        replay.record_with_barrier_before(AddOneInvocation {
            kernel: &kernel,
            values: &replay_values,
            len: 3,
        });
        stream.submit_replay(&replay.build()).wait();

        assert_eq!(replay_values.read_typed::<f32>(0, 3), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_submission_resources() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let values = Buffer::from_slice(&device, &[1.0_f32, 2.0, 3.0]);

        let mut builder = stream.create_replay_program();
        builder.record(AddOneInvocation {
            kernel: &kernel,
            values: &values,
            len: 3,
        });
        let program = builder.build();

        let submitted = stream.submit_replay(&program);
        drop(program);
        submitted.wait();

        assert_eq!(values.read_typed::<f32>(0, 3), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sequence_resources() {
        const ACTIVE_LEN: ReplayParameterKey = ReplayParameterKey::new("test.replay_sequence.active_len");

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let values = Buffer::from_slice(&device, &[1.0_f32, 2.0, 3.0]);

        let mut first_builder = stream.create_replay_program();
        first_builder.record(AddOneInvocation {
            kernel: &kernel,
            values: &values,
            len: 3,
        });
        let first = first_builder.build();

        let mut second_builder = stream.create_replay_program();
        second_builder.record(AddOneReplayInvocation {
            kernel: &kernel,
            values: &values,
            num_active_threads_key: ACTIVE_LEN,
            min_num_active_threads: 1,
            num_total_threads: 3,
            num_threads_per_threadblock: 2,
        });
        let second = second_builder.build();

        let first_arguments = ReplayArguments::new();
        let second_arguments = ReplayArguments::new().with_u32(ACTIVE_LEN, 2);
        let submitted = stream.submit_replay_sequence(&[
            ReplayExecution::new(&first, &first_arguments),
            ReplayExecution::new(&second, &second_arguments),
        ]);
        drop(first);
        drop(second);
        submitted.wait();

        assert_eq!(values.read_typed::<f32>(0, 3), vec![3.0, 4.0, 4.0]);
    }

    #[test]
    fn test_sequence_repeat() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let values = Buffer::from_slice(&device, &[1.0_f32]);
        let mut builder = stream.create_replay_program();
        builder.record(AddOneInvocation {
            kernel: &kernel,
            values: &values,
            len: 1,
        });
        let program = builder.build();
        let arguments = ReplayArguments::new();

        stream
            .submit_replay_sequence(&[
                ReplayExecution::new(&program, &arguments),
                ReplayExecution::new(&program, &arguments),
            ])
            .wait();

        assert_eq!(values.read_typed::<f32>(0, 1), vec![3.0]);
    }

    #[test]
    fn test_buffer_dependency() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let values = Buffer::from_slice(&device, &[1.0_f32, 2.0, 3.0]);

        let mut builder = stream.create_replay_program();
        builder.record(AddOneInvocation {
            kernel: &kernel,
            values: &values,
            len: 3,
        });
        builder.record(AddOneInvocation {
            kernel: &kernel,
            values: &values,
            len: 3,
        });
        let program = builder.build();
        assert_eq!(program.command_count(), 2);

        stream.submit_replay(&program).wait();

        assert_eq!(values.read_typed::<f32>(0, 3), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_submission_parameters() {
        const ACTIVE_LEN: ReplayParameterKey = ReplayParameterKey::new("test.add_one.active_len");

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let values = Buffer::from_slice(&device, &[0.0_f32; 4]);

        let mut builder = stream.create_replay_program();
        for _ in 0..2 {
            builder.record(AddOneReplayInvocation {
                kernel: &kernel,
                values: &values,
                num_active_threads_key: ACTIVE_LEN,
                min_num_active_threads: 1,
                num_total_threads: 4,
                num_threads_per_threadblock: 2,
            });
        }
        let program = builder.build();
        assert_eq!(program.stats().parameter_count, 1);

        let first = ReplayArguments::new().with_u32(ACTIVE_LEN, 2);
        stream.submit_replay_with_arguments(&program, &first).wait();
        let second = ReplayArguments::new().with_u32(ACTIVE_LEN, 4);
        stream.submit_replay_with_arguments(&program, &second).wait();

        assert_eq!(values.read_typed::<f32>(0, 4), vec![4.0, 4.0, 2.0, 2.0]);
    }

    #[test]
    fn test_parameter_bounds() {
        const ACTIVE_LEN: ReplayParameterKey = ReplayParameterKey::new("test.bounded_add_one.active_len");

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let values = Buffer::from_slice(&device, &[0.0_f32; 4]);

        let mut builder = stream.create_replay_program();
        builder.record(AddOneReplayInvocation {
            kernel: &kernel,
            values: &values,
            num_active_threads_key: ACTIVE_LEN,
            min_num_active_threads: 1,
            num_total_threads: 4,
            num_threads_per_threadblock: 2,
        });
        let program = builder.build();

        let min = ReplayArguments::new().with_u32(ACTIVE_LEN, 1);
        stream.submit_replay_with_arguments(&program, &min).wait();
        let max = ReplayArguments::new().with_u32(ACTIVE_LEN, 4);
        stream.submit_replay_with_arguments(&program, &max).wait();
        assert_eq!(values.read_typed::<f32>(0, 4), vec![2.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_residency_lifecycle() {
        const NUM_REPLAY_PROGRAMS: usize = 40;

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let values = Buffer::from_slice(&device, &[0.0_f32]);

        let programs = (0..NUM_REPLAY_PROGRAMS)
            .map(|_| {
                let mut builder = stream.create_replay_program();
                builder.record(AddOneInvocation {
                    kernel: &kernel,
                    values: &values,
                    len: 1,
                });
                builder.build()
            })
            .collect::<Vec<_>>();

        let stats = programs[0].stats();
        let shared_allocations = stats.retained_buffer_count + stats.retained_pipeline_count;
        let replay_local_allocations = 1 + usize::from(stats.parameter_buffer_bytes > 0);
        assert_eq!(
            stream.residency_set.allocation_count(),
            shared_allocations + NUM_REPLAY_PROGRAMS * replay_local_allocations
        );

        for program in &programs {
            stream.submit_replay(program).wait();
        }

        assert_eq!(values.read_typed::<f32>(0, 1), vec![NUM_REPLAY_PROGRAMS as f32]);
        drop(programs);
        assert_eq!(stream.residency_set.allocation_count(), 0);
    }
}
