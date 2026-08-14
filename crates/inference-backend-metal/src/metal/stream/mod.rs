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
pub use parameter::ReplayF32;
pub use parameter::ReplayI32;
pub use parameter::ReplayI64;
pub use parameter::ReplayParameterKey;
pub use parameter::ReplayU32;
pub use parameter::ReplayU64;
pub use parameter::ReplayValue;

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
    use crate::metal::ReplayU32;
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

    const SCALAR_REPLAY_SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void write_scalars(
            device ulong* output_u64 [[buffer(0)]],
            device int* output_i32 [[buffer(1)]],
            device long* output_i64 [[buffer(2)]],
            device float* output_f32 [[buffer(3)]],
            constant ulong& value_u64 [[buffer(4)]],
            constant int& value_i32 [[buffer(5)]],
            constant long& value_i64 [[buffer(6)]],
            constant float& value_f32 [[buffer(7)]]
        ) {
            output_u64[0] = value_u64;
            output_i32[0] = value_i32;
            output_i64[0] = value_i64;
            output_f32[0] = value_f32;
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
        num_active_threads: ReplayU32,
        min_num_active_threads: u32,
        num_total_threads: u32,
        num_threads_per_threadblock: u32,
    }

    struct ScalarReplayInvocation<'a> {
        kernel: &'a Kernel,
        output_u64: &'a Buffer,
        output_i32: &'a Buffer,
        output_i64: &'a Buffer,
        output_f32: &'a Buffer,
        value_u64: ReplayParameterKey,
        value_i32: ReplayParameterKey,
        value_i64: ReplayParameterKey,
        value_f32: ReplayParameterKey,
    }

    impl Operator for AddOneInvocation<'_> {
        fn record(self, recorder: &CommandRecorder<'_>) {
            recorder.set_kernel(self.kernel);
            recorder.set_buffer_read_write(0, self.values, 0);
            recorder.set_u32(1, self.len);
            recorder.dispatch_1d(self.len as usize, 2);
        }
    }

    impl Operator for AddOneReplayInvocation<'_> {
        fn record(self, recorder: &CommandRecorder<'_>) {
            recorder.set_kernel(self.kernel);
            recorder.set_buffer_read_write(0, self.values, 0);
            assert!(self.min_num_active_threads <= self.num_total_threads);
            match self.num_active_threads {
                ReplayU32::Fixed(num_active_threads) => {
                    assert!(
                        num_active_threads >= self.min_num_active_threads
                            && num_active_threads <= self.num_total_threads
                    );
                    recorder.set_u32(1, num_active_threads);
                },
                ReplayU32::Parameter(key) => {
                    recorder.bind_u32(1, key, self.min_num_active_threads, self.num_total_threads);
                },
            }
            recorder.dispatch_1d(
                self.num_total_threads as usize,
                self.num_threads_per_threadblock as usize,
            );
        }
    }

    impl Operator for ScalarReplayInvocation<'_> {
        fn record(self, recorder: &CommandRecorder<'_>) {
            recorder.set_kernel(self.kernel);
            recorder.set_buffer_write(0, self.output_u64, 0);
            recorder.set_buffer_write(1, self.output_i32, 0);
            recorder.set_buffer_write(2, self.output_i64, 0);
            recorder.set_buffer_write(3, self.output_f32, 0);
            recorder.bind_u64(4, self.value_u64, 1, 10_000_000_000);
            recorder.bind_i32(5, self.value_i32, -10, 10);
            recorder.bind_i64(6, self.value_i64, -10_000_000_000, 10_000_000_000);
            recorder.bind_f32(7, self.value_f32, -2.0, 2.0);
            recorder.dispatch_1d(1, 1);
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
            num_active_threads: ReplayU32::Parameter(NUM_ACTIVE_THREADS),
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
            num_active_threads: ReplayU32::Parameter(ACTIVE_LEN),
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
                num_active_threads: ReplayU32::Parameter(ACTIVE_LEN),
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
            num_active_threads: ReplayU32::Parameter(ACTIVE_LEN),
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
    fn test_fixed_replay_u32() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
        let values = Buffer::from_slice(&device, &[0.0_f32; 4]);

        let mut builder = stream.create_replay_program();
        builder.record(AddOneReplayInvocation {
            kernel: &kernel,
            values: &values,
            num_active_threads: ReplayU32::Fixed(2),
            min_num_active_threads: 1,
            num_total_threads: 4,
            num_threads_per_threadblock: 2,
        });
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(values.read_typed::<f32>(0, 4), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_replay_scalar_parameter_types() {
        const VALUE_U64: ReplayParameterKey = ReplayParameterKey::new("test.scalar.value_u64");
        const VALUE_I32: ReplayParameterKey = ReplayParameterKey::new("test.scalar.value_i32");
        const VALUE_I64: ReplayParameterKey = ReplayParameterKey::new("test.scalar.value_i64");
        const VALUE_F32: ReplayParameterKey = ReplayParameterKey::new("test.scalar.value_f32");

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Kernel::new(&device, SCALAR_REPLAY_SOURCE, "write_scalars");
        let output_u64 = Buffer::from_slice(&device, &[0_u64]);
        let output_i32 = Buffer::from_slice(&device, &[0_i32]);
        let output_i64 = Buffer::from_slice(&device, &[0_i64]);
        let output_f32 = Buffer::from_slice(&device, &[0_f32]);

        let mut builder = stream.create_replay_program();
        builder.record(ScalarReplayInvocation {
            kernel: &kernel,
            output_u64: &output_u64,
            output_i32: &output_i32,
            output_i64: &output_i64,
            output_f32: &output_f32,
            value_u64: VALUE_U64,
            value_i32: VALUE_I32,
            value_i64: VALUE_I64,
            value_f32: VALUE_F32,
        });
        let program = builder.build();
        let arguments = ReplayArguments::new()
            .with_u64(VALUE_U64, 9_000_000_000)
            .with_i32(VALUE_I32, -7)
            .with_i64(VALUE_I64, -9_000_000_000)
            .with_f32(VALUE_F32, 1.25);

        stream.submit_replay_with_arguments(&program, &arguments).wait();

        assert_eq!(output_u64.read_typed::<u64>(0, 1), vec![9_000_000_000]);
        assert_eq!(output_i32.read_typed::<i32>(0, 1), vec![-7]);
        assert_eq!(output_i64.read_typed::<i64>(0, 1), vec![-9_000_000_000]);
        assert_eq!(output_f32.read_typed::<f32>(0, 1), vec![1.25]);
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
