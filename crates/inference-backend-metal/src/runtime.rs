use crate::metal::BufferIO;
use crate::metal::Device;
use crate::metal::GpuTimestampGranularity;
use crate::metal::ReplayArguments;
use crate::metal::ReplayProgram;
use crate::metal::ReplayProgramBuilder;
use crate::metal::ReplaySubmission;
use crate::metal::Stream;

#[derive(Debug)]
pub struct MetalRuntime {
    device: Device,
    buffer_io: BufferIO,
    stream: Stream,
}

impl MetalRuntime {
    pub fn system_default() -> Self {
        Self::new(Device::system_default())
    }

    pub fn new(device: Device) -> Self {
        let buffer_io = BufferIO::new(&device);
        let stream = Stream::new_with_gpu_timestamps(&device, GpuTimestampGranularity::from_environment());
        Self {
            device,
            buffer_io,
            stream,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn buffer_io(&self) -> &BufferIO {
        &self.buffer_io
    }

    pub fn create_recorder(&self) -> ReplayProgramBuilder {
        self.stream.create_replay_program()
    }

    pub fn submit_replay(&self, replay: &ReplayProgram) -> ReplaySubmission {
        self.stream.submit_replay(replay)
    }

    pub fn submit_replay_with_arguments(
        &self,
        replay: &ReplayProgram,
        arguments: &ReplayArguments,
    ) -> ReplaySubmission {
        self.stream.submit_replay_with_arguments(replay, arguments)
    }
}
