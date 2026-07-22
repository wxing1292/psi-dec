use crate::metal::Device;
use crate::metal::ReplayArguments;
use crate::metal::ReplayProgram;
use crate::metal::ReplayProgramBuilder;
use crate::metal::ReplaySubmission;
use crate::metal::Stream;

#[derive(Debug)]
pub struct MetalRuntime {
    device: Device,
    stream: Stream,
}

impl MetalRuntime {
    pub fn system_default() -> Self {
        Self::new(Device::system_default())
    }

    pub fn new(device: Device) -> Self {
        let stream = Stream::new(&device);
        Self { device, stream }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
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
