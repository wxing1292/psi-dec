use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCreateSystemDefaultDevice;
use objc2_metal::MTLDevice;

#[derive(Clone, Debug)]
pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl Device {
    pub fn system_default() -> Self {
        let raw = MTLCreateSystemDefaultDevice().expect("system default Metal device must exist");
        Self { raw }
    }

    pub fn from_raw_retained(raw: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self { raw }
    }

    pub fn as_raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }

    pub fn as_raw_retained(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        self.raw.clone()
    }

    pub fn name(&self) -> String {
        self.as_raw().name().to_string()
    }

    pub fn max_threadblock_memory_length(&self) -> usize {
        self.as_raw().maxThreadgroupMemoryLength()
    }

    pub fn max_buffer_length(&self) -> u64 {
        self.as_raw()
            .maxBufferLength()
            .try_into()
            .expect("MTLDevice maxBufferLength must fit u64")
    }
}
