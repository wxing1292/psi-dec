//! Low-level Metal wrapper surface.
//!
//! This module owns raw mutable Metal storage and command-dispatch concepts.

pub mod buffer;
pub mod device;
pub mod dtype;
pub mod kernel;
pub mod stream;

pub use buffer::Buffer;
pub use buffer::BufferAllocationSummary;
pub use buffer::BufferView;
pub use buffer::GpuAllocationSite;
pub use buffer::buffer_allocation_summary;
pub use buffer::record_gpu_buffer_alloc;
pub use buffer::record_gpu_buffer_free;
pub use device::Device;
pub use dtype::Dtype;
pub use dtype::MetalBufferElement;
pub use kernel::Kernel;
pub use stream::CommandRecorder;
pub use stream::Operator;
pub use stream::ReplayArguments;
pub use stream::ReplayExecution;
pub use stream::ReplayParameterKey;
pub use stream::ReplayProgram;
pub use stream::ReplayProgramBuilder;
pub use stream::ReplaySubmission;
pub use stream::Stream;
