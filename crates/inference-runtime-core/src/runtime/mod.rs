pub mod pin_cache;
pub mod decoder;
pub mod scheduler;
pub mod tasks;

mod request;
pub use request::AtomicRequestStatus;
pub use request::CompletionReason;
pub use request::ExternalRequest;
pub use request::InternalRequest;
pub use request::QueuedRequest;
pub use request::RequestSlot;
pub use request::RequestSlotAllocationResult;
pub use request::RequestSlotAllocator;
pub use request::RequestStatus;
pub use request::TokenProbs;

mod token;
pub use token::Token;

mod resource;
pub use resource::ResourceID;
pub use resource::ResourcePlacement;
pub use resource::ResourceTypeID;
pub use resource::validate_resource_placements;

pub type RawRequestID = usize;
pub type RawRequestSlot = u32;
pub type RawPageID = u32;
pub type RawComputeSlotID = usize;
pub type RawComputeSlotSeq = u64;
