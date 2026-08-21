//! Backend-neutral attention model metadata and shape contracts.

pub mod gdn;
pub use gdn::GDNCore;
pub use gdn::GDNReplayShape;

pub mod gqa;
pub use gqa::DSparkBlockCapacity;
pub use gqa::DSparkBlockMetadata;
pub use gqa::DSparkGQACore;
pub use gqa::GQACore;
pub use gqa::GQAPageTableLayout;
pub use gqa::GQAReplayShape;
pub use gqa::UngatedGQACore;
