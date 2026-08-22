//! Backend-neutral attention model metadata and shape contracts.

pub mod gdn;
pub use gdn::GDNCore;
pub use gdn::GDNReplayShape;

pub mod gqa;
pub use gqa::BlockSpecCapacity;
pub use gqa::BlockSpecGQACore;
pub use gqa::BlockSpecMetadata;
pub use gqa::GQACore;
pub use gqa::GQAPageTableLayout;
pub use gqa::GQAReplayShape;
pub use gqa::UngatedGQACore;
