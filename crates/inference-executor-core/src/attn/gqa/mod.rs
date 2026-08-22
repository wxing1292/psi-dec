mod core;
pub use core::GQACore;
pub use core::GQAPageTableLayout;
pub use core::GQAReplayShape;

mod block_spec_core;
pub use block_spec_core::BlockSpecCapacity;
pub use block_spec_core::BlockSpecGQACore;
pub use block_spec_core::BlockSpecMetadata;

mod ungated_core;
pub use ungated_core::UngatedGQACore;

pub mod reference;
