mod core;
pub use core::GQACore;
pub use core::GQAPageTableLayout;
pub use core::GQAReplayShape;

mod dspark_core;
pub use dspark_core::DSparkBlockCapacity;
pub use dspark_core::DSparkBlockMetadata;
pub use dspark_core::UngatedDSparkGQACore;

mod ungated_core;
pub use ungated_core::UngatedGQACore;

pub mod reference;
