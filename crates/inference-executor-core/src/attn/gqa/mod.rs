mod core;
pub use core::GQACore;
pub use core::GQAPageTableLayout;
pub use core::GQAReplayShape;

mod ungated_core;
pub use ungated_core::UngatedGQACore;

pub mod reference;
