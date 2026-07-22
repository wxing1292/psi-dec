mod error;
pub use error::Error;

pub type Result<T> = std::result::Result<T, Error>;
pub trait SSS: Send + Sync + 'static {}
impl<T> SSS for T where T: Send + Sync + 'static {}

pub mod config;
pub mod channel;
pub mod runtime;
pub mod compute;
pub mod memory;
pub mod network;
pub mod tokenizer;
