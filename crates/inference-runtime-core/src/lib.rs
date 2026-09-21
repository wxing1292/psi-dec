pub use inference_error::Error;
pub use inference_error::Result;
pub use inference_error::log_err_internal;
pub use inference_error::log_err_unavailable;
pub use inference_error::log_info_invalid_argument;
pub trait SSS: Send + Sync + 'static {}
impl<T> SSS for T where T: Send + Sync + 'static {}

pub mod config;
pub mod channel;
pub mod chat_template;
pub mod runtime;
pub mod compute;
pub mod memory;
pub mod network;
pub mod tokenizer;
