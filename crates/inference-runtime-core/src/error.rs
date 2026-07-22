#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("corruption error: {0}")]
    Corruption(String),

    #[error("std i/o error: {0}")]
    STDIOError(#[from] std::io::Error),

    #[error("resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("internal error: {0}")]
    InternalError(String),
}

impl Error {
    pub fn corruption(msg: impl Into<String>) -> Self {
        Self::Corruption(msg.into())
    }

    pub fn from_io(err: std::io::Error) -> Self {
        Self::STDIOError(err)
    }

    pub fn resource_exhausted(msg: impl Into<String>) -> Self {
        Self::ResourceExhausted(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::InternalError(msg.into())
    }
}

#[macro_export]
macro_rules! log_err_internal {
    ($($tt:tt)*) => {{
        let msg = format!($($tt)*);
        tracing::error!(%msg);
        $crate::Error::InternalError(msg)
    }};
}
