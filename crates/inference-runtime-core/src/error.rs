#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    InvalidArgument(String),

    #[error("{0}")]
    ResourceExhausted(String),

    #[error("{0}")]
    Cancelled(String),

    #[error("{0}")]
    DeadlineExceeded(String),

    #[error("{0}")]
    Aborted(String),

    #[error("{0}")]
    Unavailable(String),

    #[error("{0}")]
    Internal(String),
}

impl Error {
    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    pub fn resource_exhausted(msg: impl Into<String>) -> Self {
        Self::ResourceExhausted(msg.into())
    }

    pub fn cancelled(msg: impl Into<String>) -> Self {
        Self::Cancelled(msg.into())
    }

    pub fn deadline_exceeded(msg: impl Into<String>) -> Self {
        Self::DeadlineExceeded(msg.into())
    }

    pub fn aborted(msg: impl Into<String>) -> Self {
        Self::Aborted(msg.into())
    }

    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self::Unavailable(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

#[macro_export]
macro_rules! log_err_internal {
    ($($tt:tt)*) => {{
        let msg = format!($($tt)*);
        tracing::error!(%msg);
        $crate::Error::Internal(msg)
    }};
}

#[macro_export]
macro_rules! log_err_unavailable {
    ($($tt:tt)*) => {{
        let msg = format!($($tt)*);
        tracing::error!(%msg);
        $crate::Error::Unavailable(msg)
    }};
}

#[macro_export]
macro_rules! log_info_invalid_argument {
    ($($tt:tt)*) => {{
        let msg = format!($($tt)*);
        tracing::info!(%msg);
        $crate::Error::InvalidArgument(msg)
    }};
}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn test_logging_macros_return_typed_errors() {
        assert!(matches!(
            crate::log_err_internal!("internal {}", 1),
            Error::Internal(message) if message == "internal 1"
        ));
        assert!(matches!(
            crate::log_err_unavailable!("unavailable {}", 2),
            Error::Unavailable(message) if message == "unavailable 2"
        ));
        assert!(matches!(
            crate::log_info_invalid_argument!("invalid {}", 3),
            Error::InvalidArgument(message) if message == "invalid 3"
        ));
    }
}
