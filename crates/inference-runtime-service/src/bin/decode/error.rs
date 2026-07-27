use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeCliError {
    message: String,
}

impl DecodeCliError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for DecodeCliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for DecodeCliError {}

impl From<String> for DecodeCliError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for DecodeCliError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

impl From<inference_runtime_core::Error> for DecodeCliError {
    fn from(error: inference_runtime_core::Error) -> Self {
        Self::new(error.to_string())
    }
}

pub type DecodeCliResult<T> = Result<T, DecodeCliError>;
