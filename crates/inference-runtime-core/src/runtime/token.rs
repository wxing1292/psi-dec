use zerocopy::IntoBytes;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Token(pub u32);
impl Token {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u32 {
        self.0
    }
}

impl AsRef<[u8]> for Token {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl Default for Token {
    fn default() -> Self {
        Self(u32::MAX)
    }
}
