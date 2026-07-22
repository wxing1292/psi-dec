use crate::memory::DeviceBlock;

#[derive(Debug, Eq, PartialEq)]
pub enum KVBlockPlacement {
    Device { block: DeviceBlock },
}

impl KVBlockPlacement {
    pub fn page_ids(&self) -> &[u32] {
        match self {
            Self::Device { block } => block.page_ids(),
        }
    }
}
