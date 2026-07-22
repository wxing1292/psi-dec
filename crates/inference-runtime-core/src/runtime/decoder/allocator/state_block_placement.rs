use crate::memory::DeviceBlock;

#[derive(Debug, Eq, PartialEq)]
pub enum StateBlockPlacement {
    Device { block: DeviceBlock },
}

impl StateBlockPlacement {
    pub fn page_ids(&self) -> &[u32] {
        match self {
            Self::Device { block } => block.page_ids(),
        }
    }
}
