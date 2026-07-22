use crate::runtime::RawComputeSlotID;
use crate::runtime::RawComputeSlotSeq;

pub struct ComputeSlot {
    id: RawComputeSlotID,
    seq: Option<RawComputeSlotSeq>,
}

impl ComputeSlot {
    pub fn new(id: RawComputeSlotID) -> Self {
        Self { id, seq: None }
    }

    pub fn id(&self) -> RawComputeSlotID {
        self.id
    }

    pub fn seq(&self) -> Option<RawComputeSlotSeq> {
        self.seq
    }

    pub fn prepare(&mut self, seq: RawComputeSlotSeq) {
        debug_assert!(self.seq.is_none(), "compute slot must be free before prepare");
        self.seq = Some(seq);
    }

    pub fn reset(&mut self) {
        debug_assert!(self.seq.is_some(), "compute slot must be in use before reset");
        self.seq = None;
    }
}
