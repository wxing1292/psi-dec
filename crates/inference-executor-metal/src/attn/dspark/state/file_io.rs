use std::ops::Range;
use std::rc::Rc;

use inference_executor_core::def::ModelExecutorError;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::dspark::state::DSparkGQAState;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for DSparkGQAState {
    type Files = GQAStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        self.request_page_table_ref().write_full_state(writer, files)
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        let request_page_table = Rc::get_mut(
            self.request_page_table
                .as_mut()
                .expect("DSpark GQA request page-table state must be loaded"),
        )
        .expect("DSpark GQA request page table must be unattached during state loading");
        request_page_table.read_full_state(reader, files)
    }
}

impl SelectedStateIO for DSparkGQAState {
    type ID = RawRequestSlot;

    fn write_selected_state(
        &self,
        writer: &mut StateSnapshotWriter,
        files: Self::Files,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        self.request_page_table_ref()
            .write_selected_state(writer, files, request_slot_ranges)
    }

    fn read_selected_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        let request_page_table = Rc::get_mut(
            self.request_page_table
                .as_mut()
                .expect("DSpark GQA request page-table state must be loaded"),
        )
        .expect("DSpark GQA request page table must be unattached during state loading");
        request_page_table.read_selected_state(reader, files, request_slot_ranges)
    }
}
