use std::mem::size_of;
use std::ops::Range;

use inference_executor_core::def::ModelExecutorError;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for GQARequestPageTable {
    type Files = GQAStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        writer.write_full_buffer(files.request_page_table(), self.page_ids_buffer())
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        reader.read_full_buffer(files.request_page_table(), self.page_ids_buffer())
    }
}

impl SelectedStateIO for GQARequestPageTable {
    type ID = RawRequestSlot;

    fn write_selected_state(
        &self,
        writer: &mut StateSnapshotWriter,
        files: Self::Files,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        writer.write_selected_buffer(
            files.request_page_table(),
            self.page_ids_buffer(),
            request_slot_ranges,
            self.page_ids_per_request() * size_of::<u32>(),
        )
    }

    fn read_selected_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        reader.read_selected_buffer(
            files.request_page_table(),
            self.page_ids_buffer(),
            request_slot_ranges,
            self.page_ids_per_request() * size_of::<u32>(),
        )
    }
}
