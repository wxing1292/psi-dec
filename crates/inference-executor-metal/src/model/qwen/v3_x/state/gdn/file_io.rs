use std::ops::Range;

use inference_executor_core::def::ModelExecutorError;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GDNStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for Qwen3xGDNState {
    type Files = GDNStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        self.request_state_table.write_full_state(writer, files)
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        self.request_state_table.read_full_state(reader, files)
    }
}

impl SelectedStateIO for Qwen3xGDNState {
    type ID = RawRequestSlot;

    fn write_selected_state(
        &self,
        writer: &mut StateSnapshotWriter,
        files: Self::Files,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        self.request_state_table
            .write_selected_state(writer, files, request_slot_ranges)
    }

    fn read_selected_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        self.request_state_table
            .read_selected_state(reader, files, request_slot_ranges)
    }
}
