use std::ops::Range;

use inference_executor_core::def::ModelExecutorError;
use inference_runtime_core::runtime::RawPageID;

use crate::model::page_arena::PageArena;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::PageArenaStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for PageArena {
    type Files = PageArenaStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        writer.write_full_buffer(files.pages(), self.buffer())
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        reader.read_full_buffer(files.pages(), self.buffer())
    }
}

impl SelectedStateIO for PageArena {
    type ID = RawPageID;

    fn write_selected_state(
        &self,
        writer: &mut StateSnapshotWriter,
        files: Self::Files,
        page_id_ranges: &[Range<RawPageID>],
    ) -> Result<(), ModelExecutorError> {
        writer.write_selected_buffer(files.pages(), self.buffer(), page_id_ranges, self.page_bytes)
    }

    fn read_selected_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
        page_id_ranges: &[Range<RawPageID>],
    ) -> Result<(), ModelExecutorError> {
        reader.read_selected_buffer(files.pages(), self.buffer(), page_id_ranges, self.page_bytes)
    }
}
