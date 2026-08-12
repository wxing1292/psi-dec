use inference_executor_core::def::ModelExecutorError;

use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for GQARequestPageTable {
    type Files = GQAStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        writer.write_buffer(files.request_page_table(), self.page_ids_buffer())
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        reader.read_buffer(files.request_page_table(), self.page_ids_buffer())
    }
}
