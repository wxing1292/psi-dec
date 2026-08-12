use std::rc::Rc;

use inference_executor_core::def::ModelExecutorError;

use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for Qwen3xGQAState {
    type Files = GQAStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        self.request_page_table().write_full_state(writer, files)
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        let request_page_table = Rc::get_mut(
            self.request_page_table
                .as_mut()
                .expect("Qwen3.x GQA request page-table state must be loaded"),
        )
        .expect("Qwen3.x GQA request page table must be unattached during state loading");
        request_page_table.read_full_state(reader, files)
    }
}
