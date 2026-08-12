use std::rc::Rc;

use inference_executor_core::def::ModelExecutorError;

use crate::attn::dspark::state::UngatedDSparkGQAState;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for UngatedDSparkGQAState {
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
