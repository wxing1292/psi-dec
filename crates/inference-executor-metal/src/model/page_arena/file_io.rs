use inference_executor_core::def::ModelExecutorError;

use crate::model::page_arena::PageArena;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::PageArenaStateSnapshotFiles;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for PageArena {
    type Files = PageArenaStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        writer.write_buffer(files.pages(), self.buffer())
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        reader.read_buffer(files.pages(), self.buffer())
    }
}
