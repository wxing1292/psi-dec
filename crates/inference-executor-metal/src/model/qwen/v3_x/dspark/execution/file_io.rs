use inference_executor_core::def::ModelExecutorError;

use crate::model::qwen::v3_x::dspark::execution::Qwen3xDSparkExecution;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for Qwen3xDSparkExecution {
    type Files = GQAStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        self.gqa_state.write_full_state(writer, files)
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        self.gqa_state.read_full_state(reader, files)
    }
}
