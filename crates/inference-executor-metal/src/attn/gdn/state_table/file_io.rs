use std::cell::RefCell;

use inference_executor_core::def::ModelExecutorError;

use crate::attn::gdn::request_state_table::GDNRequestSlots;
use crate::attn::gdn::state_table::GDNRequestStateTable;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GDNStateSnapshotFiles;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for GDNRequestStateTable {
    type Files = GDNStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        self.assert_snapshot_ready();
        let request_table = self.request_table().borrow();
        request_table.assert_full_state_ready(
            self.max_publish_jobs_per_req,
            self.num_pages_per_state_slot(),
            self.num_cache_pages,
        );
        writer.write_metadata(files.request_state_table(), &*request_table)?;
        writer.write_buffer(files.recurrent_state(), &self.resources().recurrent_states)?;
        writer.write_buffer(files.conv_state(), &self.resources().conv_states)?;
        Ok(())
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        let request_table: GDNRequestSlots = reader.read_metadata(files.request_state_table())?;
        reader.read_buffer(files.recurrent_state(), &self.resources().recurrent_states)?;
        reader.read_buffer(files.conv_state(), &self.resources().conv_states)?;
        self.request_table = Some(RefCell::new(request_table));
        Ok(())
    }
}
