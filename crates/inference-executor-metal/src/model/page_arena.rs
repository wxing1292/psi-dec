use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::def::ModelExecutorError;

use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

pub struct PageArena {
    pages: Option<Buffer>,
    num_pages: usize,
    page_bytes: usize,
}

impl PageArena {
    pub fn new(device: &Device, num_pages: usize, page_bytes: usize) -> Self {
        assert!(num_pages > 0, "page arena requires pages");
        assert!(page_bytes > 0, "page arena requires nonzero page size");
        let len_bytes = num_pages
            .checked_mul(page_bytes)
            .expect("page arena byte length must fit usize");
        Self {
            pages: Some(Buffer::new_zeroed(device, len_bytes)),
            num_pages,
            page_bytes,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        self.pages
            .as_ref()
            .expect("page arena state must be loaded before execution")
    }

    pub fn write_full_state(&self, writer: &mut StateSnapshotWriter, resource: u32) -> Result<(), ModelExecutorError> {
        writer.write_buffer(resource, self.buffer())
    }

    pub fn unload_state(&mut self) {
        assert!(self.pages.is_some(), "page arena state is not loaded");
        self.pages.take();
    }

    pub fn load_state(&mut self, device: &Device) {
        assert!(self.pages.is_none(), "page arena state is already loaded");
        self.pages = Some(Buffer::new_zeroed(
            device,
            self.num_pages
                .checked_mul(self.page_bytes)
                .expect("page arena byte length must fit usize"),
        ));
    }

    pub fn read_full_state(&self, reader: &mut StateSnapshotReader, resource: u32) -> Result<(), ModelExecutorError> {
        reader.read_buffer(resource, self.buffer())
    }
}
