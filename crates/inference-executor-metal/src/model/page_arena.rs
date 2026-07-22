use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;

pub struct PageArena {
    pages: Buffer,
}

impl PageArena {
    pub fn new(device: &Device, num_pages: usize, page_bytes: usize) -> Self {
        assert!(num_pages > 0, "page arena requires pages");
        assert!(page_bytes > 0, "page arena requires nonzero page size");
        let len_bytes = num_pages
            .checked_mul(page_bytes)
            .expect("page arena byte length must fit usize");
        Self {
            pages: Buffer::new_zeroed(device, len_bytes),
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.pages
    }
}
