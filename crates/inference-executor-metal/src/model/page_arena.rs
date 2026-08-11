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

    pub fn release_resources(&mut self) {
        assert!(self.pages.is_some(), "page arena state is not loaded");
        self.pages.take();
    }

    pub fn allocate_resources(&mut self, device: &Device) {
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use inference_backend_metal::metal::Device;

    use super::PageArena;
    use crate::model::state_snapshot::ModelFingerprint;
    use crate::model::state_snapshot::StateSnapshotReader;
    use crate::model::state_snapshot::StateSnapshotWriter;

    const NUM_PAGES: usize = 3;
    const PAGE_BYTES: usize = 16;
    const STATE_RESOURCE: u32 = 1;

    #[test]
    fn test_unload_load_fixed_state() {
        let expected = (0..NUM_PAGES * PAGE_BYTES)
            .map(|index| u8::try_from(index * 5 + 3).unwrap())
            .collect();
        assert_unload_load("fixed", expected);
    }

    #[test]
    fn test_unload_load_random_state() {
        let mut random = TestRandom::new(0x5041_4745_4152_454e);
        let expected = (0..NUM_PAGES * PAGE_BYTES).map(|_| random.next_u32() as u8).collect();
        assert_unload_load("random", expected);
    }

    fn assert_unload_load(name: &str, expected: Vec<u8>) {
        let device = Device::system_default();
        let mut arena = PageArena::new(&device, NUM_PAGES, PAGE_BYTES);
        arena.buffer().write_bytes(0, &expected);

        let snapshot_path = snapshot_path(name);
        let fingerprint = ModelFingerprint::new([0x50; 16]);
        let mut writer = StateSnapshotWriter::new(&snapshot_path, fingerprint).unwrap();
        arena.write_full_state(&mut writer, STATE_RESOURCE).unwrap();
        writer.commit().unwrap();

        arena.release_resources();
        arena.allocate_resources(&device);

        let mut reader = StateSnapshotReader::open(&snapshot_path, fingerprint).unwrap();
        arena.read_full_state(&mut reader, STATE_RESOURCE).unwrap();
        reader.finish().unwrap();

        let mut actual = vec![0; expected.len()];
        arena.buffer().read_bytes(0, &mut actual);
        assert_eq!(actual, expected);
        std::fs::remove_file(snapshot_path).unwrap();
    }

    fn snapshot_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "psi-dec-page-arena-unload-load-{}-{name}.state",
            std::process::id()
        ))
    }

    struct TestRandom(u64);

    impl TestRandom {
        fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 as u32
        }
    }
}
