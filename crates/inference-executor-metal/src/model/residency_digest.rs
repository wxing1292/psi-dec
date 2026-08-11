use std::fs::File;
use std::io::Read;
use std::path::Path;

use crc32fast::Hasher;
use inference_backend_metal::metal::Buffer;
use inference_executor_core::def::ModelExecutorError;

const DIGEST_CHUNK_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelResidencyDigest(u32);

pub struct ModelResidencyHasher {
    digest: Hasher,
}

impl ModelResidencyHasher {
    pub fn new() -> Self {
        Self { digest: Hasher::new() }
    }

    pub fn buffer(&mut self, name: &str, buffer: &Buffer) {
        let mut resource = Hasher::new();
        let len_bytes = buffer.len_bytes();
        let mut start_bytes = 0;
        while start_bytes < len_bytes {
            let chunk_bytes = DIGEST_CHUNK_BYTES.min(len_bytes - start_bytes);
            resource.update(&buffer.read_typed::<u8>(start_bytes, chunk_bytes));
            start_bytes += chunk_bytes;
        }
        self.resource(name, len_bytes as u64, resource.finalize());
    }

    pub fn file(&mut self, name: &str, path: &Path) -> Result<(), ModelExecutorError> {
        let mut file = File::open(path).map_err(|error| {
            ModelExecutorError::custom(format!("unable to open model residency digest input {path:?}: {error}"))
        })?;
        let len_bytes = file
            .metadata()
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to inspect model residency digest input {path:?}: {error}"
                ))
            })?
            .len();
        let mut resource = Hasher::new();
        let mut buffer = vec![0; DIGEST_CHUNK_BYTES];
        loop {
            let bytes = file.read(&mut buffer).map_err(|error| {
                ModelExecutorError::custom(format!("unable to read model residency digest input {path:?}: {error}"))
            })?;
            if bytes == 0 {
                break;
            }
            resource.update(&buffer[..bytes]);
        }
        self.resource(name, len_bytes, resource.finalize());
        Ok(())
    }

    pub fn finish(self) -> ModelResidencyDigest {
        ModelResidencyDigest(self.digest.finalize())
    }

    fn resource(&mut self, name: &str, len_bytes: u64, digest: u32) {
        self.digest.update(&(name.len() as u64).to_le_bytes());
        self.digest.update(name.as_bytes());
        self.digest.update(&len_bytes.to_le_bytes());
        self.digest.update(&digest.to_le_bytes());
    }
}
