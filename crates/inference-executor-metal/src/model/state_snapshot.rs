use std::fs::File;
use std::fs::OpenOptions;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher as _;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use crc32fast::Hasher;
use inference_backend_metal::metal::Buffer;
use inference_executor_core::def::ModelExecutorError;

const SNAPSHOT_MAGIC: [u8; 8] = *b"PSISTATE";
const SNAPSHOT_VERSION: u32 = 1;
const HEADER_BYTES: u64 = 40;
const SECTION_HEADER_BYTES: u64 = 24;
const COPY_BUFFER_BYTES: usize = 4 * 1024 * 1024;

static NEXT_TEMP_FILE_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_FINGERPRINT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelFingerprint([u8; 16]);

impl ModelFingerprint {
    pub fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub fn bytes(self) -> [u8; 16] {
        self.0
    }

    pub fn for_process_instance(label: &str) -> Self {
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must not precede the Unix epoch")
            .as_nanos();
        let sequence = NEXT_FINGERPRINT_ID.fetch_add(1, Ordering::Relaxed);
        let mut hasher = DefaultHasher::new();
        label.hash(&mut hasher);
        std::process::id().hash(&mut hasher);
        sequence.hash(&mut hasher);
        let entropy = hasher.finish();
        Self((time ^ (u128::from(entropy) << 64) ^ u128::from(sequence)).to_le_bytes())
    }
}

pub struct StateSnapshotWriter {
    destination: PathBuf,
    temp_path: PathBuf,
    file: Option<File>,
    fingerprint: ModelFingerprint,
    resources: Vec<u32>,
    copy_buffer: Vec<u8>,
}

pub struct StateSnapshotReader {
    file: File,
    sections: Vec<StateSnapshotSection>,
    consumed: Vec<bool>,
    copy_buffer: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StateSnapshotSection {
    resource: u32,
    payload_offset: u64,
    payload_bytes: u64,
}

impl StateSnapshotWriter {
    pub fn new(destination: &Path, fingerprint: ModelFingerprint) -> Result<Self, ModelExecutorError> {
        let parent = destination.parent().ok_or_else(|| {
            ModelExecutorError::custom(format!("model state snapshot path has no parent: {destination:?}"))
        })?;
        std::fs::create_dir_all(parent).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to create model state snapshot directory {parent:?}: {error}"
            ))
        })?;
        let temp_path = temp_path(destination);
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&temp_path)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to create temporary model state snapshot {temp_path:?}: {error}"
                ))
            })?;
        file.write_all(&[0; HEADER_BYTES as usize]).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to initialize temporary model state snapshot {temp_path:?}: {error}"
            ))
        })?;
        Ok(Self {
            destination: destination.to_path_buf(),
            temp_path,
            file: Some(file),
            fingerprint,
            resources: Vec::new(),
            copy_buffer: vec![0; COPY_BUFFER_BYTES],
        })
    }

    pub fn write_buffer(&mut self, resource: u32, buffer: &Buffer) -> Result<(), ModelExecutorError> {
        if resource == 0 || buffer.len_bytes() == 0 {
            return Err(ModelExecutorError::custom(
                "model state snapshot resources and buffers must not be empty",
            ));
        }
        if self.resources.contains(&resource) {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot resource was written twice: resource={resource}"
            )));
        }

        let payload_bytes = buffer.len_bytes_u64();
        let file = self
            .file
            .as_mut()
            .expect("model state snapshot writer file must exist before commit");
        let section_header_offset = file.stream_position().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect temporary model state snapshot {:?}: {error}",
                self.temp_path
            ))
        })?;
        file.write_all(&[0; SECTION_HEADER_BYTES as usize]).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to reserve model state section header in {:?}: {error}",
                self.temp_path
            ))
        })?;

        // TODO(model-state-io): Replace CPU staging with aligned direct or mapped I/O between shared Metal buffers
        // and snapshot storage. Preserve full-file validation and atomic publication.
        let mut hasher = Hasher::new();
        let mut start_bytes = 0;
        while start_bytes < buffer.len_bytes() {
            let chunk_bytes = (buffer.len_bytes() - start_bytes).min(self.copy_buffer.len());
            let chunk = &mut self.copy_buffer[..chunk_bytes];
            buffer.read_bytes(start_bytes, chunk);
            hasher.update(chunk);
            file.write_all(chunk).map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to write model state resource {resource} to {:?}: {error}",
                    self.temp_path
                ))
            })?;
            start_bytes += chunk_bytes;
        }

        let payload_end = file.stream_position().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect temporary model state snapshot {:?}: {error}",
                self.temp_path
            ))
        })?;
        let header = section_header(resource, payload_bytes, hasher.finalize());
        file.seek(SeekFrom::Start(section_header_offset))
            .and_then(|_| file.write_all(&header))
            .and_then(|_| file.seek(SeekFrom::Start(payload_end)))
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to finalize model state resource {resource} in {:?}: {error}",
                    self.temp_path
                ))
            })?;
        self.resources.push(resource);
        Ok(())
    }

    pub fn write_bytes(&mut self, resource: u32, bytes: &[u8]) -> Result<(), ModelExecutorError> {
        if resource == 0 || bytes.is_empty() {
            return Err(ModelExecutorError::custom(
                "model state snapshot resources and byte payloads must not be empty",
            ));
        }
        if self.resources.contains(&resource) {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot resource was written twice: resource={resource}"
            )));
        }

        let payload_bytes = u64::try_from(bytes.len())
            .map_err(|_| ModelExecutorError::custom("model state byte payload length must fit u64"))?;
        let file = self
            .file
            .as_mut()
            .expect("model state snapshot writer file must exist before commit");
        let section_header_offset = file.stream_position().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect temporary model state snapshot {:?}: {error}",
                self.temp_path
            ))
        })?;
        file.write_all(&[0; SECTION_HEADER_BYTES as usize]).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to reserve model state section header in {:?}: {error}",
                self.temp_path
            ))
        })?;
        file.write_all(bytes).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to write model state resource {resource} to {:?}: {error}",
                self.temp_path
            ))
        })?;
        let payload_end = file.stream_position().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect temporary model state snapshot {:?}: {error}",
                self.temp_path
            ))
        })?;
        let header = section_header(resource, payload_bytes, crc32fast::hash(bytes));
        file.seek(SeekFrom::Start(section_header_offset))
            .and_then(|_| file.write_all(&header))
            .and_then(|_| file.seek(SeekFrom::Start(payload_end)))
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to finalize model state resource {resource} in {:?}: {error}",
                    self.temp_path
                ))
            })?;
        self.resources.push(resource);
        Ok(())
    }

    pub fn commit(mut self) -> Result<(), ModelExecutorError> {
        let section_count = u64::try_from(self.resources.len())
            .map_err(|_| ModelExecutorError::custom("model state section count must fit u64"))?;
        let header = snapshot_header(self.fingerprint, section_count);
        let temp_path = self.temp_path.clone();
        self.file_mut()
            .seek(SeekFrom::Start(0))
            .and_then(|_| self.file_mut().write_all(&header))
            .and_then(|_| self.file_mut().sync_all())
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to finalize temporary model state snapshot {temp_path:?}: {error}"
                ))
            })?;
        self.file.take();
        if let Err(error) = std::fs::rename(&self.temp_path, &self.destination) {
            let _ = std::fs::remove_file(&self.temp_path);
            return Err(ModelExecutorError::custom(format!(
                "unable to publish model state snapshot {:?} as {:?}: {error}",
                self.temp_path, self.destination
            )));
        }
        if let Some(parent) = self.destination.parent() {
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .map_err(|error| {
                    ModelExecutorError::custom(format!(
                        "unable to sync model state snapshot directory {parent:?}: {error}"
                    ))
                })?;
        }
        Ok(())
    }

    fn file_mut(&mut self) -> &mut File {
        self.file
            .as_mut()
            .expect("model state snapshot writer file must exist before commit")
    }
}

impl Drop for StateSnapshotWriter {
    fn drop(&mut self) {
        if self.file.take().is_some() {
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

impl StateSnapshotReader {
    pub fn open(path: &Path, expected_fingerprint: ModelFingerprint) -> Result<Self, ModelExecutorError> {
        let mut file = File::open(path).map_err(|error| {
            ModelExecutorError::custom(format!("unable to open model state snapshot {path:?}: {error}"))
        })?;
        let file_bytes = file
            .metadata()
            .map_err(|error| {
                ModelExecutorError::custom(format!("unable to inspect model state snapshot {path:?}: {error}"))
            })?
            .len();
        if file_bytes < HEADER_BYTES {
            return Err(ModelExecutorError::custom("model state snapshot header is truncated"));
        }
        let mut header = [0; HEADER_BYTES as usize];
        file.read_exact(&mut header).map_err(|error| {
            ModelExecutorError::custom(format!("unable to read model state snapshot header {path:?}: {error}"))
        })?;
        let (fingerprint, section_count) = parse_snapshot_header(&header)?;
        if fingerprint != expected_fingerprint {
            return Err(ModelExecutorError::custom("model state snapshot fingerprint mismatch"));
        }
        let max_section_count = (file_bytes - HEADER_BYTES) / SECTION_HEADER_BYTES;
        if section_count > max_section_count {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot section count exceeds file capacity: sections={section_count} \
                 max={max_section_count}"
            )));
        }
        let section_capacity = usize::try_from(section_count)
            .map_err(|_| ModelExecutorError::custom("model state snapshot section count must fit usize"))?;
        let mut sections = Vec::with_capacity(section_capacity);
        for _ in 0..section_count {
            let mut section_header = [0; SECTION_HEADER_BYTES as usize];
            file.read_exact(&mut section_header).map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to read model state snapshot section header {path:?}: {error}"
                ))
            })?;
            let (resource, payload_bytes, expected_checksum) = parse_section_header(&section_header)?;
            let payload_offset = file.stream_position().map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to inspect model state snapshot position {path:?}: {error}"
                ))
            })?;
            let payload_end = payload_offset
                .checked_add(payload_bytes)
                .ok_or_else(|| ModelExecutorError::custom("model state snapshot payload offset overflow"))?;
            if payload_end > file_bytes {
                return Err(ModelExecutorError::custom("model state snapshot payload is truncated"));
            }
            let actual_checksum = checksum_file_range(&mut file, payload_bytes)?;
            if actual_checksum != expected_checksum {
                return Err(ModelExecutorError::custom(format!(
                    "model state snapshot section checksum mismatch: resource={resource}"
                )));
            }
            sections.push(StateSnapshotSection {
                resource,
                payload_offset,
                payload_bytes,
            });
        }
        if file.stream_position().map_err(|error| {
            ModelExecutorError::custom(format!("unable to inspect model state snapshot end {path:?}: {error}"))
        })? != file_bytes
        {
            return Err(ModelExecutorError::custom(
                "model state snapshot contains trailing bytes",
            ));
        }
        sections.sort_unstable_by_key(|section| section.resource);
        if sections
            .windows(2)
            .any(|sections| sections[0].resource == sections[1].resource)
        {
            return Err(ModelExecutorError::custom(
                "model state snapshot contains duplicate resources",
            ));
        }
        let consumed = vec![false; sections.len()];
        Ok(Self {
            file,
            sections,
            consumed,
            copy_buffer: vec![0; COPY_BUFFER_BYTES],
        })
    }

    pub fn read_buffer(&mut self, resource: u32, buffer: &Buffer) -> Result<(), ModelExecutorError> {
        let section_index = self
            .sections
            .binary_search_by_key(&resource, |section| section.resource)
            .ok()
            .ok_or_else(|| {
                ModelExecutorError::custom(format!("model state snapshot resource is missing: resource={resource}"))
            })?;
        if self.consumed[section_index] {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot resource was read twice: resource={resource}"
            )));
        }
        let section = self.sections[section_index];
        if section.payload_bytes != buffer.len_bytes_u64() {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot resource length mismatch: resource={resource} expected={} actual={}",
                buffer.len_bytes_u64(),
                section.payload_bytes
            )));
        }
        self.file
            .seek(SeekFrom::Start(section.payload_offset))
            .map_err(|error| {
                ModelExecutorError::custom(format!("unable to seek to model state resource {resource}: {error}"))
            })?;
        let mut start_bytes = 0;
        while start_bytes < buffer.len_bytes() {
            let chunk_bytes = (buffer.len_bytes() - start_bytes).min(self.copy_buffer.len());
            let chunk = &mut self.copy_buffer[..chunk_bytes];
            self.file.read_exact(chunk).map_err(|error| {
                ModelExecutorError::custom(format!("unable to read model state resource {resource}: {error}"))
            })?;
            buffer.write_bytes(start_bytes, chunk);
            start_bytes += chunk_bytes;
        }
        self.consumed[section_index] = true;
        Ok(())
    }

    pub fn read_bytes(&mut self, resource: u32, max_bytes: usize) -> Result<Vec<u8>, ModelExecutorError> {
        let section_index = self
            .sections
            .binary_search_by_key(&resource, |section| section.resource)
            .ok()
            .ok_or_else(|| {
                ModelExecutorError::custom(format!("model state snapshot resource is missing: resource={resource}"))
            })?;
        if self.consumed[section_index] {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot resource was read twice: resource={resource}"
            )));
        }
        let section = self.sections[section_index];
        let payload_bytes = usize::try_from(section.payload_bytes)
            .map_err(|_| ModelExecutorError::custom("model state byte payload length must fit host usize"))?;
        if payload_bytes > max_bytes {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot byte resource exceeds its limit: resource={resource} limit={max_bytes} \
                 actual={payload_bytes}"
            )));
        }
        self.file
            .seek(SeekFrom::Start(section.payload_offset))
            .map_err(|error| {
                ModelExecutorError::custom(format!("unable to seek to model state resource {resource}: {error}"))
            })?;
        let mut bytes = vec![0; payload_bytes];
        self.file.read_exact(&mut bytes).map_err(|error| {
            ModelExecutorError::custom(format!("unable to read model state resource {resource}: {error}"))
        })?;
        self.consumed[section_index] = true;
        Ok(bytes)
    }

    pub fn finish(self) -> Result<(), ModelExecutorError> {
        if let Some((section, _)) = self.sections.iter().zip(self.consumed).find(|(_, consumed)| !consumed) {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot contains an unexpected resource: resource={}",
                section.resource
            )));
        }
        Ok(())
    }
}

fn snapshot_header(fingerprint: ModelFingerprint, section_count: u64) -> [u8; 40] {
    let mut header = [0; 40];
    header[0..8].copy_from_slice(&SNAPSHOT_MAGIC);
    header[8..12].copy_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    header[16..32].copy_from_slice(&fingerprint.bytes());
    header[32..40].copy_from_slice(&section_count.to_le_bytes());
    let checksum = crc32fast::hash(&header[8..40]);
    header[12..16].copy_from_slice(&checksum.to_le_bytes());
    header
}

fn parse_snapshot_header(header: &[u8; 40]) -> Result<(ModelFingerprint, u64), ModelExecutorError> {
    if header[0..8] != SNAPSHOT_MAGIC {
        return Err(ModelExecutorError::custom("model state snapshot magic mismatch"));
    }
    let version = u32::from_le_bytes(header[8..12].try_into().expect("snapshot version width must match"));
    if version != SNAPSHOT_VERSION {
        return Err(ModelExecutorError::custom(format!(
            "model state snapshot version mismatch: expected={SNAPSHOT_VERSION} actual={version}"
        )));
    }
    let expected_checksum = u32::from_le_bytes(header[12..16].try_into().expect("snapshot checksum width must match"));
    let mut checked = *header;
    checked[12..16].fill(0);
    if crc32fast::hash(&checked[8..40]) != expected_checksum {
        return Err(ModelExecutorError::custom(
            "model state snapshot header checksum mismatch",
        ));
    }
    let fingerprint = ModelFingerprint::new(
        header[16..32]
            .try_into()
            .expect("snapshot fingerprint width must match"),
    );
    let section_count = u64::from_le_bytes(
        header[32..40]
            .try_into()
            .expect("snapshot section-count width must match"),
    );
    Ok((fingerprint, section_count))
}

fn section_header(resource: u32, payload_bytes: u64, payload_checksum: u32) -> [u8; 24] {
    let mut header = [0; 24];
    header[0..4].copy_from_slice(&resource.to_le_bytes());
    header[4..8].copy_from_slice(&payload_checksum.to_le_bytes());
    header[8..16].copy_from_slice(&payload_bytes.to_le_bytes());
    let checksum = crc32fast::hash(&header[0..16]);
    header[16..20].copy_from_slice(&checksum.to_le_bytes());
    header
}

fn parse_section_header(header: &[u8; 24]) -> Result<(u32, u64, u32), ModelExecutorError> {
    if header[20..24] != [0; 4] {
        return Err(ModelExecutorError::custom(
            "model state snapshot section reserved bytes are not zero",
        ));
    }
    let expected_checksum = u32::from_le_bytes(header[16..20].try_into().expect("section checksum width must match"));
    let mut checked = *header;
    checked[16..20].fill(0);
    if crc32fast::hash(&checked[0..16]) != expected_checksum {
        return Err(ModelExecutorError::custom(
            "model state snapshot section header checksum mismatch",
        ));
    }
    let resource = u32::from_le_bytes(header[0..4].try_into().expect("section resource width must match"));
    let payload_checksum = u32::from_le_bytes(
        header[4..8]
            .try_into()
            .expect("section payload checksum width must match"),
    );
    let payload_bytes = u64::from_le_bytes(
        header[8..16]
            .try_into()
            .expect("section payload length width must match"),
    );
    if resource == 0 || payload_bytes == 0 {
        return Err(ModelExecutorError::custom(
            "model state snapshot section header is invalid",
        ));
    }
    Ok((resource, payload_bytes, payload_checksum))
}

fn checksum_file_range(file: &mut File, len_bytes: u64) -> Result<u32, ModelExecutorError> {
    let mut hasher = Hasher::new();
    let mut remaining = len_bytes;
    let mut buffer = [0; 64 * 1024];
    while remaining > 0 {
        let chunk_bytes =
            usize::try_from(remaining.min(buffer.len() as u64)).expect("snapshot checksum chunk length must fit usize");
        file.read_exact(&mut buffer[..chunk_bytes])
            .map_err(|error| ModelExecutorError::custom(format!("unable to checksum model state snapshot: {error}")))?;
        hasher.update(&buffer[..chunk_bytes]);
        remaining -= chunk_bytes as u64;
    }
    Ok(hasher.finalize())
}

fn temp_path(destination: &Path) -> PathBuf {
    let file_name = destination
        .file_name()
        .expect("model state snapshot destination must have a file name")
        .to_string_lossy();
    let id = NEXT_TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed);
    destination.with_file_name(format!(".{file_name}.tmp-{}-{id}", std::process::id()))
}
