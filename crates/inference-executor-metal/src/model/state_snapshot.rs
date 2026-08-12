use std::collections::BTreeSet;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Seek;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::BufferIO;
use inference_backend_metal::metal::BufferIOFile;
use inference_backend_metal::metal::BufferIOFileCacheMode;
use inference_executor_core::def::ModelExecutorError;
use wincode::SchemaRead;
use wincode::SchemaReadOwned;
use wincode::SchemaWrite;
use wincode::config::Configuration;
use wincode::config::PREALLOCATION_SIZE_LIMIT_DISABLED;
#[cfg(target_endian = "big")]
use wincode::int_encoding::BigEndian as NativeEndian;
use wincode::int_encoding::FixInt;
#[cfg(target_endian = "little")]
use wincode::int_encoding::LittleEndian as NativeEndian;
use wincode::io::std_read::ReadAdapter;
use wincode::io::std_write::WriteAdapter;
use wincode::len::FixIntLen;

mod full_state_io;
pub use full_state_io::FullStateIO;
pub use full_state_io::GDNStateSnapshotFiles;
pub use full_state_io::GQAStateSnapshotFiles;
pub use full_state_io::PageArenaStateSnapshotFiles;

const SNAPSHOT_MAGIC: [u8; 8] = *b"PSISTATE";
const SNAPSHOT_VERSION: u32 = 2;
const MANIFEST_FILE_NAME: &str = "manifest";

static NEXT_TEMP_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

type StateSnapshotWincodeConfig =
    Configuration<true, PREALLOCATION_SIZE_LIMIT_DISABLED, FixIntLen<u32>, NativeEndian, FixInt, u8>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, SchemaRead, SchemaWrite)]
#[wincode(tag_encoding = "u8")]
pub enum StateSnapshotFile {
    PageArena,
    MainGQARequestPageTable,
    MainGDNRequestStateTable,
    MainGDNRecurrentState,
    MainGDNConvState,
    MTPGQARequestPageTable,
    DSparkGQARequestPageTable,
}

impl StateSnapshotFile {
    pub fn file_name(self) -> &'static str {
        match self {
            Self::PageArena => "page-arena",
            Self::MainGQARequestPageTable => "main-gqa-request-page-table",
            Self::MainGDNRequestStateTable => "main-gdn-request-state-table",
            Self::MainGDNRecurrentState => "main-gdn-recurrent-state",
            Self::MainGDNConvState => "main-gdn-conv-state",
            Self::MTPGQARequestPageTable => "mtp-gqa-request-page-table",
            Self::DSparkGQARequestPageTable => "dspark-gqa-request-page-table",
        }
    }

    fn kind(self) -> StateSnapshotFileKind {
        match self {
            Self::MainGDNRequestStateTable => StateSnapshotFileKind::Metadata,
            Self::PageArena
            | Self::MainGQARequestPageTable
            | Self::MainGDNRecurrentState
            | Self::MainGDNConvState
            | Self::MTPGQARequestPageTable
            | Self::DSparkGQARequestPageTable => StateSnapshotFileKind::Buffer,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
#[wincode(tag_encoding = "u8")]
enum StateSnapshotFileKind {
    Buffer,
    Metadata,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
struct StateSnapshotManifestEntry {
    file: StateSnapshotFile,
    kind: StateSnapshotFileKind,
    payload_bytes: u64,
}

#[derive(Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
struct StateSnapshotManifest {
    magic: [u8; 8],
    version: u32,
    files: Vec<StateSnapshotManifestEntry>,
}

enum StateSnapshotWriteFile {
    Buffer(BufferIOFile),
    Metadata(File),
}

impl StateSnapshotWriteFile {
    fn sync_all(&self) -> std::io::Result<()> {
        match self {
            Self::Buffer(file) => file.sync_all(),
            Self::Metadata(file) => file.sync_all(),
        }
    }
}

pub struct StateSnapshotWriter<'a> {
    destination: PathBuf,
    temp_path: PathBuf,
    buffer_io: &'a BufferIO,
    expected_files: Box<[StateSnapshotFile]>,
    files: Vec<(StateSnapshotManifestEntry, StateSnapshotWriteFile)>,
    published: bool,
}

pub struct StateSnapshotReader<'a> {
    path: PathBuf,
    buffer_io: &'a BufferIO,
    files: Vec<StateSnapshotManifestEntry>,
    consumed: Vec<bool>,
}

impl<'a> StateSnapshotWriter<'a> {
    pub fn new(
        destination: &Path,
        expected_files: &[StateSnapshotFile],
        buffer_io: &'a BufferIO,
    ) -> Result<Self, ModelExecutorError> {
        validate_expected_file_set(expected_files)?;
        let parent = destination.parent().ok_or_else(|| {
            ModelExecutorError::custom(format!("model state snapshot path has no parent: {destination:?}"))
        })?;
        destination.file_name().ok_or_else(|| {
            ModelExecutorError::custom(format!("model state snapshot path has no file name: {destination:?}"))
        })?;
        std::fs::create_dir_all(parent).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to create model state snapshot parent directory {parent:?}: {error}"
            ))
        })?;
        match std::fs::symlink_metadata(destination) {
            Ok(_) => {
                return Err(ModelExecutorError::custom(format!(
                    "model state snapshot destination already exists: {destination:?}"
                )));
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
            Err(error) => {
                return Err(ModelExecutorError::custom(format!(
                    "unable to inspect model state snapshot destination {destination:?}: {error}"
                )));
            },
        }

        let temp_path = temp_path(destination);
        std::fs::create_dir(&temp_path).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to create temporary model state snapshot directory {temp_path:?}: {error}"
            ))
        })?;
        Ok(Self {
            destination: destination.to_path_buf(),
            temp_path,
            buffer_io,
            expected_files: expected_files.into(),
            files: Vec::new(),
            published: false,
        })
    }

    pub fn write_buffer(
        &mut self,
        snapshot_file: StateSnapshotFile,
        buffer: &Buffer,
    ) -> Result<(), ModelExecutorError> {
        if snapshot_file.kind() != StateSnapshotFileKind::Buffer {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file does not contain a buffer: file={snapshot_file:?}"
            )));
        }
        if buffer.len_bytes() == 0 {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot buffer must not be empty: file={snapshot_file:?}"
            )));
        }
        self.assert_file_is_new(snapshot_file)?;
        let path = self.temp_path.join(snapshot_file.file_name());
        let output_file = self
            .buffer_io
            .create(&path, BufferIOFileCacheMode::Uncached)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to create model state buffer file {snapshot_file:?} at {path:?}: {error}"
                ))
            })?;
        let payload_bytes = buffer.len_bytes_u64();
        self.buffer_io
            .buffer_to_file(buffer, 0, &output_file, 0, payload_bytes)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to write model state buffer file {snapshot_file:?} to {path:?}: {error}"
                ))
            })?;
        self.files.push((
            StateSnapshotManifestEntry {
                file: snapshot_file,
                kind: StateSnapshotFileKind::Buffer,
                payload_bytes,
            },
            StateSnapshotWriteFile::Buffer(output_file),
        ));
        Ok(())
    }

    pub fn write_metadata<T>(
        &mut self,
        snapshot_file: StateSnapshotFile,
        metadata: &T,
    ) -> Result<(), ModelExecutorError>
    where
        T: SchemaWrite<StateSnapshotWincodeConfig, Src = T> + ?Sized,
    {
        if snapshot_file.kind() != StateSnapshotFileKind::Metadata {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file does not contain metadata: file={snapshot_file:?}"
            )));
        }
        self.assert_file_is_new(snapshot_file)?;
        let payload_bytes =
            wincode::config::serialized_size(metadata, state_snapshot_wincode_config()).map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to size model state metadata file {snapshot_file:?}: {error}"
                ))
            })?;
        if payload_bytes == 0 {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot metadata must not be empty: file={snapshot_file:?}"
            )));
        }
        let path = self.temp_path.join(snapshot_file.file_name());
        let mut output_file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to create model state metadata file {snapshot_file:?} at {path:?}: {error}"
                ))
            })?;
        wincode::config::serialize_into(
            WriteAdapter::new(&mut output_file),
            metadata,
            state_snapshot_wincode_config(),
        )
        .map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to encode model state metadata file {snapshot_file:?} to {path:?}: {error}"
            ))
        })?;
        self.files.push((
            StateSnapshotManifestEntry {
                file: snapshot_file,
                kind: StateSnapshotFileKind::Metadata,
                payload_bytes,
            },
            StateSnapshotWriteFile::Metadata(output_file),
        ));
        Ok(())
    }

    pub fn commit(mut self) -> Result<(), ModelExecutorError> {
        self.files.sort_unstable_by_key(|(entry, _)| entry.file);
        let manifest_entries = self.files.iter().map(|(entry, _)| *entry).collect::<Vec<_>>();
        validate_expected_files(&manifest_entries, &self.expected_files)?;
        for (entry, file) in &self.files {
            file.sync_all().map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to sync model state file {:?} in {:?}: {error}",
                    entry.file, self.temp_path
                ))
            })?;
        }

        let manifest = StateSnapshotManifest {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            files: manifest_entries,
        };
        let manifest_bytes = encode_manifest(&manifest)?;
        let manifest_path = self.temp_path.join(MANIFEST_FILE_NAME);
        let mut manifest_file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&manifest_path)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to create model state manifest {manifest_path:?}: {error}"
                ))
            })?;
        manifest_file.write_all(&manifest_bytes).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to write model state manifest {manifest_path:?}: {error}"
            ))
        })?;
        manifest_file.sync_all().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to sync model state manifest {manifest_path:?}: {error}"
            ))
        })?;
        drop(manifest_file);
        self.files.clear();

        File::open(&self.temp_path)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to sync temporary model state snapshot directory {:?}: {error}",
                    self.temp_path
                ))
            })?;
        std::fs::rename(&self.temp_path, &self.destination).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to publish model state snapshot directory {:?} as {:?}: {error}",
                self.temp_path, self.destination
            ))
        })?;
        self.published = true;

        let parent = self
            .destination
            .parent()
            .expect("model state snapshot destination parent must remain available");
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to sync model state snapshot parent directory {parent:?}: {error}"
                ))
            })?;
        Ok(())
    }

    fn assert_file_is_new(&self, snapshot_file: StateSnapshotFile) -> Result<(), ModelExecutorError> {
        if self.files.iter().any(|(entry, _)| entry.file == snapshot_file) {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file was written twice: file={snapshot_file:?}"
            )));
        }
        Ok(())
    }
}

impl Drop for StateSnapshotWriter<'_> {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_dir_all(&self.temp_path);
        }
    }
}

impl<'a> StateSnapshotReader<'a> {
    pub fn open(
        path: &Path,
        expected_files: &[StateSnapshotFile],
        buffer_io: &'a BufferIO,
    ) -> Result<Self, ModelExecutorError> {
        validate_expected_file_set(expected_files)?;
        let directory_metadata = std::fs::symlink_metadata(path).map_err(|error| {
            ModelExecutorError::custom(format!("unable to inspect model state snapshot {path:?}: {error}"))
        })?;
        if !directory_metadata.file_type().is_dir() {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot must be a directory: {path:?}"
            )));
        }

        let manifest_path = path.join(MANIFEST_FILE_NAME);
        let manifest_bytes = std::fs::read(&manifest_path).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to read model state manifest {manifest_path:?}: {error}"
            ))
        })?;
        let manifest: StateSnapshotManifest =
            wincode::config::deserialize_exact(&manifest_bytes, state_snapshot_wincode_config()).map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to decode model state manifest {manifest_path:?}: {error}"
                ))
            })?;
        validate_manifest(&manifest)?;
        validate_expected_files(&manifest.files, expected_files)?;
        validate_snapshot_directory(path, &manifest.files)?;

        let consumed = vec![false; manifest.files.len()];
        Ok(Self {
            path: path.to_path_buf(),
            buffer_io,
            files: manifest.files,
            consumed,
        })
    }

    pub fn read_buffer(&mut self, snapshot_file: StateSnapshotFile, buffer: &Buffer) -> Result<(), ModelExecutorError> {
        let file_index = self.file_index(snapshot_file, StateSnapshotFileKind::Buffer)?;
        let entry = self.files[file_index];
        if entry.payload_bytes != buffer.len_bytes_u64() {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file length mismatch: file={snapshot_file:?} expected={} actual={}",
                buffer.len_bytes_u64(),
                entry.payload_bytes
            )));
        }
        let path = self.path.join(snapshot_file.file_name());
        let input_file = self
            .buffer_io
            .open(&path, BufferIOFileCacheMode::Uncached)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to open model state buffer file {snapshot_file:?} at {path:?}: {error}"
                ))
            })?;
        self.buffer_io
            .file_to_buffer(&input_file, 0, buffer, 0, entry.payload_bytes)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to read model state buffer file {snapshot_file:?} from {path:?}: {error}"
                ))
            })?;
        self.consumed[file_index] = true;
        Ok(())
    }

    pub fn read_metadata<T>(&mut self, snapshot_file: StateSnapshotFile) -> Result<T, ModelExecutorError>
    where
        T: SchemaReadOwned<StateSnapshotWincodeConfig, Dst = T>,
    {
        let file_index = self.file_index(snapshot_file, StateSnapshotFileKind::Metadata)?;
        let entry = self.files[file_index];
        let path = self.path.join(snapshot_file.file_name());
        let mut input_file = File::open(&path).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to open model state metadata file {snapshot_file:?} at {path:?}: {error}"
            ))
        })?;
        let metadata =
            wincode::config::deserialize_from(ReadAdapter::new(&mut input_file), state_snapshot_wincode_config())
                .map_err(|error| {
                    ModelExecutorError::custom(format!(
                        "unable to decode model state metadata file {snapshot_file:?} from {path:?}: {error}"
                    ))
                })?;
        let consumed_bytes = input_file.stream_position().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect model state metadata file {snapshot_file:?} at {path:?}: {error}"
            ))
        })?;
        if consumed_bytes != entry.payload_bytes {
            return Err(ModelExecutorError::custom(format!(
                "model state metadata file contains trailing bytes: file={snapshot_file:?} expected={} \
                 consumed={consumed_bytes}",
                entry.payload_bytes
            )));
        }
        self.consumed[file_index] = true;
        Ok(metadata)
    }

    pub fn finish(self) -> Result<(), ModelExecutorError> {
        if let Some((entry, _)) = self.files.iter().zip(self.consumed).find(|(_, consumed)| !consumed) {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file was not read: file={:?}",
                entry.file
            )));
        }
        Ok(())
    }

    fn file_index(
        &self,
        snapshot_file: StateSnapshotFile,
        expected_kind: StateSnapshotFileKind,
    ) -> Result<usize, ModelExecutorError> {
        let file_index = self
            .files
            .binary_search_by_key(&snapshot_file, |entry| entry.file)
            .map_err(|_| {
                ModelExecutorError::custom(format!("model state snapshot file is missing: file={snapshot_file:?}"))
            })?;
        if self.consumed[file_index] {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file was read twice: file={snapshot_file:?}"
            )));
        }
        let actual_kind = self.files[file_index].kind;
        if actual_kind != expected_kind {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file kind mismatch: file={snapshot_file:?} expected={expected_kind:?} \
                 actual={actual_kind:?}"
            )));
        }
        Ok(file_index)
    }
}

const fn state_snapshot_wincode_config() -> StateSnapshotWincodeConfig {
    Configuration::default()
        .disable_preallocation_size_limit()
        .with_length_encoding::<FixIntLen<u32>>()
        .with_platform_endian()
        .with_fixint_encoding()
        .with_tag_encoding::<u8>()
}

fn encode_manifest(manifest: &StateSnapshotManifest) -> Result<Vec<u8>, ModelExecutorError> {
    wincode::config::serialize(manifest, state_snapshot_wincode_config())
        .map_err(|error| ModelExecutorError::custom(format!("unable to encode model state manifest: {error}")))
}

fn validate_manifest(manifest: &StateSnapshotManifest) -> Result<(), ModelExecutorError> {
    if manifest.magic != SNAPSHOT_MAGIC {
        return Err(ModelExecutorError::custom("model state snapshot magic mismatch"));
    }
    if manifest.version != SNAPSHOT_VERSION {
        return Err(ModelExecutorError::custom(format!(
            "model state snapshot version mismatch: expected={SNAPSHOT_VERSION} actual={}",
            manifest.version
        )));
    }
    if manifest.files.is_empty() {
        return Err(ModelExecutorError::custom(
            "model state snapshot must contain at least one file",
        ));
    }
    if manifest
        .files
        .windows(2)
        .any(|entries| entries[0].file >= entries[1].file)
    {
        return Err(ModelExecutorError::custom(
            "model state snapshot files must be unique and sorted",
        ));
    }
    if manifest.files.iter().any(|entry| entry.payload_bytes == 0) {
        return Err(ModelExecutorError::custom(
            "model state snapshot files must not be empty",
        ));
    }
    if manifest.files.iter().any(|entry| entry.kind != entry.file.kind()) {
        return Err(ModelExecutorError::custom(
            "model state snapshot file kind does not match its semantic file",
        ));
    }
    Ok(())
}

fn validate_snapshot_directory(path: &Path, files: &[StateSnapshotManifestEntry]) -> Result<(), ModelExecutorError> {
    let mut expected_names = files
        .iter()
        .map(|entry| entry.file.file_name())
        .collect::<BTreeSet<_>>();
    expected_names.insert(MANIFEST_FILE_NAME);

    for entry in std::fs::read_dir(path).map_err(|error| {
        ModelExecutorError::custom(format!("unable to enumerate model state snapshot {path:?}: {error}"))
    })? {
        let entry = entry.map_err(|error| {
            ModelExecutorError::custom(format!("unable to enumerate model state snapshot {path:?}: {error}"))
        })?;
        let file_name = entry.file_name();
        let file_name = file_name.to_str().ok_or_else(|| {
            ModelExecutorError::custom(format!(
                "model state snapshot contains a non-UTF-8 file name: {file_name:?}"
            ))
        })?;
        if !expected_names.remove(file_name) {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot contains an unexpected file: {file_name:?}"
            )));
        }
        let file_type = entry.file_type().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect model state snapshot entry {:?}: {error}",
                entry.path()
            ))
        })?;
        if !file_type.is_file() {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot entry must be a regular file: {:?}",
                entry.path()
            )));
        }
    }
    if let Some(file_name) = expected_names.first() {
        return Err(ModelExecutorError::custom(format!(
            "model state snapshot file is missing: {file_name:?}"
        )));
    }

    for entry in files {
        let file_path = path.join(entry.file.file_name());
        let actual_bytes = std::fs::metadata(&file_path)
            .map_err(|error| {
                ModelExecutorError::custom(format!("unable to inspect model state file {:?}: {error}", entry.file))
            })?
            .len();
        if actual_bytes != entry.payload_bytes {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file length mismatch: file={:?} expected={} actual={actual_bytes}",
                entry.file, entry.payload_bytes
            )));
        }
    }
    Ok(())
}

fn validate_expected_files(
    actual_files: &[StateSnapshotManifestEntry],
    expected_files: &[StateSnapshotFile],
) -> Result<(), ModelExecutorError> {
    let expected = validate_expected_file_set(expected_files)?;
    let actual = actual_files.iter().map(|entry| entry.file).collect::<BTreeSet<_>>();
    if let Some(file) = expected.difference(&actual).next() {
        return Err(ModelExecutorError::custom(format!(
            "model state snapshot file is missing: file={file:?}"
        )));
    }
    if let Some(file) = actual.difference(&expected).next() {
        return Err(ModelExecutorError::custom(format!(
            "model state snapshot contains an unexpected file: file={file:?}"
        )));
    }
    Ok(())
}

fn validate_expected_file_set(
    expected_files: &[StateSnapshotFile],
) -> Result<BTreeSet<StateSnapshotFile>, ModelExecutorError> {
    let expected = expected_files.iter().copied().collect::<BTreeSet<_>>();
    if expected.len() != expected_files.len() {
        return Err(ModelExecutorError::custom(
            "model state snapshot expected file set contains duplicates",
        ));
    }
    if expected.is_empty() {
        return Err(ModelExecutorError::custom(
            "model state snapshot expected file set must not be empty",
        ));
    }
    Ok(expected)
}

fn temp_path(destination: &Path) -> PathBuf {
    let file_name = destination
        .file_name()
        .expect("model state snapshot destination must have a file name")
        .to_string_lossy();
    let id = NEXT_TEMP_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
    destination.with_file_name(format!(".{file_name}.tmp-{}-{id}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::BufferIO;
    use inference_backend_metal::metal::Device;
    use wincode::SchemaRead;
    use wincode::SchemaWrite;

    use super::StateSnapshotFile;
    use super::StateSnapshotReader;
    use super::StateSnapshotWriter;

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
    struct TestMetadata {
        slots: Vec<u32>,
    }

    const PAGE_ARENA_FILES: &[StateSnapshotFile] = &[StateSnapshotFile::PageArena];
    const PAGE_ARENA_AND_GDN_METADATA_FILES: &[StateSnapshotFile] = &[
        StateSnapshotFile::PageArena,
        StateSnapshotFile::MainGDNRequestStateTable,
    ];
    const GDN_METADATA_FILES: &[StateSnapshotFile] = &[StateSnapshotFile::MainGDNRequestStateTable];

    #[test]
    fn test_directory_snapshot_round_trip() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source_values = (0..4096).map(|index| (index % 251) as u8).collect::<Vec<_>>();
        let source = Buffer::from_slice(&device, &source_values);
        let restored = Buffer::new_zeroed(&device, source_values.len());
        let metadata = TestMetadata {
            slots: vec![3, 5, 8, 13],
        };
        let path = test_path("round-trip");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_AND_GDN_METADATA_FILES, &buffer_io).unwrap();
        writer.write_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer
            .write_metadata(StateSnapshotFile::MainGDNRequestStateTable, &metadata)
            .unwrap();
        writer.commit().unwrap();

        assert!(path.join("manifest").is_file());
        assert!(path.join("page-arena").is_file());
        assert!(path.join("main-gdn-request-state-table").is_file());

        let mut reader = StateSnapshotReader::open(&path, PAGE_ARENA_AND_GDN_METADATA_FILES, &buffer_io).unwrap();
        reader.read_buffer(StateSnapshotFile::PageArena, &restored).unwrap();
        let restored_metadata: TestMetadata = reader
            .read_metadata(StateSnapshotFile::MainGDNRequestStateTable)
            .unwrap();
        reader.finish().unwrap();

        assert_eq!(restored.read_typed::<u8>(0, source_values.len()), source_values);
        assert_eq!(restored_metadata, metadata);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_reader_rejects_unexpected_file() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("unexpected-file");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &buffer_io).unwrap();
        writer.write_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer.commit().unwrap();
        std::fs::write(path.join("unexpected"), b"unexpected").unwrap();

        let error = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &buffer_io)
            .err()
            .expect("unexpected file must fail snapshot validation");
        assert!(error.to_string().contains("unexpected file"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_reader_rejects_truncated_file_before_restore() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("truncated-file");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &buffer_io).unwrap();
        writer.write_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer.commit().unwrap();
        OpenOptions::new()
            .write(true)
            .open(path.join("page-arena"))
            .unwrap()
            .set_len(8)
            .unwrap();

        let error = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &buffer_io)
            .err()
            .expect("truncated file must fail snapshot validation");
        assert!(error.to_string().contains("length mismatch"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_writer_rejects_semantic_file_kind_mismatch() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("writer-kind-mismatch");
        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &buffer_io).unwrap();

        let buffer_error = writer
            .write_buffer(StateSnapshotFile::MainGDNRequestStateTable, &source)
            .unwrap_err();
        assert!(buffer_error.to_string().contains("does not contain a buffer"));
        let metadata_error = writer
            .write_metadata(StateSnapshotFile::PageArena, &TestMetadata { slots: vec![1] })
            .unwrap_err();
        assert!(metadata_error.to_string().contains("does not contain metadata"));

        drop(writer);
        assert!(!path.exists());
    }

    #[test]
    fn test_reader_rejects_manifest_kind_mismatch() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("manifest-kind-mismatch");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &buffer_io).unwrap();
        writer.write_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer.commit().unwrap();

        let mut manifest = read_manifest(&path);
        manifest.files[0].kind = super::StateSnapshotFileKind::Metadata;
        std::fs::write(path.join("manifest"), super::encode_manifest(&manifest).unwrap()).unwrap();

        let error = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &buffer_io)
            .err()
            .expect("semantic file kind mismatch must fail snapshot validation");
        assert!(error.to_string().contains("kind does not match"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_reader_rejects_duplicate_expected_file() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("duplicate-expected-file");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &buffer_io).unwrap();
        writer.write_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer.commit().unwrap();

        let error = StateSnapshotReader::open(
            &path,
            &[StateSnapshotFile::PageArena, StateSnapshotFile::PageArena],
            &buffer_io,
        )
        .err()
        .expect("duplicate expected file must fail snapshot validation");
        assert!(error.to_string().contains("contains duplicates"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_writer_rejects_incomplete_expected_file_set() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("incomplete-expected-file-set");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_AND_GDN_METADATA_FILES, &buffer_io).unwrap();
        writer.write_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        let error = writer.commit().unwrap_err();

        assert!(error.to_string().contains("file is missing"));
        assert!(!path.exists());
    }

    #[test]
    fn test_writer_rejects_unexpected_file() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("writer-unexpected-file");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &buffer_io).unwrap();
        writer.write_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer
            .write_buffer(StateSnapshotFile::MainGQARequestPageTable, &source)
            .unwrap();
        let error = writer.commit().unwrap_err();

        assert!(error.to_string().contains("unexpected file"));
        assert!(!path.exists());
    }

    #[test]
    fn test_writer_rejects_duplicate_file() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("writer-duplicate-file");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &buffer_io).unwrap();
        writer.write_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        let error = writer.write_buffer(StateSnapshotFile::PageArena, &source).unwrap_err();

        assert!(error.to_string().contains("written twice"));
        drop(writer);
        assert!(!path.exists());
    }

    #[test]
    fn test_metadata_reader_rejects_trailing_bytes() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let path = test_path("metadata-trailing-bytes");

        let mut writer = StateSnapshotWriter::new(&path, GDN_METADATA_FILES, &buffer_io).unwrap();
        writer
            .write_metadata(
                StateSnapshotFile::MainGDNRequestStateTable,
                &TestMetadata { slots: vec![1, 2, 3] },
            )
            .unwrap();
        writer.commit().unwrap();

        let metadata_path = path.join("main-gdn-request-state-table");
        OpenOptions::new()
            .append(true)
            .open(&metadata_path)
            .unwrap()
            .write_all(&[0xaa])
            .unwrap();
        let mut manifest = read_manifest(&path);
        manifest.files[0].payload_bytes += 1;
        std::fs::write(path.join("manifest"), super::encode_manifest(&manifest).unwrap()).unwrap();

        let mut reader = StateSnapshotReader::open(&path, GDN_METADATA_FILES, &buffer_io).unwrap();
        let error = reader
            .read_metadata::<TestMetadata>(StateSnapshotFile::MainGDNRequestStateTable)
            .unwrap_err();
        assert!(error.to_string().contains("trailing bytes"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_uncommitted_snapshot_drop_removes_temporary_directory() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let path = test_path("uncommitted-drop");
        let writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &buffer_io).unwrap();
        let temp_path = writer.temp_path.clone();

        assert!(temp_path.is_dir());
        assert!(!path.exists());
        drop(writer);
        assert!(!temp_path.exists());
        assert!(!path.exists());
    }

    fn read_manifest(path: &std::path::Path) -> super::StateSnapshotManifest {
        let bytes = std::fs::read(path.join("manifest")).unwrap();
        wincode::config::deserialize_exact(&bytes, super::state_snapshot_wincode_config()).unwrap()
    }

    fn test_path(name: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "psi-dec-state-snapshot-test-{}-{name}-{id}",
            std::process::id()
        ))
    }
}
