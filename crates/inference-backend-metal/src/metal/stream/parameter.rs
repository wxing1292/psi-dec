use std::cell::RefCell;
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLDevice;
use objc2_metal::MTLResourceOptions;

use crate::metal::record_gpu_buffer_alloc;
use crate::metal::stream::PARAMETER_BUFFER_ALIGNMENT;
use crate::metal::stream::TrackedGpuAllocation;

/// Stable lookup key for one submission-time replay value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayParameterKey(&'static str);

impl ReplayParameterKey {
    pub const fn new(name: &'static str) -> Self {
        assert!(!name.is_empty(), "Metal replay parameter name must not be empty");
        Self(name)
    }
}

/// Submission values keyed by the replay parameter table declared while recording.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct ReplayArguments {
    values: HashMap<ReplayParameterKey, u32>,
}

impl ReplayArguments {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_u32(&mut self, key: ReplayParameterKey, value: u32) {
        assert!(
            self.values.insert(key, value).is_none(),
            "Metal replay argument {:?} was set twice",
            key
        );
    }

    pub fn with_u32(mut self, key: ReplayParameterKey, value: u32) -> Self {
        self.set_u32(key, value);
        self
    }
}

#[derive(Debug)]
pub struct ReplayParameterTable {
    entries: HashMap<ReplayParameterKey, ReplayParameterEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReplayParameterEntry {
    offset_bytes: usize,
    min_value: u32,
    max_value: u32,
}

impl ReplayParameterTable {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn validate(&self, arguments: &ReplayArguments) {
        assert_eq!(
            arguments.values.len(),
            self.entries.len(),
            "Metal replay submission must provide every declared parameter exactly once"
        );
        for (&key, entry) in &self.entries {
            let value = arguments
                .values
                .get(&key)
                .copied()
                .unwrap_or_else(|| panic!("Metal replay submission is missing parameter {:?}", key));
            assert!(
                value >= entry.min_value && value <= entry.max_value,
                "Metal replay parameter {:?} value={} is outside {}..={}",
                key,
                value,
                entry.min_value,
                entry.max_value
            );
        }
    }

    pub fn write(&self, buffer: &ProtocolObject<dyn MTLBuffer>, arguments: &ReplayArguments) {
        for (&key, entry) in &self.entries {
            let value = arguments.values[&key];
            assert!(
                entry.offset_bytes + std::mem::size_of::<u32>() <= buffer.length(),
                "Metal replay parameter offset exceeds parameter buffer"
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    std::ptr::from_ref(&value).cast::<u8>(),
                    buffer.contents().as_ptr().cast::<u8>().add(entry.offset_bytes),
                    std::mem::size_of::<u32>(),
                );
            }
        }
    }
}

/// Build-time host packer for fixed kernel arguments and dynamic replay parameters.
#[derive(Debug, Default)]
pub struct CommandParameterLayoutBuilder {
    bytes: RefCell<Vec<u8>>,
    replay_entries: RefCell<HashMap<ReplayParameterKey, ReplayParameterEntry>>,
}

#[derive(Debug)]
pub struct CommandParameterLayout {
    pub bytes: Vec<u8>,
    pub replay_parameter_table: ReplayParameterTable,
}

impl CommandParameterLayoutBuilder {
    pub fn build(self) -> CommandParameterLayout {
        CommandParameterLayout {
            bytes: self.bytes.into_inner(),
            replay_parameter_table: ReplayParameterTable {
                entries: self.replay_entries.into_inner(),
            },
        }
    }

    pub fn push_bytes<T>(&self, values: &[T]) -> usize {
        let len_bytes = std::mem::size_of_val(values);
        assert!(len_bytes > 0);
        let mut bytes = self.bytes.borrow_mut();
        let offset = align_up(bytes.len(), PARAMETER_BUFFER_ALIGNMENT);
        let end = offset
            .checked_add(len_bytes)
            .expect("Metal command parameter byte length overflow");
        bytes.resize(end, 0);
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), bytes.as_mut_ptr().add(offset), len_bytes);
        }
        offset
    }

    pub fn bind_u32(&self, key: ReplayParameterKey, min_value: u32, max_value: u32) -> usize {
        assert!(min_value <= max_value, "Metal replay parameter domain is empty");
        if let Some(entry) = self.replay_entries.borrow().get(&key).copied() {
            assert_eq!(
                (entry.min_value, entry.max_value),
                (min_value, max_value),
                "Metal replay parameter {:?} has inconsistent domains",
                key
            );
            return entry.offset_bytes;
        }

        let offset_bytes = self.push_bytes(&[0_u32]);
        let previous = self.replay_entries.borrow_mut().insert(
            key,
            ReplayParameterEntry {
                offset_bytes,
                min_value,
                max_value,
            },
        );
        assert!(previous.is_none());
        offset_bytes
    }
}

pub fn allocate_parameter_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    allocation_kind: &'static str,
) -> Option<(Retained<ProtocolObject<dyn MTLBuffer>>, TrackedGpuAllocation)> {
    if bytes.is_empty() {
        return None;
    }

    let len_bytes = align_up(bytes.len(), PARAMETER_BUFFER_ALIGNMENT);
    let buffer = device
        .newBufferWithLength_options(
            len_bytes,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        )
        .expect("Metal parameter buffer allocation failed");
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().as_ptr().cast::<u8>(), bytes.len());
    }
    let allocation_site = record_gpu_buffer_alloc(allocation_kind, len_bytes);
    Some((buffer, TrackedGpuAllocation::new(allocation_site, len_bytes)))
}

fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::CommandParameterLayoutBuilder;
    use super::ReplayParameterKey;

    const NUM_ACTIVE_THREADS: ReplayParameterKey = ReplayParameterKey::new("test.num_active_threads");

    #[test]
    fn test_key_reuse() {
        let builder = CommandParameterLayoutBuilder::default();
        let first = builder.bind_u32(NUM_ACTIVE_THREADS, 64, 128);
        let second = builder.bind_u32(NUM_ACTIVE_THREADS, 64, 128);

        assert_eq!(first, second);
        assert_eq!(builder.build().replay_parameter_table.len(), 1);
    }
}
