use std::cell::Cell;
use std::cell::RefCell;
use std::ffi::OsStr;
use std::ptr::NonNull;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::MTL4ComputeCommandEncoder;
use objc2_metal::MTL4CounterHeap;
use objc2_metal::MTL4CounterHeapDescriptor;
use objc2_metal::MTL4CounterHeapType;
use objc2_metal::MTL4TimestampGranularity;
use objc2_metal::MTL4TimestampHeapEntry;
use objc2_metal::MTLDevice;

const GPU_TIMESTAMPS_ENV: &str = "PSI_DEC_METAL_GPU_TIMESTAMPS";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuTimestampGranularity {
    Relaxed,
    Precise,
}

impl GpuTimestampGranularity {
    pub(crate) fn from_environment() -> Option<Self> {
        Self::parse(std::env::var_os(GPU_TIMESTAMPS_ENV).as_deref())
    }

    fn parse(value: Option<&OsStr>) -> Option<Self> {
        match value.and_then(OsStr::to_str) {
            None | Some("") | Some("0" | "off" | "OFF") => None,
            Some("1" | "relaxed" | "RELAXED") => Some(Self::Relaxed),
            Some("precise" | "PRECISE") => Some(Self::Precise),
            Some(value) => panic!("{GPU_TIMESTAMPS_ENV} must be one of off, relaxed, or precise; got {value:?}"),
        }
    }

    fn metal(self) -> MTL4TimestampGranularity {
        match self {
            Self::Relaxed => MTL4TimestampGranularity::Relaxed,
            Self::Precise => MTL4TimestampGranularity::Precise,
        }
    }
}

#[derive(Debug)]
struct CounterHeap {
    raw: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
    capacity: usize,
}

#[derive(Debug)]
pub(super) struct TimestampProfiler {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    granularity: GpuTimestampGranularity,
    frequency_hz: u64,
    available: Cell<bool>,
    heap: RefCell<Option<CounterHeap>>,
}

impl TimestampProfiler {
    pub(super) fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        granularity: GpuTimestampGranularity,
    ) -> Option<Self> {
        let frequency_hz = device.queryTimestampFrequency();
        if frequency_hz == 0 {
            eprintln!("Metal GPU timestamps are unavailable because the device reported a zero timestamp frequency");
            return None;
        }
        Some(Self {
            device,
            granularity,
            frequency_hz,
            available: Cell::new(true),
            heap: RefCell::new(None),
        })
    }

    pub(super) fn is_available(&self) -> bool {
        self.available.get()
    }

    pub(super) fn begin(&self, num_boundaries: usize) -> Option<SubmissionTimestamps> {
        if !self.available.get() {
            return None;
        }
        let needs_allocation = self
            .heap
            .borrow()
            .as_ref()
            .is_none_or(|heap| heap.capacity < num_boundaries);
        if needs_allocation {
            let descriptor = MTL4CounterHeapDescriptor::new();
            descriptor.setType(MTL4CounterHeapType::Timestamp);
            unsafe {
                descriptor.setCount(num_boundaries);
            }
            let raw = match self.device.newCounterHeapWithDescriptor_error(&descriptor) {
                Ok(raw) => raw,
                Err(error) => {
                    self.available.set(false);
                    eprintln!("Metal GPU timestamps are unavailable: {error:?}");
                    return None;
                },
            };
            *self.heap.borrow_mut() = Some(CounterHeap {
                raw,
                capacity: num_boundaries,
            });
        }
        let raw = self
            .heap
            .borrow()
            .as_ref()
            .expect("Metal timestamp heap must exist after allocation")
            .raw
            .clone();
        unsafe {
            raw.invalidateCounterRange(NSRange {
                location: 0,
                length: num_boundaries,
            });
        }
        Some(SubmissionTimestamps {
            heap: raw,
            granularity: self.granularity,
            frequency_hz: self.frequency_hz,
            num_boundaries,
        })
    }
}

#[derive(Debug)]
pub(super) struct SubmissionTimestamps {
    heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
    granularity: GpuTimestampGranularity,
    frequency_hz: u64,
    num_boundaries: usize,
}

impl SubmissionTimestamps {
    pub(super) fn write(&self, encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>, index: usize) {
        unsafe {
            encoder.writeTimestampWithGranularity_intoHeap_atIndex(self.granularity.metal(), &self.heap, index);
        }
    }

    pub(super) fn resolve(&self) -> Option<Vec<Duration>> {
        let data = unsafe {
            self.heap.resolveCounterRange(NSRange {
                location: 0,
                length: self.num_boundaries,
            })
        }?;
        let len_bytes = self
            .num_boundaries
            .checked_mul(std::mem::size_of::<MTL4TimestampHeapEntry>())?;
        if data.length() != len_bytes {
            return None;
        }
        let mut entries = vec![MTL4TimestampHeapEntry { timestamp: 0 }; self.num_boundaries];
        let destination =
            NonNull::new(entries.as_mut_ptr().cast()).expect("timestamp output allocation must be non-null");
        unsafe {
            data.getBytes_length(destination, len_bytes);
        }
        durations_from_timestamps(
            &entries.iter().map(|entry| entry.timestamp).collect::<Vec<_>>(),
            self.frequency_hz,
        )
    }
}

fn durations_from_timestamps(timestamps: &[u64], frequency_hz: u64) -> Option<Vec<Duration>> {
    if timestamps.len() < 2 || timestamps.contains(&0) {
        return None;
    }
    timestamps
        .windows(2)
        .map(|pair| {
            pair[1]
                .checked_sub(pair[0])
                .map(|ticks| duration_from_ticks(ticks, frequency_hz))
        })
        .collect()
}

fn duration_from_ticks(ticks: u64, frequency_hz: u64) -> Duration {
    let seconds = ticks / frequency_hz;
    let remaining_ticks = ticks % frequency_hz;
    let nanoseconds = (u128::from(remaining_ticks) * 1_000_000_000_u128 / u128::from(frequency_hz)) as u32;
    Duration::new(seconds, nanoseconds)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::time::Duration;

    use super::GpuTimestampGranularity;
    use super::durations_from_timestamps;

    #[test]
    fn test_durations_from_timestamps_success() {
        assert_eq!(
            durations_from_timestamps(&[10, 1_010, 3_510], 1_000),
            Some(vec![Duration::from_secs(1), Duration::from_millis(2_500)])
        );
        assert_eq!(durations_from_timestamps(&[10, 9], 1_000), None);
        assert_eq!(durations_from_timestamps(&[0, 10], 1_000), None);
    }

    #[test]
    fn test_parse_success() {
        assert_eq!(GpuTimestampGranularity::parse(None), None);
        assert_eq!(GpuTimestampGranularity::parse(Some(OsStr::new("off"))), None);
        assert_eq!(
            GpuTimestampGranularity::parse(Some(OsStr::new("relaxed"))),
            Some(GpuTimestampGranularity::Relaxed)
        );
        assert_eq!(
            GpuTimestampGranularity::parse(Some(OsStr::new("precise"))),
            Some(GpuTimestampGranularity::Precise)
        );
    }
}
