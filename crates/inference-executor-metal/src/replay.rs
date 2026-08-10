use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use inference_backend_metal::metal::ReplayProgram;

use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::ReplayRecorder;

pub trait ReplayComponent {
    type Key: Clone + Debug + Eq + Hash;
    type Input<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key;

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>);
}

pub struct Replay<T: ReplayComponent> {
    component: T,
    cache: ReplayCache<T::Key>,
}

impl<T: ReplayComponent> Replay<T> {
    pub fn new(name: &'static str, component: T) -> Self {
        Self {
            component,
            cache: ReplayCache::new(name),
        }
    }

    pub fn component(&self) -> &T {
        &self.component
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn record<'a>(&'a mut self, runtime: &MetalReplayRuntime<'_>, input: &T::Input<'a>) -> (T::Key, bool) {
        let key = self.component.replay_key(input);
        let cache_hit = self.cache.contains(&key);
        if !cache_hit {
            let mut recorder = runtime.create_recorder();
            self.component.record(&mut recorder, input);
            self.cache.insert_recorded_replay(key.clone(), recorder.build());
        }
        (key, cache_hit)
    }

    pub fn replay(&self, key: &T::Key) -> &ReplayProgram {
        self.cache.replay(key)
    }
}

struct ReplayCache<K> {
    name: &'static str,
    entries: HashMap<K, ReplayProgram>,
}

impl<K> ReplayCache<K>
where
    K: Debug + Eq + Hash,
{
    fn new(name: &'static str) -> Self {
        Self {
            name,
            entries: HashMap::new(),
        }
    }

    fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    fn insert_recorded_replay(&mut self, key: K, replay: ReplayProgram) {
        match self.entries.entry(key) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(replay);
            },
            std::collections::hash_map::Entry::Occupied(entry) => {
                panic!("{} replay cache filled twice for key {:?}", self.name, entry.key());
            },
        }
    }

    fn replay(&self, key: &K) -> &ReplayProgram {
        self.entries
            .get(key)
            .unwrap_or_else(|| panic!("{} replay cache missing recorded batch for key {:?}", self.name, key))
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Stream;
    use inference_backend_metal::operators::Bf16ConcatRowsBuffers;
    use inference_backend_metal::operators::Bf16ConcatRowsConfig;
    use inference_backend_metal::operators::Bf16ConcatRowsKernel;
    use inference_executor_core::backend::recorder::Recorder;

    use super::Replay;
    use super::ReplayComponent;
    use crate::def::replay_op::MetalReplayRuntime;
    use crate::def::replay_op::ReplayOp;
    use crate::def::replay_op::ReplayRecorder;

    const NUM_ACTIVE_ROWS: inference_backend_metal::metal::ReplayParameterKey =
        inference_backend_metal::metal::ReplayParameterKey::new("test.replay.num_active_rows");

    struct CountedComponent {
        records: Cell<usize>,
        kernel: Bf16ConcatRowsKernel,
        lhs: Buffer,
        rhs: Buffer,
        output: Buffer,
    }

    impl ReplayComponent for CountedComponent {
        type Key = u32;
        type Input<'a> = u32;

        fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
            *input
        }

        fn record<'a>(&'a self, recorder: &mut ReplayRecorder, _input: &Self::Input<'a>) {
            self.records.set(self.records.get() + 1);
            recorder.record(ReplayOp::opaque(self.kernel.invoke_bucketed(
                1,
                NUM_ACTIVE_ROWS,
                Bf16ConcatRowsBuffers {
                    lhs: &self.lhs,
                    rhs: &self.rhs,
                    output: &self.output,
                },
            )));
        }
    }

    fn component(device: &Device) -> CountedComponent {
        CountedComponent {
            records: Cell::new(0),
            kernel: Bf16ConcatRowsKernel::new(device, Bf16ConcatRowsConfig { num_columns: 4 }),
            lhs: Buffer::new_zeroed_elements(device, 4, inference_backend_metal::metal::Dtype::Bfloat16),
            rhs: Buffer::new_zeroed_elements(device, 4, inference_backend_metal::metal::Dtype::Bfloat16),
            output: Buffer::new_zeroed_elements(device, 8, inference_backend_metal::metal::Dtype::Bfloat16),
        }
    }

    #[test]
    fn record_is_idempotent_and_replay_is_strict() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let mut replay = Replay::new("test", component(&device));

        let (key, cache_hit) = replay.record(&runtime, &7);
        assert!(!cache_hit);
        assert_eq!(replay.component().records.get(), 1);
        assert_eq!(replay.record(&runtime, &7), (key, true));
        assert_eq!(replay.component().records.get(), 1);
        let _ = replay.replay(&key);
    }

    #[test]
    #[should_panic(expected = "replay cache missing recorded batch")]
    fn replay_panics_before_record() {
        let device = Device::system_default();
        let replay = Replay::new("test", component(&device));
        let _ = replay.replay(&1);
    }
}
