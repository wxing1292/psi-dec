use std::collections::VecDeque;
use std::hash::Hash;

use ahash::AHashMap;

#[derive(Clone, Copy)]
struct Entry<T> {
    value: T,
    generation: u64,
}

/// An ordered set with `VecDeque` front and back operations.
///
/// `push_front` moves an existing value to the front. `push_back` keeps an existing value at its current position.
/// Replaced and removed entries become tombstones. Pop operations skip tombstones, and periodic compaction removes
/// tombstones that are not at either end. Compaction keeps the physical entry count at most twice the logical length.
pub struct DedupVecDeque<T>
where
    T: Copy + Eq + Hash,
{
    entries: VecDeque<Entry<T>>,
    current_generations: AHashMap<T, u64>,
    num_tombstones: usize,
    next_generation: u64,
}

impl<T> DedupVecDeque<T>
where
    T: Copy + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            current_generations: AHashMap::new(),
            num_tombstones: 0,
            next_generation: 0,
        }
    }

    pub fn push_front(&mut self, value: T) {
        if self.front().copied() == Some(value) {
            return;
        }
        let generation = self.take_generation();
        if self.current_generations.insert(value, generation).is_some() {
            self.num_tombstones += 1;
        }
        self.entries.push_front(Entry { value, generation });
        self.compact_if_needed();
    }

    pub fn push_back(&mut self, value: T) {
        if self.current_generations.contains_key(&value) {
            return;
        }
        let generation = self.take_generation();
        let previous_generation = self.current_generations.insert(value, generation);
        debug_assert!(
            previous_generation.is_none(),
            "new back entry must not replace a current generation"
        );
        self.entries.push_back(Entry { value, generation });
        self.compact_if_needed();
    }

    pub fn pop_front(&mut self) -> Option<T> {
        while let Some(entry) = self.entries.pop_front() {
            if self.is_current(entry) {
                let generation = self.current_generations.remove(&entry.value);
                debug_assert_eq!(generation, Some(entry.generation), "popped front entry must be current");
                self.compact_if_needed();
                return Some(entry.value);
            }
            debug_assert!(0 < self.num_tombstones, "popped front tombstone must be tracked");
            self.num_tombstones -= 1;
        }
        debug_assert!(
            self.current_generations.is_empty(),
            "empty deduplicated deque must not have current entries"
        );
        None
    }

    pub fn pop_back(&mut self) -> Option<T> {
        while let Some(entry) = self.entries.pop_back() {
            if self.is_current(entry) {
                let generation = self.current_generations.remove(&entry.value);
                debug_assert_eq!(generation, Some(entry.generation), "popped back entry must be current");
                self.compact_if_needed();
                return Some(entry.value);
            }
            debug_assert!(0 < self.num_tombstones, "popped back tombstone must be tracked");
            self.num_tombstones -= 1;
        }
        debug_assert!(
            self.current_generations.is_empty(),
            "empty deduplicated deque must not have current entries"
        );
        None
    }

    pub fn front(&self) -> Option<&T> {
        self.iter().next()
    }

    pub fn back(&self) -> Option<&T> {
        self.iter().next_back()
    }

    pub fn remove(&mut self, value: &T) -> bool {
        let removed = self.current_generations.remove(value).is_some();
        if removed {
            self.num_tombstones += 1;
            self.compact_if_needed();
        }
        removed
    }

    pub fn contains(&self, value: &T) -> bool {
        self.current_generations.contains_key(value)
    }

    pub fn len(&self) -> usize {
        self.current_generations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.current_generations.is_empty()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.entries.iter().filter_map(|entry| {
            (self.current_generations.get(&entry.value) == Some(&entry.generation)).then_some(&entry.value)
        })
    }

    fn take_generation(&mut self) -> u64 {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);
        generation
    }

    fn is_current(&self, entry: Entry<T>) -> bool {
        self.current_generations.get(&entry.value) == Some(&entry.generation)
    }

    fn compact_if_needed(&mut self) {
        debug_assert_eq!(
            self.entries.len(),
            self.current_generations.len() + self.num_tombstones,
            "physical entries must equal current entries plus tombstones"
        );
        if self.num_tombstones <= self.current_generations.len() {
            return;
        }

        let current_generations = &self.current_generations;
        self.entries
            .retain(|entry| current_generations.get(&entry.value) == Some(&entry.generation));
        self.num_tombstones = 0;
        debug_assert_eq!(
            self.entries.len(),
            self.current_generations.len(),
            "compacted deduplicated deque must contain one entry per current value"
        );
    }
}

impl<T> Default for DedupVecDeque<T>
where
    T: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_front_without_duplicate() {
        let mut deque = DedupVecDeque::new();

        deque.push_front(0);
        deque.push_front(1);
        deque.push_front(2);

        assert_eq!(deque.iter().copied().collect::<Vec<_>>(), vec![2, 1, 0]);
        assert_eq!(deque.front(), Some(&2));
        assert_eq!(deque.back(), Some(&0));
        assert_eq!(deque.len(), 3);
    }

    #[test]
    fn test_push_front_with_duplicate() {
        let mut deque = DedupVecDeque::new();

        deque.push_front(0);
        deque.push_front(1);
        deque.push_front(0);

        assert_eq!(deque.iter().copied().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(deque.front(), Some(&0));
        assert_eq!(deque.back(), Some(&1));
        assert_eq!(deque.len(), 2);
    }

    #[test]
    fn test_push_back_without_duplicate() {
        let mut deque = DedupVecDeque::new();

        deque.push_back(0);
        deque.push_back(1);
        deque.push_back(2);

        assert_eq!(deque.iter().copied().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(deque.front(), Some(&0));
        assert_eq!(deque.back(), Some(&2));
        assert_eq!(deque.len(), 3);
    }

    #[test]
    fn test_push_back_with_duplicate() {
        let mut deque = DedupVecDeque::new();

        deque.push_back(0);
        deque.push_back(1);
        deque.push_back(0);

        assert_eq!(deque.iter().copied().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(deque.front(), Some(&0));
        assert_eq!(deque.back(), Some(&1));
        assert_eq!(deque.len(), 2);
    }

    #[test]
    fn test_pop_front_without_duplicate() {
        let mut deque = DedupVecDeque::new();
        deque.push_front(0);
        deque.push_front(1);
        deque.push_front(2);

        assert_eq!(deque.pop_front(), Some(2));
        assert_eq!(deque.pop_front(), Some(1));
        assert_eq!(deque.pop_front(), Some(0));
        assert_eq!(deque.pop_front(), None);
        assert!(deque.is_empty());
    }

    #[test]
    fn test_pop_front_with_duplicate() {
        let mut deque = DedupVecDeque::new();
        deque.push_front(0);
        deque.push_front(1);
        deque.push_front(0);

        assert_eq!(deque.pop_front(), Some(0));
        assert_eq!(deque.pop_front(), Some(1));
        assert_eq!(deque.pop_front(), None);
        assert!(deque.is_empty());
    }

    #[test]
    fn test_pop_back_without_duplicate() {
        let mut deque = DedupVecDeque::new();
        deque.push_back(0);
        deque.push_back(1);
        deque.push_back(2);

        assert_eq!(deque.pop_back(), Some(2));
        assert_eq!(deque.pop_back(), Some(1));
        assert_eq!(deque.pop_back(), Some(0));
        assert_eq!(deque.pop_back(), None);
        assert!(deque.is_empty());
    }

    #[test]
    fn test_pop_back_with_duplicate() {
        let mut deque = DedupVecDeque::new();
        deque.push_back(0);
        deque.push_back(1);
        deque.push_back(0);

        assert_eq!(deque.pop_back(), Some(1));
        assert_eq!(deque.pop_back(), Some(0));
        assert_eq!(deque.pop_back(), None);
        assert!(deque.is_empty());
    }

    #[test]
    fn test_remove_existing() {
        let mut deque = DedupVecDeque::new();
        deque.push_back(0);
        deque.push_back(1);
        deque.push_back(2);

        assert!(deque.contains(&1));
        assert!(deque.remove(&1));
        assert!(!deque.contains(&1));
        assert_eq!(deque.iter().copied().collect::<Vec<_>>(), vec![0, 2]);
    }

    #[test]
    fn test_remove_missing() {
        let mut deque = DedupVecDeque::new();
        deque.push_back(0);
        deque.push_back(1);

        assert!(!deque.contains(&2));
        assert!(!deque.remove(&2));
        assert_eq!(deque.iter().copied().collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn test_compaction() {
        let mut deque = DedupVecDeque::new();
        deque.push_back(0);
        deque.push_back(1);
        deque.push_back(2);

        deque.push_front(1);
        deque.push_front(2);
        deque.push_front(1);
        assert_eq!(deque.num_tombstones, 3);

        deque.push_front(2);
        assert_eq!(deque.iter().copied().collect::<Vec<_>>(), vec![2, 1, 0]);
        assert_eq!(deque.num_tombstones, 0);
        assert_eq!(deque.entries.len(), deque.len());
    }
}
