//! Persistent cross-module Qwen3.5 MTP hidden-state cache.

use std::ops::Range;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_runtime_core::runtime::RawRequestSlot;

/// Token identities at the next Main input index, from the last completed wave.
/// Only Decode retains the hidden rows needed to replace a speculative tail.
/// Both nonempty variants store K tokens. The last draft has no cached KV slot.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum Qwen35MTPCacheState {
    #[default]
    Empty,
    Prefill {
        token_index: u32,
        token_ids: Vec<i32>,
    },
    Decode {
        token_index: u32,
        token_ids: Vec<i32>,
    },
}

impl Qwen35MTPCacheState {
    pub fn requires_tail_repair(&self, input_token_index: u32, input_token_ids: &[i32]) -> bool {
        let (token_index, token_ids) = match self {
            Self::Empty => return false,
            Self::Prefill { token_index, token_ids } | Self::Decode { token_index, token_ids } => {
                (*token_index, token_ids)
            },
        };
        // Runtime must retain the anchor and its position across request-local waves.
        debug_assert_eq!(input_token_index, token_index, "MTP cached anchor index changed");
        debug_assert_eq!(input_token_ids[0], token_ids[0], "MTP cached anchor token changed");
        let matches = token_ids
            .iter()
            .zip(input_token_ids)
            .all(|(cached, input)| cached == input);
        match self {
            Self::Prefill { .. } => {
                debug_assert!(matches, "MTP canonical Prefill lookahead changed");
                false
            },
            Self::Decode { .. } => !matches,
            Self::Empty => unreachable!(),
        }
    }
}

/// Maps one request slot and non-final logical MTP module to its contiguous physical hidden-state cache rows.
fn mtp_hidden_state_cache_row_range(
    max_request_slots: usize,
    req_slot: RawRequestSlot,
    module_index: usize,
) -> Range<u32> {
    debug_assert!(max_request_slots > 0);
    let num_module_rows = module_index + 1;
    let req_slot = req_slot as usize;
    debug_assert!(req_slot < max_request_slots);
    let module_base = max_request_slots * module_index * num_module_rows / 2;
    let req_base = module_base + req_slot * num_module_rows;
    let req_end = req_base + num_module_rows;
    debug_assert!(u32::try_from(req_end).is_ok());
    req_base as u32..req_end as u32
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MTPHiddenStateCacheLayout {
    max_request_slots: usize,
    num_spec_tokens: usize,
    hidden_dim: usize,
    num_rows: usize,
}

impl Qwen35MTPHiddenStateCacheLayout {
    pub fn new(max_request_slots: usize, num_spec_tokens: usize, hidden_dim: usize) -> Self {
        assert!(
            max_request_slots > 0,
            "qwen3.5 MTP hidden-state cache requires request slots"
        );
        assert!(
            num_spec_tokens > 0,
            "qwen3.5 MTP hidden-state cache requires proposal steps"
        );
        assert!(
            hidden_dim > 0,
            "qwen3.5 MTP hidden-state cache requires a hidden dimension"
        );
        let num_rows_per_request = num_spec_tokens
            .checked_mul(num_spec_tokens - 1)
            .and_then(|value| value.checked_div(2))
            .expect("qwen3.5 MTP hidden-state cache row count must fit usize");
        let num_rows = max_request_slots
            .checked_mul(num_rows_per_request)
            .expect("qwen3.5 MTP hidden-state cache row count must fit usize");
        u32::try_from(num_rows).expect("qwen3.5 MTP hidden-state cache row indices must fit u32");
        num_rows
            .checked_mul(hidden_dim)
            .and_then(|value| value.checked_mul(Dtype::Bfloat16.item_size()))
            .expect("qwen3.5 MTP hidden-state cache byte length must fit usize");
        Self {
            max_request_slots,
            num_spec_tokens,
            hidden_dim,
            num_rows,
        }
    }

    pub fn num_rows(self) -> usize {
        self.num_rows
    }

    pub fn num_elements(self) -> usize {
        self.num_rows * self.hidden_dim
    }

    pub fn row_range(self, req_slot: RawRequestSlot, module_index: usize) -> Range<u32> {
        debug_assert!(module_index + 1 < self.num_spec_tokens);
        mtp_hidden_state_cache_row_range(self.max_request_slots, req_slot, module_index)
    }
}

pub struct Qwen35MTPHiddenStateCache {
    layout: Qwen35MTPHiddenStateCacheLayout,
    hidden_states: Option<Buffer>,
    request_states: Vec<Qwen35MTPCacheState>,
}

impl Qwen35MTPHiddenStateCache {
    pub fn new(device: &Device, max_request_slots: usize, num_spec_tokens: usize, hidden_dim: usize) -> Self {
        let layout = Qwen35MTPHiddenStateCacheLayout::new(max_request_slots, num_spec_tokens, hidden_dim);
        Self {
            hidden_states: (layout.num_rows() > 0)
                .then(|| Buffer::new_zeroed_elements(device, layout.num_elements(), Dtype::Bfloat16)),
            layout,
            request_states: vec![Qwen35MTPCacheState::Empty; max_request_slots],
        }
    }

    pub fn layout(&self) -> Qwen35MTPHiddenStateCacheLayout {
        self.layout
    }

    pub fn hidden_states(&self) -> Option<&Buffer> {
        self.hidden_states.as_ref()
    }

    pub fn request_state(&self, req_slot: RawRequestSlot) -> &Qwen35MTPCacheState {
        &self.request_states[req_slot as usize]
    }

    pub fn row_range(&self, req_slot: RawRequestSlot, module_index: usize) -> Range<u32> {
        self.layout.row_range(req_slot, module_index)
    }

    pub fn set_request_state(&mut self, req_slot: RawRequestSlot, state: Qwen35MTPCacheState) {
        match &state {
            Qwen35MTPCacheState::Empty => {},
            Qwen35MTPCacheState::Prefill { token_ids, .. } | Qwen35MTPCacheState::Decode { token_ids, .. } => {
                debug_assert_eq!(token_ids.len(), self.layout.num_spec_tokens);
            },
        }
        self.request_states[req_slot as usize] = state;
    }

    pub fn reset_req_slots(&mut self, req_slots: &[RawRequestSlot]) {
        for &req_slot in req_slots {
            self.request_states[req_slot as usize] = Qwen35MTPCacheState::Empty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_tail_compares_only_cached_overlapping_tokens() {
        let state = Qwen35MTPCacheState::Decode {
            token_index: 3,
            token_ids: vec![10, 11, 12],
        };
        for input in [&[10][..], &[10, 11], &[10, 11, 12], &[10, 11, 12, 99, 100]] {
            assert!(!state.requires_tail_repair(3, input));
        }
        for input in [&[10, 90][..], &[10, 11, 90], &[10, 90, 12, 13]] {
            assert!(state.requires_tail_repair(3, input));
        }
        assert!(!Qwen35MTPCacheState::Empty.requires_tail_repair(50, &[90, 91, 92]));
        let prefill = Qwen35MTPCacheState::Prefill {
            token_index: 3,
            token_ids: vec![10, 11, 12],
        };
        assert!(!prefill.requires_tail_repair(3, &[10, 11, 12, 99]));
        let one_module = Qwen35MTPCacheState::Decode {
            token_index: 3,
            token_ids: vec![10],
        };
        assert!(!one_module.requires_tail_repair(3, &[10, 99]));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_tail_protocol_rejects_changed_anchor_or_canonical_lookahead() {
        for state in [
            Qwen35MTPCacheState::Prefill {
                token_index: 3,
                token_ids: vec![10, 11, 12],
            },
            Qwen35MTPCacheState::Decode {
                token_index: 3,
                token_ids: vec![10, 11, 12],
            },
        ] {
            assert!(std::panic::catch_unwind(|| state.requires_tail_repair(4, &[10, 11])).is_err());
            assert!(std::panic::catch_unwind(|| state.requires_tail_repair(3, &[20, 11])).is_err());
        }
        let state = Qwen35MTPCacheState::Prefill {
            token_index: 3,
            token_ids: vec![10, 11, 12],
        };
        assert!(std::panic::catch_unwind(|| state.requires_tail_repair(3, &[10, 99])).is_err());
    }

    #[test]
    fn test_request_state_survives_until_replaced_or_slot_reset() {
        let device = Device::system_default();
        let mut cache = Qwen35MTPHiddenStateCache::new(&device, 2, 3, 8);
        let decode = Qwen35MTPCacheState::Decode {
            token_index: 3,
            token_ids: vec![10, 11, 12],
        };
        cache.set_request_state(0, decode.clone());
        cache.set_request_state(1, decode.clone());
        assert!(cache.request_state(0).requires_tail_repair(3, &[10, 99]));
        assert_eq!(cache.request_state(0), &decode);
        let prefill = Qwen35MTPCacheState::Prefill {
            token_index: 4,
            token_ids: vec![99, 100, 101],
        };
        cache.set_request_state(0, prefill.clone());
        assert_eq!(cache.request_state(0), &prefill);
        cache.reset_req_slots(&[0]);
        assert_eq!(cache.request_state(0), &Qwen35MTPCacheState::Empty);
        assert_eq!(cache.request_state(1), &decode);
    }

    #[test]
    fn test_layout_is_module_major() {
        let layout = Qwen35MTPHiddenStateCacheLayout::new(2, 3, 8);
        assert_eq!(layout.num_rows(), 6);
        assert_eq!(layout.num_elements(), 48);
        assert_eq!(layout.row_range(0, 0), 0..1);
        assert_eq!(layout.row_range(1, 0), 1..2);
        assert_eq!(layout.row_range(0, 1), 2..4);
        assert_eq!(layout.row_range(1, 1), 4..6);
    }

    #[test]
    fn test_layout_rows_are_disjoint_and_fill_capacity() {
        let max_request_slots = 3;
        let num_spec_tokens = 5;
        let layout = Qwen35MTPHiddenStateCacheLayout::new(max_request_slots, num_spec_tokens, 8);
        let mut rows = Vec::new();
        for module_index in 0..num_spec_tokens - 1 {
            for req_slot in 0..max_request_slots as RawRequestSlot {
                rows.extend(layout.row_range(req_slot, module_index));
            }
        }
        rows.sort_unstable();
        assert_eq!(rows, (0..layout.num_rows() as u32).collect::<Vec<_>>());
        assert_eq!(
            layout.num_rows(),
            max_request_slots * num_spec_tokens * (num_spec_tokens - 1) / 2
        );
    }

    #[test]
    fn test_one_module_has_no_hidden_state_buffer() {
        let layout = Qwen35MTPHiddenStateCacheLayout::new(2, 1, 8);
        assert_eq!(layout.num_rows(), 0);
        assert_eq!(layout.num_elements(), 0);
    }
}
