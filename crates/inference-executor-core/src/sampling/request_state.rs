use inference_runtime_core::runtime::RawRequestSlot;
use rand::RngExt;
use rand::rngs::SmallRng;

pub struct RequestSamplingState {
    rng: SmallRng,
    seeds: Vec<Option<u32>>,
}

impl RequestSamplingState {
    pub fn new(num_req_slots: usize) -> Self {
        Self::from_rng(num_req_slots, rand::make_rng())
    }

    fn from_rng(num_req_slots: usize, rng: SmallRng) -> Self {
        assert!(num_req_slots > 0, "request sampling state requires request slots");
        Self {
            rng,
            seeds: vec![None; num_req_slots],
        }
    }

    pub fn resolve(&mut self, req_slot: RawRequestSlot, seed: Option<u32>) -> u32 {
        let req_slot = req_slot as usize;
        let resolved_seed = self
            .seeds
            .get_mut(req_slot)
            .unwrap_or_else(|| panic!("request sampling slot {req_slot} exceeds capacity"));
        match *resolved_seed {
            Some(resolved_seed) => {
                if let Some(seed) = seed {
                    assert_eq!(seed, resolved_seed, "request sampling seed changed while slot is live");
                }
                resolved_seed
            },
            None => {
                let seed = seed.unwrap_or_else(|| self.rng.random());
                *resolved_seed = Some(seed);
                seed
            },
        }
    }

    pub fn reset(&mut self, req_slots: &[RawRequestSlot]) {
        for &req_slot in req_slots {
            let req_slot = req_slot as usize;
            let seed = self
                .seeds
                .get_mut(req_slot)
                .unwrap_or_else(|| panic!("request sampling slot {req_slot} exceeds capacity"));
            *seed = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    fn fixture_state() -> RequestSamplingState {
        RequestSamplingState::from_rng(3, SmallRng::seed_from_u64(7))
    }

    #[test]
    fn test_resolve() {
        let mut state = fixture_state();
        let first = state.resolve(0, None);
        let second = state.resolve(1, None);

        assert_ne!(first, second);
        assert_eq!(state.resolve(0, None), first);
        assert_eq!(state.resolve(1, Some(second)), second);
        assert_eq!(state.resolve(2, Some(42)), 42);
    }

    #[test]
    fn test_reset() {
        let mut state = fixture_state();
        let first = state.resolve(0, None);
        state.reset(&[0]);

        assert_ne!(state.resolve(0, None), first);
    }
}
