use std::cell::Cell;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GDNReplayShape;

/// Capacity-sized GPU metadata and replay shape refreshed during GDN state
/// preparation and shared by all GDN layers.
pub struct GDNMetadataBuffers {
    cu_tokens: Buffer,
    src_state_slots: Buffer,
    dst_state_slots: Buffer,
    flat_candidate_state_slots: Buffer,
    replay_shape: Cell<Option<GDNReplayShape>>,
}

impl GDNMetadataBuffers {
    pub fn new(device: &Device, max_requests: usize, max_tokens: usize) -> Self {
        assert!(max_requests > 0, "GDN batch metadata requires requests");
        assert!(max_tokens > 0, "GDN batch metadata requires tokens");
        assert!(u32::try_from(max_requests).is_ok(), "GDN request capacity must fit u32");
        assert!(u32::try_from(max_tokens).is_ok(), "GDN token capacity must fit u32");
        Self {
            cu_tokens: Buffer::new_zeroed_elements(
                device,
                max_requests
                    .checked_add(1)
                    .expect("GDN cumulative-token capacity must fit usize"),
                Dtype::Uint32,
            ),
            src_state_slots: Buffer::new_zeroed_elements(device, max_requests, Dtype::Uint32),
            dst_state_slots: Buffer::new_zeroed_elements(device, max_requests, Dtype::Uint32),
            flat_candidate_state_slots: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            replay_shape: Cell::new(None),
        }
    }

    pub fn cu_tokens(&self) -> &Buffer {
        &self.cu_tokens
    }

    pub fn src_state_slots(&self) -> &Buffer {
        &self.src_state_slots
    }

    pub fn dst_state_slots(&self) -> &Buffer {
        &self.dst_state_slots
    }

    pub fn flat_candidate_state_slots(&self) -> &Buffer {
        &self.flat_candidate_state_slots
    }

    pub fn update(
        &self,
        cu_tokens: &[u32],
        src_state_slots: &[u32],
        dst_state_slots: &[u32],
        flat_candidate_state_slots: &[u32],
    ) -> GDNReplayShape {
        assert!(!src_state_slots.is_empty(), "GDN batch metadata requires requests");
        assert_eq!(cu_tokens.len(), src_state_slots.len() + 1);
        assert_eq!(src_state_slots.len(), dst_state_slots.len());
        assert!(src_state_slots.len() < self.cu_tokens.len_bytes() / size_of::<u32>());
        assert_eq!(cu_tokens[0], 0, "GDN batch cu_tokens must start at zero");
        assert!(
            cu_tokens.windows(2).all(|window| window[0] < window[1]),
            "GDN batch cu_tokens must assign at least one token to every request"
        );
        let num_tokens = cu_tokens[cu_tokens.len() - 1] as usize;
        assert_eq!(flat_candidate_state_slots.len(), num_tokens);
        assert!(num_tokens <= self.flat_candidate_state_slots.len_bytes() / size_of::<u32>());
        let replay_shape = GDNReplayShape {
            num_reqs: src_state_slots
                .len()
                .try_into()
                .expect("GDN batch metadata count must fit u32"),
            num_tokens: num_tokens.try_into().expect("GDN batch token count must fit u32"),
        };
        replay_shape.validate();

        self.cu_tokens.write_typed(0, cu_tokens);
        self.src_state_slots.write_typed(0, src_state_slots);
        self.dst_state_slots.write_typed(0, dst_state_slots);
        self.flat_candidate_state_slots
            .write_typed(0, flat_candidate_state_slots);
        self.replay_shape.set(Some(replay_shape));
        replay_shape
    }

    pub fn replay_shape(&self) -> GDNReplayShape {
        self.replay_shape
            .get()
            .expect("GDN batch metadata must be updated before recording")
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;

    use super::GDNMetadataBuffers;

    #[test]
    fn test_update_accepts_request_capacity() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 2, 2);

        let shape = metadata.update(&[0, 1, 2], &[3, 4], &[5, 6], &[7, 8]);

        assert_eq!(shape, metadata.replay_shape());
        assert_eq!(metadata.src_state_slots().read_typed::<u32>(0, 2), vec![3, 4]);
        assert_eq!(metadata.dst_state_slots().read_typed::<u32>(0, 2), vec![5, 6]);
        assert_eq!(
            metadata.flat_candidate_state_slots().read_typed::<u32>(0, 2),
            vec![7, 8]
        );
    }

    #[test]
    #[should_panic(expected = "GDN batch cu_tokens must start at zero")]
    fn test_update_rejects_nonzero_cumulative_start() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 1, 2);

        metadata.update(&[1, 2], &[3], &[4], &[5, 6]);
    }

    #[test]
    #[should_panic(expected = "GDN batch cu_tokens must assign at least one token to every request")]
    fn test_update_rejects_empty_request_window() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 2, 2);

        metadata.update(&[0, 1, 1], &[3, 4], &[5, 6], &[7]);
    }
}
