use std::cell::Cell;
use std::mem::size_of;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::replay::ReplayBucketPolicy;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GDNReplayBucketPolicy {
    requests: ReplayBucketPolicy,
    tokens: ReplayBucketPolicy,
}

impl GDNReplayBucketPolicy {
    pub fn new(max_requests: u32, max_tokens: u32, token_topology_boundaries: &[u32]) -> Self {
        Self {
            requests: ReplayBucketPolicy::new(max_requests),
            tokens: ReplayBucketPolicy::with_topology_boundaries(max_tokens, token_topology_boundaries),
        }
    }

    pub fn max_requests(&self) -> u32 {
        self.requests.max_capacity()
    }

    pub fn max_tokens(&self) -> u32 {
        self.tokens.max_capacity()
    }

    fn capacities(&self, num_reqs: u32, num_tokens: u32) -> (u32, u32) {
        (self.requests.capacity(num_reqs), self.tokens.capacity(num_tokens))
    }

    fn request_capacity(&self, num_reqs: u32) -> u32 {
        self.requests.capacity(num_reqs)
    }
}

/// Capacity-sized GPU metadata and replay shape refreshed during GDN state
/// preparation and shared by all GDN layers.
pub struct GDNMetadataBuffers {
    cu_tokens: Buffer,
    src_state_slots: Buffer,
    flat_materialized_state_slots: Buffer,
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
            flat_materialized_state_slots: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            replay_shape: Cell::new(None),
        }
    }

    pub fn cu_tokens(&self) -> &Buffer {
        &self.cu_tokens
    }

    pub fn src_state_slots(&self) -> &Buffer {
        &self.src_state_slots
    }

    pub fn flat_materialized_state_slots(&self) -> &Buffer {
        &self.flat_materialized_state_slots
    }

    pub fn update(
        &self,
        cu_tokens: &[u32],
        src_state_slots: &[u32],
        flat_materialized_state_slots: &[u32],
    ) -> GDNReplayShape {
        self.update_with_policy(cu_tokens, src_state_slots, flat_materialized_state_slots, None, None)
    }

    pub fn update_bucketed(
        &self,
        cu_tokens: &[u32],
        src_state_slots: &[u32],
        flat_materialized_state_slots: &[u32],
        policy: &GDNReplayBucketPolicy,
    ) -> GDNReplayShape {
        self.validate_bucket_policy(policy);
        self.update_with_policy(
            cu_tokens,
            src_state_slots,
            flat_materialized_state_slots,
            Some(policy),
            None,
        )
    }

    pub fn update_bucketed_with_token_capacity(
        &self,
        cu_tokens: &[u32],
        src_state_slots: &[u32],
        flat_materialized_state_slots: &[u32],
        policy: &GDNReplayBucketPolicy,
        total_tokens: u32,
    ) -> GDNReplayShape {
        self.validate_bucket_policy(policy);
        self.update_with_policy(
            cu_tokens,
            src_state_slots,
            flat_materialized_state_slots,
            Some(policy),
            Some(total_tokens),
        )
    }

    fn validate_bucket_policy(&self, policy: &GDNReplayBucketPolicy) {
        assert_eq!(
            policy.max_requests() as usize + 1,
            self.cu_tokens.len_bytes() / size_of::<u32>(),
            "GDN replay request policy must match metadata capacity"
        );
        assert_eq!(
            policy.max_requests() as usize,
            self.src_state_slots.len_bytes() / size_of::<u32>(),
            "GDN replay request policy must match state-slot metadata capacity"
        );
        assert_eq!(
            policy.max_tokens() as usize,
            self.flat_materialized_state_slots.len_bytes() / size_of::<u32>(),
            "GDN replay token policy must match metadata capacity"
        );
    }

    fn update_with_policy(
        &self,
        cu_tokens: &[u32],
        src_state_slots: &[u32],
        flat_materialized_state_slots: &[u32],
        policy: Option<&GDNReplayBucketPolicy>,
        selected_total_tokens: Option<u32>,
    ) -> GDNReplayShape {
        assert!(!src_state_slots.is_empty(), "GDN batch metadata requires requests");
        assert_eq!(cu_tokens.len(), src_state_slots.len() + 1);
        assert!(src_state_slots.len() < self.cu_tokens.len_bytes() / size_of::<u32>());
        assert_eq!(cu_tokens[0], 0, "GDN batch cu_tokens must start at zero");
        assert!(
            cu_tokens.windows(2).all(|window| window[0] < window[1]),
            "GDN batch cu_tokens must assign at least one token to every request"
        );
        let num_tokens = cu_tokens[cu_tokens.len() - 1] as usize;
        assert_eq!(flat_materialized_state_slots.len(), num_tokens);
        assert!(num_tokens <= self.flat_materialized_state_slots.len_bytes() / size_of::<u32>());
        let num_reqs = src_state_slots
            .len()
            .try_into()
            .expect("GDN batch metadata count must fit u32");
        let num_tokens = num_tokens.try_into().expect("GDN batch token count must fit u32");
        let (total_reqs, total_tokens) = match (policy, selected_total_tokens) {
            (Some(policy), Some(total_tokens)) => {
                assert!(
                    total_tokens >= num_tokens,
                    "GDN caller-owned token capacity must contain all active tokens"
                );
                assert!(
                    total_tokens <= policy.max_tokens(),
                    "GDN caller-owned token capacity must not exceed the metadata capacity"
                );
                (policy.request_capacity(num_reqs), total_tokens)
            },
            (Some(policy), None) => policy.capacities(num_reqs, num_tokens),
            (None, None) => (num_reqs, num_tokens),
            (None, Some(_)) => panic!("GDN caller-owned token capacity requires a replay bucket policy"),
        };
        let replay_shape = GDNReplayShape::new(num_reqs, total_reqs, num_tokens, total_tokens);

        self.cu_tokens.write_typed(0, cu_tokens);
        self.src_state_slots.write_typed(0, src_state_slots);
        self.flat_materialized_state_slots
            .write_typed(0, flat_materialized_state_slots);
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
    use inference_executor_core::attn::GDNReplayShape;

    use super::GDNMetadataBuffers;
    use super::GDNReplayBucketPolicy;

    #[test]
    fn test_update_accepts_request_capacity() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 2, 2);

        let shape = metadata.update(&[0, 1, 2], &[3, 4], &[7, 8]);

        assert_eq!(shape, GDNReplayShape::new(2, 2, 2, 2));
        assert_eq!(shape, metadata.replay_shape());
        assert_eq!(metadata.src_state_slots().read_typed::<u32>(0, 2), vec![3, 4]);
        assert_eq!(
            metadata.flat_materialized_state_slots().read_typed::<u32>(0, 2),
            vec![7, 8]
        );
    }

    #[test]
    fn test_bucketed_update_uses_independent_request_and_token_capacities() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 6, 8);
        let policy = GDNReplayBucketPolicy::new(6, 8, &[]);

        let shape = metadata.update_bucketed(&[0, 1, 2, 3, 4, 7], &[10, 11, 12, 13, 14], &[u32::MAX; 7], &policy);

        assert_eq!(shape, GDNReplayShape::new(5, 6, 7, 8));
        assert_eq!(shape, metadata.replay_shape());
    }

    #[test]
    fn test_bucketed_update_uses_caller_owned_token_capacity_without_rebucketing() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 6, 8);
        let policy = GDNReplayBucketPolicy::new(6, 8, &[]);

        let shape = metadata.update_bucketed_with_token_capacity(
            &[0, 1, 2, 3, 4, 7],
            &[10, 11, 12, 13, 14],
            &[u32::MAX; 7],
            &policy,
            7,
        );

        assert_eq!(shape, GDNReplayShape::new(5, 6, 7, 7));
        assert_eq!(shape, metadata.replay_shape());
    }

    #[test]
    #[should_panic(expected = "GDN caller-owned token capacity must contain all active tokens")]
    fn test_bucketed_update_rejects_small_caller_owned_token_capacity() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 1, 8);
        let policy = GDNReplayBucketPolicy::new(1, 8, &[]);

        metadata.update_bucketed_with_token_capacity(&[0, 7], &[10], &[u32::MAX; 7], &policy, 6);
    }

    #[test]
    #[should_panic(expected = "GDN caller-owned token capacity must not exceed the metadata capacity")]
    fn test_bucketed_update_rejects_large_caller_owned_token_capacity() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 1, 8);
        let policy = GDNReplayBucketPolicy::new(1, 8, &[]);

        metadata.update_bucketed_with_token_capacity(&[0, 1], &[10], &[u32::MAX], &policy, 9);
    }

    #[test]
    #[should_panic(expected = "GDN batch cu_tokens must start at zero")]
    fn test_update_rejects_nonzero_cumulative_start() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 1, 2);

        metadata.update(&[1, 2], &[3], &[5, 6]);
    }

    #[test]
    #[should_panic(expected = "GDN batch cu_tokens must assign at least one token to every request")]
    fn test_update_rejects_empty_request_window() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 2, 2);

        metadata.update(&[0, 1, 1], &[3, 4], &[7]);
    }
}
