use crate::compute::QueryTokens;
use crate::compute::SampledTokens;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::RawRequestID;

#[mockall::automock]
pub trait DevResp: Send + 'static {
    fn id(&self) -> RawRequestID;
}

#[mockall::automock]
pub trait BatchDevResp<DeviceResp>: Send + 'static
where
    DeviceResp: DevResp,
{
    fn seq(&self) -> RawComputeSlotSeq;

    fn spec_stats(&self, num_spec_tokens: usize) -> SpecStats {
        SpecStats::new(num_spec_tokens)
    }

    fn from_parts(seq: RawComputeSlotSeq, dev_resps: Vec<DeviceResp>) -> Self;
    fn into_inner(self) -> (RawComputeSlotSeq, Vec<DeviceResp>);
}

pub struct SpecStats {
    proposed_by_index: Vec<u64>,
    accepted_by_index: Vec<u64>,
}

impl SpecStats {
    pub fn new(num_spec_tokens: usize) -> Self {
        Self {
            proposed_by_index: vec![0; num_spec_tokens],
            accepted_by_index: vec![0; num_spec_tokens],
        }
    }

    pub fn record_spec_info(&mut self, num_proposed: usize, num_accepted: usize) {
        debug_assert!(num_accepted <= num_proposed);
        debug_assert!(num_proposed <= self.proposed_by_index.len());

        for count in &mut self.proposed_by_index[..num_proposed] {
            *count += 1;
        }
        for count in &mut self.accepted_by_index[..num_accepted] {
            *count += 1;
        }
    }

    pub fn accumulate(&mut self, delta: &Self) {
        debug_assert_eq!(self.proposed_by_index.len(), delta.proposed_by_index.len());
        debug_assert_eq!(self.accepted_by_index.len(), delta.accepted_by_index.len());

        for (total, delta) in self.proposed_by_index.iter_mut().zip(&delta.proposed_by_index) {
            *total += delta;
        }
        for (total, delta) in self.accepted_by_index.iter_mut().zip(&delta.accepted_by_index) {
            *total += delta;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.proposed_by_index.iter().all(|&count| count == 0)
    }

    pub fn reset(&mut self) {
        self.proposed_by_index.fill(0);
        self.accepted_by_index.fill(0);
    }

    pub fn len(&self) -> usize {
        self.proposed_by_index.len()
    }

    pub fn proposed_by_index(&self) -> &[u64] {
        &self.proposed_by_index
    }

    pub fn accepted_by_index(&self) -> &[u64] {
        &self.accepted_by_index
    }
}

pub struct DeviceResponse {
    pub req_id: RawRequestID,
    pub query_tokens: QueryTokens,
    pub sampled_tokens: SampledTokens,
}

impl DevResp for DeviceResponse {
    fn id(&self) -> RawRequestID {
        self.req_id
    }
}

pub struct BatchDeviceResponse {
    pub seq: RawComputeSlotSeq,
    pub dev_resps: Vec<DeviceResponse>,
}

impl BatchDeviceResponse {
    pub fn new<I>(seq: RawComputeSlotSeq, dev_resps: I) -> Self
    where
        I: IntoIterator<Item = DeviceResponse> + 'static,
    {
        Self {
            seq,
            dev_resps: dev_resps.into_iter().collect(),
        }
    }
}

impl BatchDevResp<DeviceResponse> for BatchDeviceResponse {
    fn seq(&self) -> RawComputeSlotSeq {
        self.seq
    }

    fn spec_stats(&self, num_spec_tokens: usize) -> SpecStats {
        let mut stats = SpecStats::new(num_spec_tokens);
        for response in &self.dev_resps {
            let (QueryTokens::Decode { spec_tokens, .. }, SampledTokens::Decode { validated_tokens, .. }) =
                (&response.query_tokens, &response.sampled_tokens)
            else {
                continue;
            };
            stats.record_spec_info(spec_tokens.len(), validated_tokens.len());
        }
        stats
    }

    fn from_parts(seq: RawComputeSlotSeq, dev_resps: Vec<DeviceResponse>) -> Self {
        Self { seq, dev_resps }
    }

    fn into_inner(self) -> (RawComputeSlotSeq, Vec<DeviceResponse>) {
        (self.seq, self.dev_resps)
    }
}
