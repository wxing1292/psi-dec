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

    fn from_parts(seq: RawComputeSlotSeq, dev_resps: Vec<DeviceResp>) -> Self;
    fn into_inner(self) -> (RawComputeSlotSeq, Vec<DeviceResp>);
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

    fn from_parts(seq: RawComputeSlotSeq, dev_resps: Vec<DeviceResponse>) -> Self {
        Self { seq, dev_resps }
    }

    fn into_inner(self) -> (RawComputeSlotSeq, Vec<DeviceResponse>) {
        (self.seq, self.dev_resps)
    }
}
