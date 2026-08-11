use crate::compute::BatchDeviceRequest;
use crate::compute::BatchDeviceResponse;

pub enum ReplayableModelExecutorRequest<BatchRequest = BatchDeviceRequest> {
    Batch(BatchRequest),
    Start,
    Stop,
}

pub enum ReplayableModelExecutorResponse<BatchResponse = BatchDeviceResponse> {
    Batch(BatchResponse),
    Started,
    Stopped,
}
