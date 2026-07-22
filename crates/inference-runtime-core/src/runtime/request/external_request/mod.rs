use async_channel::Receiver;

use crate::runtime::RawRequestID;
use crate::runtime::request::AtomicRequestStatus;
use crate::runtime::request::RequestStatus;
use crate::runtime::request::TokenProbs;

pub struct ExternalRequest {
    req_id: RawRequestID,
    req_status: AtomicRequestStatus,
    token_prob_rx: Receiver<TokenProbs>,
}

impl ExternalRequest {
    pub fn new(req_id: RawRequestID, req_status: AtomicRequestStatus, token_prob_rx: Receiver<TokenProbs>) -> Self {
        Self {
            req_id,
            req_status,
            token_prob_rx,
        }
    }

    pub fn req_id(&self) -> RawRequestID {
        self.req_id
    }

    pub fn status(&self) -> RequestStatus {
        self.req_status.load()
    }

    pub fn token_prob_rx(&self) -> &Receiver<TokenProbs> {
        &self.token_prob_rx
    }

    delegate::delegate! {
        to self.req_status {
            pub fn store_cancelled(&self) -> bool;
        }
    }
}

impl Drop for ExternalRequest {
    fn drop(&mut self) {
        if self.req_status.store_cancelled() {
            tracing::debug!(
                target: "inference-runtime-core::request",
                phase = "request.cancelled",
                request_id = self.req_id,
                "request cancelled"
            );
        }
    }
}
