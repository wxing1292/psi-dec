use async_channel::Receiver;
use async_channel::Sender;
use async_channel::TrySendError;

use crate::runtime::RawRequestID;
use crate::runtime::request::AtomicRequestStatus;
use crate::runtime::request::RequestEvent;
use crate::runtime::request::RequestStatus;

pub struct ExternalRequest {
    req_id: RawRequestID,
    req_status: AtomicRequestStatus,
    event_rx: Receiver<RequestEvent>,
    cancel_tx: Sender<RawRequestID>,
}

impl ExternalRequest {
    pub fn new(
        req_id: RawRequestID,
        req_status: AtomicRequestStatus,
        event_rx: Receiver<RequestEvent>,
        cancel_tx: Sender<RawRequestID>,
    ) -> Self {
        Self {
            req_id,
            req_status,
            event_rx,
            cancel_tx,
        }
    }

    pub fn req_id(&self) -> RawRequestID {
        self.req_id
    }

    pub fn status(&self) -> RequestStatus {
        self.req_status.load()
    }

    pub fn event_rx(&self) -> &Receiver<RequestEvent> {
        &self.event_rx
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
            match self.cancel_tx.try_send(self.req_id) {
                Ok(()) | Err(TrySendError::Closed(_)) => {},
                Err(TrySendError::Full(_)) => {
                    unreachable!("unbounded request cancellation channel cannot be full")
                },
            }
        }
    }
}
