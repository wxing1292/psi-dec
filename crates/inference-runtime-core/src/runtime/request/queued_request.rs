use std::sync::Arc;

use async_channel::Sender;

use crate::config::SamplingConfig;
use crate::runtime::RawRequestID;
use crate::runtime::RequestSlot;
use crate::runtime::RequestTokenPositions;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::TrieDecoderBlocks;
use crate::runtime::request::AtomicRequestStatus;
use crate::runtime::request::InternalRequest;
use crate::runtime::request::RequestEvent;
use crate::runtime::resource::processor::ResourceProcessors;

pub struct QueuedRequest<const N: usize, const P: usize, const L: usize, DBC>
where
    DBC: MultiLaneBlockCache<P, L>,
{
    req_id: RawRequestID,
    req_status: AtomicRequestStatus,
    decoder_blocks: TrieDecoderBlocks<N, P, L, DBC>,
    token_positions: Option<RequestTokenPositions>,
    resource_processors: Arc<ResourceProcessors>,
    event_tx: Sender<RequestEvent>,
    sampling_config: SamplingConfig,
    context_window: usize,
}

impl<const N: usize, const P: usize, const L: usize, DBC> QueuedRequest<N, P, L, DBC>
where
    DBC: MultiLaneBlockCache<P, L>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        req_id: RawRequestID,
        req_status: AtomicRequestStatus,
        decoder_blocks: TrieDecoderBlocks<N, P, L, DBC>,
        token_positions: Option<RequestTokenPositions>,
        resource_processors: Arc<ResourceProcessors>,
        event_tx: Sender<RequestEvent>,
        sampling_config: SamplingConfig,
        context_window: usize,
    ) -> Self {
        Self {
            req_id,
            req_status,
            decoder_blocks,
            token_positions,
            resource_processors,
            event_tx,
            sampling_config,
            context_window,
        }
    }

    pub fn req_id(&self) -> RawRequestID {
        self.req_id
    }
}

impl<const N: usize, const P: usize, const L: usize, DBC> From<(QueuedRequest<N, P, L, DBC>, RequestSlot)>
    for InternalRequest<N, P, L, DBC>
where
    DBC: MultiLaneBlockCache<P, L>,
{
    fn from((queued_request, req_slot): (QueuedRequest<N, P, L, DBC>, RequestSlot)) -> Self {
        let QueuedRequest {
            req_id,
            req_status,
            decoder_blocks,
            token_positions,
            resource_processors,
            event_tx,
            sampling_config,
            context_window,
        } = queued_request;
        Self::new(
            req_id,
            req_slot,
            req_status,
            decoder_blocks,
            token_positions,
            resource_processors,
            event_tx,
            sampling_config,
            context_window,
        )
    }
}
