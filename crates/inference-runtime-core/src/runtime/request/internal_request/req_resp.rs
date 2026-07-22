use super::stop_sequence::StopSequences;
use crate::compute::DeviceRequest;
use crate::compute::DeviceResponse;
use crate::compute::SampledTokens;
use crate::runtime::RawRequestID;
use crate::runtime::decoder::trie_cache::DecoderBlocks;
use crate::runtime::decoder::trie_cache::InitBlockOnceResult;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::token_consumption;
use crate::runtime::request::InternalRequest;
use crate::runtime::request::internal_request::StopSequenceMatch;
use crate::runtime::scheduler::CancelResult;
use crate::runtime::scheduler::CommitResult;
use crate::runtime::scheduler::PrepareResult;
use crate::runtime::scheduler::UserRequest;

impl<const N: usize, const P: usize, const L: usize, DBC> UserRequest<DeviceRequest, DeviceResponse>
    for InternalRequest<N, P, L, DBC>
where
    DBC: MultiLaneBlockCache<P, L>,
{
    fn id(&self) -> RawRequestID {
        self.req_id
    }

    fn request_estimate(&self) -> usize {
        1
    }

    fn token_estimate(&self, token_budget: usize) -> usize {
        let num_ready_tokens = self.decoder_blocks.num_ready_tokens();
        let num_queued_tokens = self.decoder_blocks.num_queued_tokens();
        let num_spec_tokens = self.decoder_blocks.num_spec_tokens();
        let token_consumption =
            token_consumption::<L>(token_budget, num_ready_tokens, num_queued_tokens, num_spec_tokens);
        token_consumption.token_consumption()
    }

    fn prepare(&mut self, token_budget: usize) -> PrepareResult<DeviceRequest> {
        if self.status().is_terminal() {
            return PrepareResult::Terminal;
        }

        let mut ready_token_slots = self.decoder_blocks.ready_token_slots();
        while ready_token_slots < token_budget {
            ready_token_slots = match self.decoder_blocks.init_block_once() {
                InitBlockOnceResult::ResourceLimitExceeded => {
                    return PrepareResult::ResourceLimitExceeded;
                },
                InitBlockOnceResult::Await { wait } => {
                    return PrepareResult::Await { wait };
                },
                InitBlockOnceResult::Success { ready_token_slots } => ready_token_slots,
            }
        }

        match self.decoder_blocks.prepare(token_budget) {
            Some(decoder_query_tokens) => {
                let decoder_sync_blocks = self.decoder_blocks.prepare_blocks();
                let dev_req = DeviceRequest::new(
                    self.req_id(),
                    self.req_slot(),
                    decoder_query_tokens,
                    decoder_sync_blocks,
                    self.sampling_config().clone(),
                );
                PrepareResult::Continue(dev_req)
            },
            None => PrepareResult::Pending,
        }
    }

    fn cancel(&mut self, dev_req: DeviceRequest) -> CancelResult {
        let DeviceRequest {
            req_id,
            req_slot,
            decoder_query_tokens,
            decoder_sync_blocks,
            ..
        } = dev_req;
        assert_eq!(self.req_id(), req_id, "cancel response request ID mismatch");
        assert_eq!(self.req_slot(), req_slot, "cancel response request slot mismatch");

        self.decoder_blocks.cancel_blocks(decoder_sync_blocks);
        self.decoder_blocks.cancel(decoder_query_tokens);

        if self.status().is_terminal() {
            CancelResult::Terminal
        } else {
            CancelResult::Continue
        }
    }

    fn commit(&mut self, dev_resp: DeviceResponse) -> CommitResult {
        let DeviceResponse {
            req_id,
            query_tokens,
            sampled_tokens,
        } = dev_resp;
        assert_eq!(self.req_id, req_id, "device response request ID mismatch");
        let remaining_visible_tokens = self
            .sampling_config()
            .max_sampled_tokens
            .saturating_sub(self.decoder_blocks.num_sampled_tokens());
        let stop_match = self.match_stop_sequence(&sampled_tokens);
        let mut token_probs = stop_match.visible_token_probs(&sampled_tokens);
        if let Some(token_probs) = &mut token_probs {
            token_probs.tokens.truncate(remaining_visible_tokens);
            token_probs.probs.truncate(remaining_visible_tokens);
        }
        self.decoder_blocks.commit(query_tokens, sampled_tokens);
        if let Some(token_probs) = token_probs
            && !token_probs.tokens.is_empty()
        {
            self.send_token_probs(token_probs);
        }
        if stop_match.matched() || self.decoder_blocks.num_sampled_tokens() >= self.sampling_config().max_sampled_tokens
        {
            self.store_completed();
        }

        if self.status().is_terminal() {
            CommitResult::Terminal
        } else {
            CommitResult::Continue
        }
    }
}

impl<const N: usize, const P: usize, const L: usize, DBC> InternalRequest<N, P, L, DBC>
where
    DBC: MultiLaneBlockCache<P, L>,
{
    fn match_stop_sequence(&self, sampled_tokens: &SampledTokens) -> StopSequenceMatch {
        let stop_sequences = self.sampling_config().stop_sequences.as_slice();
        let stop_sequences = StopSequences::new(stop_sequences);
        stop_sequences.match_decode(self.decoder_blocks.sampled_tokens_rev(), sampled_tokens)
    }
}
