use futures_lite::future::Boxed;

use crate::compute::QueryTokens;
use crate::compute::SampledTokens;

mod trie;
pub use trie::TrieDecoderBlocks;

mod consumption;
pub use consumption::TokenConsumption;
pub use consumption::token_consumption;

mod util;
pub use util::cache_tokens;
pub use util::pop_front_queued_tokens;
pub use util::push_front_queued_tokens;
pub use util::push_tokens;
pub use util::schedule_tokens;
pub use util::unschedule_tokens;

pub enum InitBlockOnceResult {
    ResourceLimitExceeded,
    Await { wait: Boxed<()> },
    Success { ready_token_slots: usize },
}

pub enum UninitBlockOnceResult {
    Success { cached_token_slots: usize },
}

#[mockall::automock]
pub trait DecoderBlocks {
    fn ready_token_slots(&self) -> usize;
    fn init_block_once(&mut self) -> InitBlockOnceResult;
    fn uninit_block_once(&mut self) -> UninitBlockOnceResult;

    /// Reserve and materialize the next query transaction.
    fn prepare(&mut self, token_budget: usize) -> Option<QueryTokens>;

    /// Roll back a prepared transaction without committing its scheduled tokens.
    fn cancel(&mut self, query_tokens: QueryTokens);

    /// Finalize a prepared transaction and materialize the resulting cached tokens / blocks.
    fn commit(&mut self, query_tokens: QueryTokens, sampled_tokens: SampledTokens);
}
