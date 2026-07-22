use crate::runtime::Token;

mod block_metadata;
pub use block_metadata::BlockMetadata;

mod immutable_block;
pub use immutable_block::ImmutableBlock;

mod semi_immutable_block;
pub use semi_immutable_block::SemiImmutableBlock;

mod mutable_block;
pub use mutable_block::MutableBlock;

pub trait DecoderBlock {
    fn cached_tokens(&self) -> &[Token];
    fn scheduled_tokens(&self) -> &[Token];
    fn ready_tokens(&self) -> &[Token];
    fn total_tokens(&self) -> &[Token];

    fn ready_token_slots(&self) -> usize;

    fn cache_tokens(&mut self, tokens: &[Token]);
    fn schedule_tokens(&mut self, num_tokens: usize) -> &[Token];
    fn unschedule_tokens(&mut self, tokens: &[Token]);
}
