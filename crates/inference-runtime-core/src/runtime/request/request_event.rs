use crate::runtime::CompletionReason;
use crate::runtime::TokenProbs;

#[derive(Debug)]
pub enum RequestEvent {
    TokenProbs(TokenProbs),
    TurnCompleted(CompletionReason),
}
