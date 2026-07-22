/// Backend replay recorder contract.
///
/// A recorder accepts backend-specific operators and builds a backend-specific
/// replay artifact. A barrier is an attribute of its consumer operator, not a
/// standalone operation or a property left behind by its producer.
pub trait Recorder<'a> {
    type Operator;
    type Replay;

    fn record(&mut self, operator: Self::Operator);
    fn record_with_barrier_before(&mut self, operator: Self::Operator);
    fn build(self) -> Self::Replay;
}
