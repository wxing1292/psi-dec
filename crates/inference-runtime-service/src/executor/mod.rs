use std::sync::Arc;

use inference_executor_core::model::EncoderExecutorLifecycle;

mod replayable_decoder_model_event_loop;
pub use replayable_decoder_model_event_loop::ReplayableDecoderModelEventLoop;

pub struct ReplayableModelExecutors<M> {
    decoder: M,
    encoders: Vec<Arc<dyn EncoderExecutorLifecycle>>,
}

impl<M> ReplayableModelExecutors<M> {
    pub fn new(decoder: M, encoders: Vec<Arc<dyn EncoderExecutorLifecycle>>) -> Self {
        Self { decoder, encoders }
    }

    pub fn decoder(&self) -> &M {
        &self.decoder
    }

    pub fn into_parts(self) -> (M, Vec<Arc<dyn EncoderExecutorLifecycle>>) {
        (self.decoder, self.encoders)
    }
}
