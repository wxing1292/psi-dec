use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use inference_runtime_core::runtime::Token;

use crate::runtime::InferenceRuntime;

pub mod decode;
pub mod messages;

pub struct Inference<const N: usize, const L: usize, const P: usize> {
    runtime: Arc<InferenceRuntime<N, L, P>>,
    default_stop_sequences: Vec<Vec<Token>>,
    next_request_id: AtomicUsize,
}

impl<const N: usize, const L: usize, const P: usize> Inference<N, L, P> {
    pub fn new(runtime: Arc<InferenceRuntime<N, L, P>>, default_stop_sequences: Vec<Vec<Token>>) -> Self {
        assert!(
            default_stop_sequences.iter().all(|sequence| !sequence.is_empty()),
            "default stop sequences must not be empty"
        );
        Self {
            runtime,
            default_stop_sequences,
            next_request_id: AtomicUsize::new(1),
        }
    }
}
