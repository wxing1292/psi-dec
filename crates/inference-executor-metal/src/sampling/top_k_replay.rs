use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_executor_core::sampling::TopKSamplingShape;

use crate::def::replay_op::ReplayRecorder;
use crate::replay::ReplayComponent;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingInputs;
use crate::sampling::top_k_sampling::TopKSamplingOutput;
use crate::sampling::top_k_sampling::TopKSamplingSparseDistributionOutput;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TopKSamplingReplayKey {
    pub num_sampling_input_capacity: u32,
    pub top_k: u32,
}

pub struct Sampling {
    pub sampler: Rc<TopKSampling>,
}

pub struct DraftSampling {
    pub sampler: Rc<TopKSampling>,
}

#[derive(Clone, Copy)]
pub struct SamplingInput<'a> {
    pub shape: TopKSamplingShape,
    pub logits: &'a Buffer,
    pub output: TopKSamplingOutput<'a>,
}

#[derive(Clone, Copy)]
pub struct DraftSamplingInput<'a> {
    pub shape: TopKSamplingShape,
    pub logits: &'a Buffer,
    pub output: TopKSamplingOutput<'a>,
    pub sparse: TopKSamplingSparseDistributionOutput<'a>,
}

fn sampling_key(shape: TopKSamplingShape) -> TopKSamplingReplayKey {
    TopKSamplingReplayKey {
        num_sampling_input_capacity: shape.num_total_sampling_inputs,
        top_k: shape.top_k,
    }
}

impl ReplayComponent for Sampling {
    type Key = TopKSamplingReplayKey;
    type Input<'a> = SamplingInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        sampling_key(input.shape)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.sampler.record_bf16(
            recorder,
            input.shape,
            TopKSamplingInputs {
                logits: input.logits,
                logits_offset_bytes: 0,
            },
            input.output,
        );
    }
}

impl ReplayComponent for DraftSampling {
    type Key = TopKSamplingReplayKey;
    type Input<'a> = DraftSamplingInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        sampling_key(input.shape)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.sampler.record_bf16_with_sparse_distribution(
            recorder,
            input.shape,
            TopKSamplingInputs {
                logits: input.logits,
                logits_offset_bytes: 0,
            },
            input.output,
            input.sparse,
        );
    }
}
