use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfidenceWeightBindings;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkMarkovWeightBindings;
use inference_executor_core::sampling::SamplerConfig;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayRecorder;
use crate::model::gather::Gather;
use crate::model::qwen::v3_x::dspark::sampling::Qwen3xDSparkMarkov;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedInput;
use crate::replay::ReplayComponent;
use crate::sampling::dspark_markov::DSparkMarkovReplayShape;
use crate::sampling::dspark_markov::DSparkProposal;
use crate::sampling::spec_probs::SpecProbsStore;

pub struct Qwen3xDSparkGatherUnembed {
    block_size: u32,
    max_requests: u32,
    gather: Gather,
    unembed: Option<Rc<Unembed>>,
    row_indices: Buffer,
}

pub struct Qwen3xDSparkSampling {
    markov: Qwen3xDSparkMarkov,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDSparkGatherUnembedArgs<'a> {
    pub num_requests: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub logits: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDSparkSamplingArgs<'a> {
    pub shape: DSparkMarkovReplayShape,
    pub logits: &'a Buffer,
    pub hidden: &'a Buffer,
    pub distribution_store: &'a SpecProbsStore,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDSparkGatherUnembedReplayKey {
    num_requests: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDSparkSamplingReplayKey {
    shape: DSparkMarkovReplayShape,
}

impl Qwen3xDSparkGatherUnembed {
    pub fn new(device: &Device, block_size: usize, max_requests: usize, hidden_dim: u32, unembed: Rc<Unembed>) -> Self {
        assert!(block_size > 0, "Qwen3 DSpark GatherUnembed requires block rows");
        assert!(max_requests > 0, "Qwen3 DSpark GatherUnembed requires requests");
        let max_rows = max_requests
            .checked_mul(block_size)
            .expect("Qwen3 DSpark gather-index capacity must fit usize");
        u32::try_from(max_rows).expect("Qwen3 DSpark gather-index capacity must fit u32");
        Self {
            block_size: block_size as u32,
            max_requests: max_requests as u32,
            gather: Gather::new(device, hidden_dim),
            unembed: Some(unembed),
            row_indices: Buffer::new_zeroed_elements(device, max_rows, Dtype::Uint32),
        }
    }

    pub fn load_weights(&mut self, unembed: Rc<Unembed>) {
        assert!(
            self.unembed.is_none(),
            "Qwen3.x DSpark unembed weights are already loaded"
        );
        self.unembed = Some(unembed);
    }

    pub fn unload_weights(&mut self) -> Rc<Unembed> {
        self.unembed
            .take()
            .expect("Qwen3.x DSpark unembed weights are not loaded")
    }

    fn unembed(&self) -> &Unembed {
        self.unembed
            .as_deref()
            .expect("Qwen3.x DSpark unembed weights must be loaded before execution")
    }

    pub fn prepare(&self, num_requests: usize) {
        assert!(num_requests > 0, "Qwen3 DSpark GatherUnembed requires requests");
        assert!(num_requests <= self.max_requests as usize);
        let block_size = self.block_size as usize;
        let num_rows = num_requests * block_size;
        let mut row_indices = Vec::with_capacity(num_rows);
        for block_offset in 0..block_size {
            for request_index in 0..num_requests {
                row_indices.push((request_index * block_size + block_offset) as u32);
            }
        }
        self.row_indices.write_typed(0, &row_indices);
    }
}

impl ReplayComponent for Qwen3xDSparkGatherUnembed {
    type Key = Qwen3xDSparkGatherUnembedReplayKey;
    type Input<'a> = Qwen3xDSparkGatherUnembedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3xDSparkGatherUnembedReplayKey {
            num_requests: input.num_requests,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        assert!(input.num_requests > 0 && input.num_requests <= self.max_requests);
        let num_rows = input.num_requests * self.block_size;
        self.gather.record(
            recorder,
            num_rows,
            ReplayU32::Fixed(num_rows),
            input.hidden_input,
            &self.row_indices,
            input.hidden_output,
        );
        let _ = <Unembed as ReplayLayer>::record(
            self.unembed(),
            recorder,
            UnembedInput {
                num_total_rows: num_rows,
                num_active_rows: ReplayU32::Fixed(num_rows),
                hidden: input.hidden_output,
                logits: input.logits,
            },
        );
    }
}

impl Qwen3xDSparkSampling {
    pub fn new(markov: Qwen3xDSparkMarkov) -> Self {
        Self { markov }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen3xDSparkMarkovWeightBindings,
        confidence_bindings: &Qwen3xDSparkConfidenceWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        self.markov.load_weights(device, store, bindings, confidence_bindings)
    }

    pub fn unload_weights(&mut self) {
        self.markov.unload_weights();
    }

    pub fn prepare(
        &self,
        req_slots: &[u32],
        anchor_token_ids: &[u32],
        anchor_positions: &[u32],
        sampler_configs: &[SamplerConfig],
        distribution_store: &SpecProbsStore,
    ) -> DSparkMarkovReplayShape {
        self.markov.prepare(
            req_slots,
            anchor_token_ids,
            anchor_positions,
            sampler_configs,
            distribution_store,
        )
    }

    pub fn add_replay_arguments(&self, shape: DSparkMarkovReplayShape, arguments: &mut ReplayArguments) {
        self.markov.add_replay_arguments(shape, arguments);
    }

    pub fn read_proposal(&self, req_slots: &[u32], distribution_store: &mut SpecProbsStore) -> DSparkProposal {
        self.markov.read_proposal(req_slots, distribution_store)
    }
}

impl ReplayComponent for Qwen3xDSparkSampling {
    type Key = Qwen3xDSparkSamplingReplayKey;
    type Input<'a> = Qwen3xDSparkSamplingArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3xDSparkSamplingReplayKey { shape: input.shape }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.markov.record(
            recorder,
            input.shape,
            input.logits,
            input.hidden,
            input.distribution_store,
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_request_major_rows_transpose_to_step_major_rows() {
        let num_requests = 2usize;
        let block_size = 3usize;
        let mut indices = Vec::new();
        for step in 0..block_size {
            for request in 0..num_requests {
                indices.push(request * block_size + step);
            }
        }
        assert_eq!(indices, [0, 3, 1, 4, 2, 5]);
    }
}
