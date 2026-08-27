use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
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

const DSPARK_GATHER_UNEMBED_NUM_ACTIVE_ROWS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3x.dspark.gather_unembed.num_active_rows");

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
    pub sample_positions: &'a Buffer,
    pub hidden: &'a Buffer,
    pub distribution_store: &'a SpecProbsStore,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDSparkGatherUnembedReplayKey {
    num_total_rows: u32,
    unembed_topology: affine_quantized::KernelKind,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDSparkSamplingReplayKey {
    num_total_requests: u32,
    num_total_sampling_inputs: u32,
    top_k: u32,
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

    pub fn replay_arguments(&self, key: &Qwen3xDSparkGatherUnembedReplayKey) -> ReplayArguments {
        ReplayArguments::new().with_u32(DSPARK_GATHER_UNEMBED_NUM_ACTIVE_ROWS, key.num_total_rows)
    }
}

impl ReplayComponent for Qwen3xDSparkGatherUnembed {
    type Key = Qwen3xDSparkGatherUnembedReplayKey;
    type Input<'a> = Qwen3xDSparkGatherUnembedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        assert!(input.num_requests > 0 && input.num_requests <= self.max_requests);
        let num_total_rows = input.num_requests * self.block_size;
        Qwen3xDSparkGatherUnembedReplayKey {
            num_total_rows,
            unembed_topology: self.unembed().replay_topology(num_total_rows),
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let num_total_rows = input.num_requests * self.block_size;
        self.gather.record(
            recorder,
            num_total_rows,
            ReplayU32::Parameter(DSPARK_GATHER_UNEMBED_NUM_ACTIVE_ROWS),
            input.hidden_input,
            &self.row_indices,
            input.hidden_output,
        );
        let _ = <Unembed as ReplayLayer>::record(
            self.unembed(),
            recorder,
            UnembedInput {
                num_total_rows,
                num_active_rows: ReplayU32::Parameter(DSPARK_GATHER_UNEMBED_NUM_ACTIVE_ROWS),
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

    pub fn prepare_static(
        &self,
        req_slots: &[u32],
        sampler_configs: &[SamplerConfig],
        distribution_store: &SpecProbsStore,
    ) -> DSparkMarkovReplayShape {
        self.markov
            .prepare_static(req_slots, sampler_configs, distribution_store)
    }

    pub fn anchor_token_ids(&self) -> &Buffer {
        self.markov.anchor_token_ids()
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
        Qwen3xDSparkSamplingReplayKey {
            num_total_requests: input.shape.num_total_requests,
            num_total_sampling_inputs: input.shape.sampling.num_total_sampling_inputs,
            top_k: input.shape.sampling.top_k,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.markov.record(
            recorder,
            input.shape,
            input.logits,
            input.sample_positions,
            input.hidden,
            input.distribution_store,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use half::bf16;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
    use inference_executor_core::mlp::dense::reference::quantized_affine_reference;

    use super::*;
    use crate::def::replay_op::MetalReplayRuntime;
    use crate::model::unembedding::UnembedConfig;
    use crate::model::unembedding::fixture_unembed;
    use crate::replay::Replay;

    const MAX_REQUESTS: u32 = 4;
    const BLOCK_SIZE: u32 = 3;
    const NUM_TOTAL_ROWS: u32 = MAX_REQUESTS * BLOCK_SIZE;
    const VOCAB_SIZE: u32 = 32;
    const HIDDEN_DIM: u32 = 32;
    const GROUP_SIZE: u32 = 32;

    #[test]
    fn test_replay_bucketing() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let (component, weights) = fixture_component(&device);
        component.prepare(MAX_REQUESTS as usize);
        let buffers = TestBuffers::new(&device);
        let input = buffers.input(MAX_REQUESTS);
        let mut replay = Replay::new("Qwen3.x DSpark GatherUnembed component test", component);
        let (recorded_key, cache_hit) = replay.record(&runtime, &input);
        assert!(!cache_hit);

        for num_active_requests in [1_u32, 4, 3, 2] {
            replay.component().prepare(num_active_requests as usize);
            assert_eq!(replay.record(&runtime, &input), (recorded_key.clone(), true));
            let num_active_rows = num_active_requests * BLOCK_SIZE;
            runtime
                .submit_replay_with_arguments(
                    replay.replay(&recorded_key),
                    &ReplayArguments::new().with_u32(DSPARK_GATHER_UNEMBED_NUM_ACTIVE_ROWS, num_active_rows),
                )
                .wait();

            let gathered = buffers.gathered_reference(num_active_requests as usize);
            let actual_hidden = buffers
                .hidden_output
                .read_typed::<u16>(0, num_active_rows as usize * HIDDEN_DIM as usize);
            let expected_hidden = gathered
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            assert_eq!(actual_hidden, expected_hidden);

            let expected_logits = quantized_affine_reference(
                QuantizedAffineReferenceShape {
                    num_rows: num_active_rows as usize,
                    output_dim: VOCAB_SIZE as usize,
                    input_dim: HIDDEN_DIM as usize,
                    group_size: GROUP_SIZE as usize,
                    bits: 8,
                },
                &gathered,
                &weights.weight,
                &weights.scales,
                &weights.biases,
            );
            let actual_logits = buffers
                .logits
                .read_typed::<u16>(0, num_active_rows as usize * VOCAB_SIZE as usize)
                .into_iter()
                .map(|bits| bf16::from_bits(bits).to_f32())
                .collect::<Vec<_>>();
            assert_close(&actual_logits, &expected_logits, 0.125);
        }
    }

    struct TestBuffers {
        hidden_input: Buffer,
        hidden_output: Buffer,
        logits: Buffer,
        hidden_values: Vec<f32>,
    }

    impl TestBuffers {
        fn new(device: &Device) -> Self {
            let hidden_values = (0..NUM_TOTAL_ROWS as usize * HIDDEN_DIM as usize)
                .map(|index| bf16::from_f32(((index * 23 + 9) % 127) as f32 * 0.015_625 - 0.875).to_f32())
                .collect::<Vec<_>>();
            let hidden_bits = hidden_values
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            Self {
                hidden_input: Buffer::from_slice(device, &hidden_bits),
                hidden_output: Buffer::new_zeroed_elements(
                    device,
                    NUM_TOTAL_ROWS as usize * HIDDEN_DIM as usize,
                    Dtype::Bfloat16,
                ),
                logits: Buffer::new_zeroed_elements(
                    device,
                    NUM_TOTAL_ROWS as usize * VOCAB_SIZE as usize,
                    Dtype::Bfloat16,
                ),
                hidden_values,
            }
        }

        fn input(&self, num_requests: u32) -> Qwen3xDSparkGatherUnembedArgs<'_> {
            Qwen3xDSparkGatherUnembedArgs {
                num_requests,
                hidden_input: &self.hidden_input,
                hidden_output: &self.hidden_output,
                logits: &self.logits,
            }
        }

        fn gathered_reference(&self, num_active_requests: usize) -> Vec<f32> {
            let mut output = Vec::with_capacity(num_active_requests * BLOCK_SIZE as usize * HIDDEN_DIM as usize);
            for block_offset in 0..BLOCK_SIZE as usize {
                for request_index in 0..num_active_requests {
                    let source_row = request_index * BLOCK_SIZE as usize + block_offset;
                    let begin = source_row * HIDDEN_DIM as usize;
                    output.extend_from_slice(&self.hidden_values[begin..begin + HIDDEN_DIM as usize]);
                }
            }
            output
        }
    }

    struct TestUnembedWeights {
        weight: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
    }

    fn fixture_component(device: &Device) -> (Qwen3xDSparkGatherUnembed, TestUnembedWeights) {
        let config = UnembedConfig {
            max_tokens: NUM_TOTAL_ROWS,
            vocab_size: VOCAB_SIZE,
            hidden_dim: HIDDEN_DIM,
            group_size: GROUP_SIZE,
            bits: 8,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let (unembed, weight, scales, biases) = fixture_unembed(device, config);
        (
            Qwen3xDSparkGatherUnembed::new(
                device,
                BLOCK_SIZE as usize,
                MAX_REQUESTS as usize,
                HIDDEN_DIM,
                Rc::new(unembed),
            ),
            TestUnembedWeights { weight, scales, biases },
        )
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
