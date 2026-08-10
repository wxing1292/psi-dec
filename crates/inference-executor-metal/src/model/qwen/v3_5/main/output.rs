use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::num_main_output_rows;
use inference_executor_core::replay::ReplayBucketPolicy;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::model::gather::Gather;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedBucketedInput;
use crate::model::unembedding::UnembedInput;
use crate::replay::ReplayComponent;

const QWEN35_GATHER_UNEMBED_NUM_ACTIVE_ROWS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3.5.gather_unembed.num_active_rows");

pub struct Qwen35GatherUnembed {
    gather: Gather,
    unembed: Rc<Unembed>,
    replay_bucket_policy: ReplayBucketPolicy,
}

#[derive(Clone, Copy)]
pub struct Qwen35GatherUnembedArgs<'a> {
    pub num_rows: u32,
    pub hidden_input: &'a Buffer,
    pub row_indices: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub logits: &'a Buffer,
}

impl Qwen35GatherUnembed {
    pub fn new(device: &Device, hidden_dim: u32, unembed: Rc<Unembed>) -> Self {
        let replay_bucket_policy =
            ReplayBucketPolicy::with_topology_boundaries(unembed.max_tokens(), &unembed.replay_topology_boundaries());
        Self {
            gather: Gather::new(device, hidden_dim),
            unembed,
            replay_bucket_policy,
        }
    }

    pub fn unembed(&self) -> Rc<Unembed> {
        Rc::clone(&self.unembed)
    }

    pub fn max_rows(&self) -> u32 {
        self.replay_bucket_policy.max_capacity()
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35GatherUnembedArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.gather.record(
            recorder,
            args.num_rows,
            args.hidden_input,
            args.row_indices,
            args.hidden_output,
        );
        <Unembed as ReplayLayer>::record(
            &self.unembed,
            recorder,
            UnembedInput {
                num_rows: args.num_rows,
                hidden: args.hidden_output,
                logits: args.logits,
            },
        )
    }

    pub fn prepare_replay(&self, num_active_rows: u32) -> (Qwen35GatherUnembedReplayKey, ReplayArguments) {
        let key = self.replay_key_for_active_rows(num_active_rows);
        let arguments = ReplayArguments::new().with_u32(QWEN35_GATHER_UNEMBED_NUM_ACTIVE_ROWS, num_active_rows);
        (key, arguments)
    }

    pub fn record_bucketed<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_rows: u32,
        args: Qwen35GatherUnembedArgs<'a>,
    ) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert_eq!(
            self.replay_bucket_policy.capacity(args.num_rows),
            num_total_rows,
            "qwen3.5 GatherUnembed replay total row count must match its selected bucket"
        );
        assert_eq!(
            self.unembed.replay_topology(args.num_rows),
            self.unembed.replay_topology(num_total_rows),
            "qwen3.5 GatherUnembed replay bucket must preserve unembed topology"
        );
        self.gather.record_bucketed(
            recorder,
            num_total_rows,
            QWEN35_GATHER_UNEMBED_NUM_ACTIVE_ROWS,
            args.hidden_input,
            args.row_indices,
            args.hidden_output,
        );
        self.unembed.record_bucketed(
            recorder,
            UnembedBucketedInput {
                num_total_rows,
                num_active_rows_key: QWEN35_GATHER_UNEMBED_NUM_ACTIVE_ROWS,
                hidden: args.hidden_output,
                logits: args.logits,
            },
        )
    }

    fn replay_key_for_active_rows(&self, num_active_rows: u32) -> Qwen35GatherUnembedReplayKey {
        let num_total_rows = self.replay_bucket_policy.capacity(num_active_rows);
        Qwen35GatherUnembedReplayKey::for_capacity(num_total_rows, self.unembed.replay_topology(num_total_rows))
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35GatherUnembedReplayKey {
    num_total_rows: u32,
    unembed_topology: Option<AffineQuantizedMatmulKernelKind>,
}

impl Qwen35GatherUnembedReplayKey {
    /// Creates a source-compatible legacy exact/manual identity.
    ///
    /// Production bucketed replay uses the component-owned row-capacity policy
    /// and records the selected unembed topology through
    /// [`Qwen35GatherUnembed::prepare_replay`].
    pub fn from_microbatch(microbatch: &Qwen35Microbatch) -> Self {
        let num_main_output_rows = num_main_output_rows(microbatch)
            .try_into()
            .expect("qwen3.5 Main output row count must fit u32");
        assert!(
            num_main_output_rows > 0,
            "qwen3.5 GatherUnembed replay requires Main output rows"
        );
        Self {
            num_total_rows: num_main_output_rows,
            unembed_topology: None,
        }
    }

    pub fn num_main_output_rows(&self) -> u32 {
        self.num_total_rows
    }

    pub fn num_total_rows(&self) -> u32 {
        self.num_total_rows
    }

    pub fn unembed_topology(&self) -> Option<AffineQuantizedMatmulKernelKind> {
        self.unembed_topology
    }

    fn for_capacity(num_total_rows: u32, unembed_topology: AffineQuantizedMatmulKernelKind) -> Self {
        Self {
            num_total_rows,
            unembed_topology: Some(unembed_topology),
        }
    }
}

impl ReplayComponent for Qwen35GatherUnembed {
    type Key = Qwen35GatherUnembedReplayKey;
    type Input<'a> = Qwen35GatherUnembedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.replay_key_for_active_rows(input.num_rows)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        self.record_bucketed(recorder, key.num_total_rows, *input);
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    use inference_backend_metal::metal::Dtype;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::attn::gdn::state::GDNStateTxn;
    use inference_executor_core::checkpoint::QuantizedTensorBindings;
    use inference_executor_core::checkpoint::SafeTensorIndex;
    use inference_executor_core::sampling::SamplerConfig;
    use safetensors::Dtype as SafeTensorDtype;
    use safetensors::tensor::View;
    use safetensors::tensor::serialize_to_file;

    use super::*;
    use crate::checkpoint::SafeTensorStore;
    use crate::def::replay_op::MetalReplayRuntime;
    use crate::model::unembedding::UnembedConfig;
    use crate::replay::Replay;

    const MAX_ROWS: u32 = 32;
    const VOCAB_SIZE: u32 = 32;
    const HIDDEN_DIM: u32 = 32;
    const GROUP_SIZE: u32 = 32;
    const OUTPUT_CANARY: u16 = 0x7fc1;

    const WEIGHT: &str = "unembed.weight";
    const SCALES: &str = "unembed.scales";
    const BIASES: &str = "unembed.biases";

    struct OwnedTensor {
        dtype: SafeTensorDtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for &OwnedTensor {
        fn dtype(&self) -> SafeTensorDtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    struct TempModelDir(PathBuf);

    impl TempModelDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "psi-qwen35-gather-unembed-test-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempModelDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn stage_policy_preserves_unembed_topology_and_prepares_one_active_argument() {
        let device = Device::system_default();
        let component = test_component(&device);
        let buffers = TestBuffers::new(&device);

        assert_eq!(component.max_rows(), MAX_ROWS);
        let (three_key, three_arguments) = component.prepare_replay(3);
        let (four_key, four_arguments) = component.prepare_replay(4);
        let (five_key, five_arguments) = component.prepare_replay(5);

        assert_eq!(three_key.num_total_rows, 4);
        assert_eq!(three_key, four_key);
        assert_eq!(five_key.num_total_rows, 6);
        assert_ne!(four_key, five_key);
        assert_eq!(component.replay_key(&buffers.input(3)), three_key);
        assert_eq!(
            three_arguments,
            ReplayArguments::new().with_u32(QWEN35_GATHER_UNEMBED_NUM_ACTIVE_ROWS, 3)
        );
        assert_eq!(
            four_arguments,
            ReplayArguments::new().with_u32(QWEN35_GATHER_UNEMBED_NUM_ACTIVE_ROWS, 4)
        );
        assert_eq!(
            five_arguments,
            ReplayArguments::new().with_u32(QWEN35_GATHER_UNEMBED_NUM_ACTIVE_ROWS, 5)
        );

        for num_active_rows in 1..=MAX_ROWS {
            let key = component.replay_key_for_active_rows(num_active_rows);
            assert_eq!(
                component.unembed.replay_topology(num_active_rows),
                component.unembed.replay_topology(key.num_total_rows),
                "num_active_rows={num_active_rows} num_total_rows={}",
                key.num_total_rows
            );
            assert_eq!(
                key.unembed_topology,
                Some(component.unembed.replay_topology(key.num_total_rows))
            );
        }
        for boundary in component.unembed.replay_topology_boundaries() {
            if boundary > 1 && boundary <= MAX_ROWS {
                assert_eq!(component.replay_bucket_policy.capacity(boundary - 1), boundary - 1);
            }
        }

        assert_ne!(
            Qwen35GatherUnembedReplayKey::for_capacity(4, AffineQuantizedMatmulKernelKind::QmvBn8Bk32),
            Qwen35GatherUnembedReplayKey::for_capacity(4, AffineQuantizedMatmulKernelKind::QmmBm8Bn32)
        );
        let legacy_key = Qwen35GatherUnembedReplayKey::from_microbatch(&one_request_microbatch(4, 3));
        assert_eq!(legacy_key.num_total_rows(), three_key.num_total_rows());
        assert_eq!(legacy_key.unembed_topology(), None);
        assert_ne!(legacy_key, three_key);
        assert_panics(|| {
            component.prepare_replay(0);
        });
        assert_panics(|| {
            component.prepare_replay(MAX_ROWS + 1);
        });
    }

    #[test]
    fn bucketed_replay_matches_exact_and_preserves_tails_across_one_two_one() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let component = test_component(&device);
        let buffers = TestBuffers::new(&device);

        buffers.fill_outputs(OUTPUT_CANARY);
        buffers.write_active_inputs(1);
        let mut exact_recorder = runtime.create_recorder();
        component.record(&mut exact_recorder, buffers.input(1));
        let exact_replay = exact_recorder.build();
        assert_eq!(exact_replay.stats().parameter_count, 0);
        runtime.submit_replay(&exact_replay).wait();
        let exact_hidden = buffers.hidden_output.read_typed::<u16>(0, HIDDEN_DIM as usize);
        let exact_logits = buffers.logits.read_typed::<u16>(0, VOCAB_SIZE as usize);

        let (_, active_one_arguments) = component.prepare_replay(1);
        let (active_two_key, active_two_arguments) = component.prepare_replay(2);
        let mut replay = Replay::new("qwen3.5 GatherUnembed test", component);
        let (recorded_key, cache_hit) = replay.record(&runtime, &buffers.input(2));
        assert!(!cache_hit);
        assert_eq!(recorded_key, active_two_key);
        assert_eq!(replay.replay(&recorded_key).stats().parameter_count, 1);

        buffers.fill_outputs(OUTPUT_CANARY);
        buffers.write_active_inputs(1);
        runtime
            .submit_replay_with_arguments(replay.replay(&recorded_key), &active_one_arguments)
            .wait();
        assert_eq!(
            buffers.hidden_output.read_typed::<u16>(0, HIDDEN_DIM as usize),
            exact_hidden
        );
        assert_eq!(buffers.logits.read_typed::<u16>(0, VOCAB_SIZE as usize), exact_logits);
        assert_output_tails(&buffers, 1);

        buffers.fill_outputs(OUTPUT_CANARY);
        buffers.write_active_inputs(2);
        runtime
            .submit_replay_with_arguments(replay.replay(&recorded_key), &active_two_arguments)
            .wait();
        assert_output_tails(&buffers, 2);

        buffers.fill_outputs(OUTPUT_CANARY);
        buffers.write_active_inputs(1);
        runtime
            .submit_replay_with_arguments(replay.replay(&recorded_key), &active_one_arguments)
            .wait();
        assert_eq!(
            buffers.hidden_output.read_typed::<u16>(0, HIDDEN_DIM as usize),
            exact_hidden
        );
        assert_eq!(buffers.logits.read_typed::<u16>(0, VOCAB_SIZE as usize), exact_logits);
        assert_output_tails(&buffers, 1);
    }

    #[test]
    fn bucketed_replay_rejects_invalid_arguments_capacities_and_short_buffers() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let component = test_component(&device);
        let buffers = TestBuffers::new(&device);
        buffers.write_active_inputs(2);
        let (key, _) = component.prepare_replay(2);
        let mut replay = Replay::new("qwen3.5 GatherUnembed invalid-input test", component);
        let (recorded_key, _) = replay.record(&runtime, &buffers.input(2));
        assert_eq!(recorded_key, key);
        let program = replay.replay(&recorded_key);

        assert_panics(|| {
            let _ = stream.submit_replay_with_arguments(program, &ReplayArguments::new());
        });
        assert_panics(|| {
            let _ = stream.submit_replay_with_arguments(
                program,
                &ReplayArguments::new().with_i32(QWEN35_GATHER_UNEMBED_NUM_ACTIVE_ROWS, 1),
            );
        });
        for num_active_rows in [0, key.num_total_rows + 1] {
            assert_panics(|| {
                let _ = stream.submit_replay_with_arguments(
                    program,
                    &ReplayArguments::new().with_u32(QWEN35_GATHER_UNEMBED_NUM_ACTIVE_ROWS, num_active_rows),
                );
            });
        }

        assert_panics(|| {
            let mut recorder = runtime.create_recorder();
            replay.component().record_bucketed(&mut recorder, 4, buffers.input(2));
        });

        let short_input = Buffer::new_zeroed(&device, 1);
        assert_panics(|| {
            let mut recorder = runtime.create_recorder();
            replay.component().record_bucketed(
                &mut recorder,
                key.num_total_rows,
                Qwen35GatherUnembedArgs {
                    hidden_input: &short_input,
                    ..buffers.input(2)
                },
            );
        });

        let short_indices = Buffer::new_zeroed_elements(&device, 1, Dtype::Uint32);
        assert_panics(|| {
            let mut recorder = runtime.create_recorder();
            replay.component().record_bucketed(
                &mut recorder,
                key.num_total_rows,
                Qwen35GatherUnembedArgs {
                    row_indices: &short_indices,
                    ..buffers.input(2)
                },
            );
        });

        let short_hidden_output = Buffer::new_zeroed_elements(&device, HIDDEN_DIM, Dtype::Bfloat16);
        assert_panics(|| {
            let mut recorder = runtime.create_recorder();
            replay.component().record_bucketed(
                &mut recorder,
                key.num_total_rows,
                Qwen35GatherUnembedArgs {
                    hidden_output: &short_hidden_output,
                    ..buffers.input(2)
                },
            );
        });

        let short_logits = Buffer::new_zeroed_elements(&device, VOCAB_SIZE, Dtype::Bfloat16);
        assert_panics(|| {
            let mut recorder = runtime.create_recorder();
            replay.component().record_bucketed(
                &mut recorder,
                key.num_total_rows,
                Qwen35GatherUnembedArgs {
                    logits: &short_logits,
                    ..buffers.input(2)
                },
            );
        });
    }

    struct TestBuffers {
        hidden_input: Buffer,
        row_indices: Buffer,
        hidden_output: Buffer,
        logits: Buffer,
    }

    impl TestBuffers {
        fn new(device: &Device) -> Self {
            Self {
                hidden_input: Buffer::new_zeroed_elements(
                    device,
                    MAX_ROWS as usize * HIDDEN_DIM as usize,
                    Dtype::Bfloat16,
                ),
                row_indices: Buffer::new_zeroed_elements(device, MAX_ROWS, Dtype::Uint32),
                hidden_output: Buffer::new_zeroed_elements(
                    device,
                    MAX_ROWS as usize * HIDDEN_DIM as usize,
                    Dtype::Bfloat16,
                ),
                logits: Buffer::new_zeroed_elements(device, MAX_ROWS as usize * VOCAB_SIZE as usize, Dtype::Bfloat16),
            }
        }

        fn input(&self, num_rows: u32) -> Qwen35GatherUnembedArgs<'_> {
            Qwen35GatherUnembedArgs {
                num_rows,
                hidden_input: &self.hidden_input,
                row_indices: &self.row_indices,
                hidden_output: &self.hidden_output,
                logits: &self.logits,
            }
        }

        fn write_active_inputs(&self, num_active_rows: u32) {
            let mut hidden = vec![OUTPUT_CANARY; MAX_ROWS as usize * HIDDEN_DIM as usize];
            hidden[..num_active_rows as usize * HIDDEN_DIM as usize].fill(0);
            self.hidden_input.write_typed(0, &hidden);
            let mut indices = vec![u32::MAX; MAX_ROWS as usize];
            for (row, index) in indices.iter_mut().take(num_active_rows as usize).enumerate() {
                *index = row as u32;
            }
            self.row_indices.write_typed(0, &indices);
        }

        fn fill_outputs(&self, value: u16) {
            for buffer in [&self.hidden_output, &self.logits] {
                buffer.write_typed(0, &vec![value; buffer.len_bytes() / size_of::<u16>()]);
            }
        }
    }

    fn assert_output_tails(buffers: &TestBuffers, num_active_rows: usize) {
        for (buffer, num_values_per_row) in [
            (&buffers.hidden_output, HIDDEN_DIM as usize),
            (&buffers.logits, VOCAB_SIZE as usize),
        ] {
            let values = buffer.read_typed::<u16>(0, buffer.len_bytes() / size_of::<u16>());
            let active_values = num_active_rows * num_values_per_row;
            assert!(values[..active_values].iter().all(|&value| value == 0));
            assert!(values[active_values..].iter().all(|&value| value == OUTPUT_CANARY));
        }
    }

    fn test_component(device: &Device) -> Qwen35GatherUnembed {
        const FILE_NAME: &str = "model.safetensors";
        let weight_bytes = VOCAB_SIZE as usize * HIDDEN_DIM as usize;
        let affine_elements = VOCAB_SIZE as usize * HIDDEN_DIM as usize / GROUP_SIZE as usize;
        let tensors = HashMap::from([
            (
                WEIGHT.to_string(),
                OwnedTensor {
                    dtype: SafeTensorDtype::U32,
                    shape: vec![weight_bytes / size_of::<u32>()],
                    data: vec![0; weight_bytes],
                },
            ),
            (
                SCALES.to_string(),
                OwnedTensor {
                    dtype: SafeTensorDtype::BF16,
                    shape: vec![affine_elements],
                    data: vec![0; affine_elements * size_of::<u16>()],
                },
            ),
            (
                BIASES.to_string(),
                OwnedTensor {
                    dtype: SafeTensorDtype::BF16,
                    shape: vec![affine_elements],
                    data: vec![0; affine_elements * size_of::<u16>()],
                },
            ),
        ]);
        let model_dir = TempModelDir::new();
        serialize_to_file(
            tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
            None,
            &model_dir.0.join(FILE_NAME),
        )
        .unwrap();
        let index = SafeTensorIndex::new(
            tensors
                .keys()
                .map(|name| (name.clone(), FILE_NAME.to_string()))
                .collect(),
        )
        .unwrap();
        let mut store = SafeTensorStore::new(&model_dir.0, index);
        let config = UnembedConfig {
            max_tokens: MAX_ROWS,
            vocab_size: VOCAB_SIZE,
            hidden_dim: HIDDEN_DIM,
            group_size: GROUP_SIZE,
            bits: 8,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let mut unembed = Unembed::new(device, config);
        unembed
            .load_weights(
                device,
                &mut store,
                QuantizedTensorBindings {
                    weight: WEIGHT.to_string(),
                    scales: SCALES.to_string(),
                    biases: BIASES.to_string(),
                },
            )
            .unwrap();
        Qwen35GatherUnembed::new(device, config.hidden_dim, Rc::new(unembed))
    }

    fn one_request_microbatch(num_tokens: u32, num_spec_tokens: u32) -> Qwen35Microbatch {
        let num_output_rows = num_spec_tokens + 1;
        Qwen35Microbatch::new(
            vec![0],
            vec![0],
            vec![0],
            (0..num_tokens).map(|token| token as i32).collect(),
            vec![0, num_tokens],
            vec![GDNStateTxn::new(0, num_tokens, num_spec_tokens)],
            vec![Vec::new()],
            vec![SamplerConfig::default()],
            (0..num_tokens)
                .map(|token_offset| token_offset + num_output_rows >= num_tokens)
                .collect(),
        )
    }

    fn assert_panics(f: impl FnOnce()) {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
    }
}
