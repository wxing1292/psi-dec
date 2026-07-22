use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPKernels;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::Operator;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::dense::DenseMLPReplayShape;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::def::replay_op::ReplayOp;
use inference_executor_metal::mlp::dense::backend::DenseMLP;
use inference_executor_metal::mlp::dense::backend::DenseMLPMetalConfig;
use inference_executor_metal::mlp::dense::backend::DenseMLPReplayInput;
use inference_executor_metal::mlp::dense::scratch::DenseMLPScratchBindings;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

const SHARD: &str = "model-00001-of-00003.safetensors";
const LAYER0_MLP: &str = "language_model.model.layers.0.mlp";
const HIDDEN_DIM: u32 = 5120;
const INTERMEDIATE_DIM: u32 = 17_408;
const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

fn main() {
    let args = Args::parse();

    let device = Device::system_default();
    let weights = RealDenseMLPWeights::load(&device, &args.model_dir);
    for num_tokens in args.tokens {
        let fixture = RealDenseMLPFixture::new(&device, num_tokens, &weights);
        for case in &args.cases {
            let replay = fixture.replay(*case);
            let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || {
                fixture.submit_replay(&replay)
            });
            print_perf(
                fixture.stream.backend_name(),
                case.key(),
                num_tokens,
                args.iters,
                &samples,
                &replay,
            );
        }
    }
}

struct Args {
    model_dir: PathBuf,
    tokens: Vec<u32>,
    cases: Vec<DenseMLPBenchCase>,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            tokens: vec![1, 2, 4, 8, 10, 16, 32, 64],
            cases: vec![DenseMLPBenchCase::FullAuto],
            iters: 50,
            warmup_iters: 20,
            runs: 3,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut iter, &arg)),
                "--tokens" => args.tokens = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--cases" => args.cases = parse_cases(&next_arg(&mut iter, &arg)),
                "--iters" => args.iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                },
                _ => panic!("unknown argument {arg:?}; pass --help for usage"),
            }
        }
        assert!(!args.tokens.is_empty(), "--tokens must include at least one value");
        assert!(!args.cases.is_empty(), "--cases must include at least one case");
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args
    }
}

#[derive(Clone, Copy, Debug)]
enum DenseMLPBenchCase {
    FullAuto,
    FullQmv,
    FullQmm,
    GateUpAuto,
    GateUpQmv,
    GateUpQmm,
    Activation,
    DownAuto,
    DownQmv,
    DownQmm,
}

impl DenseMLPBenchCase {
    fn parse(value: &str) -> Self {
        match value {
            "full_auto" => Self::FullAuto,
            "full_qmv" => Self::FullQmv,
            "full_qmm" => Self::FullQmm,
            "gate_up_auto" => Self::GateUpAuto,
            "gate_up_qmv" => Self::GateUpQmv,
            "gate_up_qmm" => Self::GateUpQmm,
            "activation" => Self::Activation,
            "down_auto" => Self::DownAuto,
            "down_qmv" => Self::DownQmv,
            "down_qmm" => Self::DownQmm,
            _ => panic!("unknown dense MLP bench case {value:?}"),
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::FullAuto => "full_auto",
            Self::FullQmv => "full_qmv",
            Self::FullQmm => "full_qmm",
            Self::GateUpAuto => "gate_up_auto",
            Self::GateUpQmv => "gate_up_qmv",
            Self::GateUpQmm => "gate_up_qmm",
            Self::Activation => "activation",
            Self::DownAuto => "down_auto",
            Self::DownQmv => "down_qmv",
            Self::DownQmm => "down_qmm",
        }
    }
}

struct RealDenseMLPFixture<'a> {
    stream: Stream,
    shape: QuantizedDenseMLPShape,
    backend: DenseMLP,
    kernels: QuantizedDenseMLPKernels,
    hidden_state: Buffer,
    output: Buffer,
    gate_up_proj: Buffer,
    activation: Buffer,
    gate_up_qmv: AffineQuantizedMatmulKernel,
    gate_up_qmm: AffineQuantizedMatmulKernel,
    down_qmv: AffineQuantizedMatmulKernel,
    down_qmm: AffineQuantizedMatmulKernel,
    weights: &'a RealDenseMLPWeights,
}

impl<'a> RealDenseMLPFixture<'a> {
    fn new(device: &Device, num_tokens: u32, weights: &'a RealDenseMLPWeights) -> Self {
        let config = dense_config();
        let backend = DenseMLP::new(
            device,
            DenseMLPCore {
                model_layer_index: 0,
                hidden_dim: HIDDEN_DIM as usize,
                intermediate_dim: INTERMEDIATE_DIM as usize,
            },
            DenseMLPMetalConfig {
                group_size: GROUP_SIZE,
                bits: BITS,
                dtype: Dtype::Bfloat16,
            },
        );
        let shape = QuantizedDenseMLPShape { num_tokens };
        let hidden_state = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM as usize));
        Self {
            stream: Stream::new(device),
            shape,
            backend,
            kernels: QuantizedDenseMLPKernels::new(device, config),
            hidden_state,
            output: Buffer::new_zeroed(device, config.down_shape(shape).output_bytes()),
            gate_up_proj: Buffer::from_slice(
                device,
                &hidden_fixture(num_tokens as usize, (INTERMEDIATE_DIM * 2) as usize),
            ),
            activation: Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, INTERMEDIATE_DIM as usize)),
            gate_up_qmv: AffineQuantizedMatmulKernel::new(device, gate_up_affine_shape(1)),
            gate_up_qmm: AffineQuantizedMatmulKernel::new(
                device,
                gate_up_affine_shape(qmv_batch_limit(HIDDEN_DIM, INTERMEDIATE_DIM * 2)),
            ),
            down_qmv: AffineQuantizedMatmulKernel::new(device, down_affine_shape(1)),
            down_qmm: AffineQuantizedMatmulKernel::new(
                device,
                down_affine_shape(qmv_batch_limit(INTERMEDIATE_DIM, HIDDEN_DIM)),
            ),
            weights,
        }
    }

    fn replay(&self, case: DenseMLPBenchCase) -> ReplayProgram {
        match case {
            DenseMLPBenchCase::FullAuto => self.forward_replay(),
            DenseMLPBenchCase::FullQmv => self.forced_full_replay(DenseAffinePolicy::Qmv),
            DenseMLPBenchCase::FullQmm => self.forced_full_replay(DenseAffinePolicy::Qmm),
            DenseMLPBenchCase::GateUpAuto => {
                build_single_invocation_replay(
                    &self.stream,
                    self.kernels.invoke_gate_up(
                        self.shape,
                        &self.hidden_state,
                        &self.gate_up_proj,
                        self.weights.as_borrowed(),
                    ),
                )
            },
            DenseMLPBenchCase::GateUpQmv => self.forced_gate_up_replay(DenseAffinePolicy::Qmv),
            DenseMLPBenchCase::GateUpQmm => self.forced_gate_up_replay(DenseAffinePolicy::Qmm),
            DenseMLPBenchCase::Activation => {
                build_single_invocation_replay(
                    &self.stream,
                    self.kernels
                        .invoke_activation(self.shape, &self.gate_up_proj, &self.activation),
                )
            },
            DenseMLPBenchCase::DownAuto => {
                build_single_invocation_replay(
                    &self.stream,
                    self.kernels
                        .invoke_down(self.shape, &self.activation, &self.output, self.weights.as_borrowed()),
                )
            },
            DenseMLPBenchCase::DownQmv => self.forced_down_replay(DenseAffinePolicy::Qmv),
            DenseMLPBenchCase::DownQmm => self.forced_down_replay(DenseAffinePolicy::Qmm),
        }
    }

    fn forward_replay(&self) -> ReplayProgram {
        let mut builder = MetalReplayRuntime::new(&self.stream).create_recorder();
        let _ = <DenseMLP as ReplayLayer>::record(
            &self.backend,
            &mut builder,
            DenseMLPReplayInput {
                shape: DenseMLPReplayShape {
                    num_tokens: self.shape.num_tokens,
                },
                hidden_state: &self.hidden_state,
                next_hidden_state: &self.output,
                scratch: DenseMLPScratchBindings {
                    gate_up_proj: &self.gate_up_proj,
                    activation: &self.activation,
                },
                weights: self.weights.as_borrowed(),
            },
        );
        builder.build()
    }

    fn submit_replay(&self, replay: &ReplayProgram) {
        MetalReplayRuntime::new(&self.stream).submit_replay(replay).wait();
    }

    fn forced_full_replay(&self, policy: DenseAffinePolicy) -> ReplayProgram {
        let mut builder = MetalReplayRuntime::new(&self.stream).create_recorder();
        self.record_forced_gate_up(&mut builder, policy);
        builder.record_with_barrier_before(ReplayOp::opaque(self.kernels.invoke_activation(
            self.shape,
            &self.gate_up_proj,
            &self.activation,
        )));
        self.record_forced_down(&mut builder, policy);
        builder.build()
    }

    fn forced_gate_up_replay(&self, policy: DenseAffinePolicy) -> ReplayProgram {
        let mut builder = MetalReplayRuntime::new(&self.stream).create_recorder();
        self.record_forced_gate_up(&mut builder, policy);
        builder.build()
    }

    fn forced_down_replay(&self, policy: DenseAffinePolicy) -> ReplayProgram {
        let mut builder = MetalReplayRuntime::new(&self.stream).create_recorder();
        self.record_forced_down(&mut builder, policy);
        builder.build()
    }

    fn record_forced_gate_up<'b>(
        &'b self,
        builder: &mut impl Recorder<'b, Operator = ReplayOp<'b>>,
        policy: DenseAffinePolicy,
    ) {
        let kernel = match policy {
            DenseAffinePolicy::Qmv => &self.gate_up_qmv,
            DenseAffinePolicy::Qmm => &self.gate_up_qmm,
        };
        let weights = self.weights.as_borrowed();
        builder.record_with_barrier_before(ReplayOp::opaque(kernel.invoke_with_shape(
            gate_up_affine_shape(self.shape.num_tokens),
            &self.gate_up_proj,
            0,
            &self.hidden_state,
            0,
            weights.gate_up_weight,
            0,
            weights.gate_up_scales,
            0,
            weights.gate_up_biases,
            0,
        )));
    }

    fn record_forced_down<'b>(
        &'b self,
        builder: &mut impl Recorder<'b, Operator = ReplayOp<'b>>,
        policy: DenseAffinePolicy,
    ) {
        let kernel = match policy {
            DenseAffinePolicy::Qmv => &self.down_qmv,
            DenseAffinePolicy::Qmm => &self.down_qmm,
        };
        let weights = self.weights.as_borrowed();
        builder.record_with_barrier_before(ReplayOp::opaque(kernel.invoke_with_shape(
            down_affine_shape(self.shape.num_tokens),
            &self.output,
            0,
            &self.activation,
            0,
            weights.down_weight,
            0,
            weights.down_scales,
            0,
            weights.down_biases,
            0,
        )));
    }
}

#[derive(Clone, Copy)]
enum DenseAffinePolicy {
    Qmv,
    Qmm,
}

struct RealDenseMLPWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl RealDenseMLPWeights {
    fn load(device: &Device, model_dir: &Path) -> Self {
        let shard_path = model_dir.join(SHARD);
        let mapped = MappedFile::open(&shard_path);
        let tensors = SafeTensors::deserialize(mapped.as_bytes()).unwrap_or_else(|err| {
            panic!(
                "unable to deserialize safetensors shard {}: {err:?}",
                shard_path.display()
            )
        });
        let gate_weight = tensor_bytes(&tensors, &tensor_name("gate_proj", "weight"), safetensors::Dtype::U32);
        let up_weight = tensor_bytes(&tensors, &tensor_name("up_proj", "weight"), safetensors::Dtype::U32);
        let gate_scales = tensor_bytes(&tensors, &tensor_name("gate_proj", "scales"), safetensors::Dtype::BF16);
        let up_scales = tensor_bytes(&tensors, &tensor_name("up_proj", "scales"), safetensors::Dtype::BF16);
        let gate_biases = tensor_bytes(&tensors, &tensor_name("gate_proj", "biases"), safetensors::Dtype::BF16);
        let up_biases = tensor_bytes(&tensors, &tensor_name("up_proj", "biases"), safetensors::Dtype::BF16);
        let down_weight = tensor_bytes(&tensors, &tensor_name("down_proj", "weight"), safetensors::Dtype::U32);
        let down_scales = tensor_bytes(&tensors, &tensor_name("down_proj", "scales"), safetensors::Dtype::BF16);
        let down_biases = tensor_bytes(&tensors, &tensor_name("down_proj", "biases"), safetensors::Dtype::BF16);

        let gate_up_weight = concat_bytes(&gate_weight, &up_weight);
        let gate_up_scales = concat_bytes(&gate_scales, &up_scales);
        let gate_up_biases = concat_bytes(&gate_biases, &up_biases);

        let config = dense_config();
        let shape = QuantizedDenseMLPShape { num_tokens: 1 };
        let gate_up_shape = config.gate_up_shape(shape);
        assert_eq!(gate_up_weight.len(), gate_up_shape.weight_bytes());
        assert_eq!(gate_up_scales.len(), gate_up_shape.affine_param_bytes());
        assert_eq!(gate_up_biases.len(), gate_up_shape.affine_param_bytes());
        assert_eq!(down_weight.len(), config.down_shape(shape).weight_bytes());
        assert_eq!(down_scales.len(), config.down_shape(shape).affine_param_bytes());
        assert_eq!(down_biases.len(), config.down_shape(shape).affine_param_bytes());

        Self {
            gate_up_weight: Buffer::from_slice(device, &gate_up_weight),
            gate_up_scales: Buffer::from_slice(device, &gate_up_scales),
            gate_up_biases: Buffer::from_slice(device, &gate_up_biases),
            down_weight: Buffer::from_slice(device, &down_weight),
            down_scales: Buffer::from_slice(device, &down_scales),
            down_biases: Buffer::from_slice(device, &down_biases),
        }
    }

    fn as_borrowed(&self) -> QuantizedDenseMLPWeights<'_> {
        QuantizedDenseMLPWeights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

struct MappedFile {
    ptr: *mut libc::c_void,
    len: usize,
}

impl MappedFile {
    fn open(path: &Path) -> Self {
        let file = File::open(path).unwrap_or_else(|err| panic!("unable to open {}: {err}", path.display()));
        let len = file
            .metadata()
            .unwrap_or_else(|err| panic!("unable to stat {}: {err}", path.display()))
            .len() as usize;
        assert!(len > 0, "safetensors shard must not be empty");
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            panic!("unable to mmap {}: {}", path.display(), std::io::Error::last_os_error());
        }
        unsafe {
            let _ = libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
        Self { ptr, len }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}

fn tensor_name(proj: &str, part: &str) -> String {
    format!("{LAYER0_MLP}.{proj}.{part}")
}

fn tensor_bytes(tensors: &SafeTensors<'_>, name: &str, dtype: safetensors::Dtype) -> Vec<u8> {
    let view = tensors
        .tensor(name)
        .unwrap_or_else(|err| panic!("missing safetensor {name}: {err:?}"));
    assert_eq!(view.dtype(), dtype, "unexpected dtype for tensor {name}");
    validate_tensor_shape(name, &view);
    view.data().to_vec()
}

fn validate_tensor_shape(name: &str, view: &TensorView<'_>) {
    let shape = view.shape();
    if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
        assert_eq!(shape, &[INTERMEDIATE_DIM as usize, packed_k_words(HIDDEN_DIM)]);
    } else if name.ends_with("down_proj.weight") {
        assert_eq!(shape, &[HIDDEN_DIM as usize, packed_k_words(INTERMEDIATE_DIM)]);
    } else if name.ends_with("gate_proj.scales")
        || name.ends_with("gate_proj.biases")
        || name.ends_with("up_proj.scales")
        || name.ends_with("up_proj.biases")
    {
        assert_eq!(shape, &[INTERMEDIATE_DIM as usize, groups(HIDDEN_DIM)]);
    } else if name.ends_with("down_proj.scales") || name.ends_with("down_proj.biases") {
        assert_eq!(shape, &[HIDDEN_DIM as usize, groups(INTERMEDIATE_DIM)]);
    } else {
        panic!("unexpected dense MLP tensor name {name}");
    }
}

fn concat_bytes(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend_from_slice(left);
    out.extend_from_slice(right);
    out
}

fn dense_config() -> QuantizedDenseMLPConfig {
    QuantizedDenseMLPConfig {
        hidden_dim: HIDDEN_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        group_size: GROUP_SIZE,
        bits: BITS,
        dtype: Dtype::Bfloat16,
    }
}

fn gate_up_affine_shape(num_tokens: u32) -> AffineQuantizedMatmulShape {
    dense_affine_shape(num_tokens, INTERMEDIATE_DIM * 2, HIDDEN_DIM)
}

fn down_affine_shape(num_tokens: u32) -> AffineQuantizedMatmulShape {
    dense_affine_shape(num_tokens, HIDDEN_DIM, INTERMEDIATE_DIM)
}

fn dense_affine_shape(num_tokens: u32, n: u32, k: u32) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape {
        m: num_tokens.try_into().expect("dense MLP affine m must fit i32"),
        n: n.try_into().expect("dense MLP affine n must fit i32"),
        k: k.try_into().expect("dense MLP affine k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("dense MLP group size must fit i32"),
        bits: BITS.try_into().expect("dense MLP bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        affine_dtype: Dtype::Bfloat16,
    }
}

fn qmv_batch_limit(input_dim: u32, output_dim: u32) -> u32 {
    if input_dim <= 2048 && output_dim <= 2048 {
        18
    } else if input_dim <= 4096 && output_dim <= 4096 {
        12
    } else {
        10
    }
}

fn hidden_fixture(tokens: usize, hidden_dim: usize) -> Vec<u16> {
    (0..tokens * hidden_dim)
        .map(|index| bf16::from_f32(((index % 23) as f32 - 11.0) * 0.03125).to_bits())
        .collect()
}

fn packed_k_words(k: u32) -> usize {
    (k as usize * BITS as usize) / 32
}

fn groups(k: u32) -> usize {
    (k / GROUP_SIZE) as usize
}

fn build_single_invocation_replay<I>(stream: &Stream, invocation: I) -> ReplayProgram
where
    I: Operator,
{
    let mut builder = MetalReplayRuntime::new(stream).create_recorder();
    builder.record(ReplayOp::opaque(invocation));
    builder.build()
}

fn next_arg(iter: &mut impl Iterator<Item = String>, flag: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{flag} requires a value; pass --help for usage"))
}

fn parse_u32_list(value: &str, flag: &str) -> Vec<u32> {
    parse_list(value)
        .into_iter()
        .map(|part| {
            part.parse::<u32>()
                .unwrap_or_else(|err| panic!("invalid {flag} entry {part:?}: {err}"))
        })
        .collect()
}

fn parse_cases(value: &str) -> Vec<DenseMLPBenchCase> {
    parse_list(value).into_iter().map(DenseMLPBenchCase::parse).collect()
}

fn parse_list(value: &str) -> Vec<&str> {
    let values = value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    assert!(!values.is_empty(), "list argument must contain at least one value");
    values
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {flag} value {value:?}: {err}"))
}

fn print_help() {
    println!(
        "\
dense_mlp real-weight replay bench

Options:
  --model-dir PATH
  --tokens 1,2,4,8,10,16,32,64
  --cases full_auto,full_qmv,full_qmm,gate_up_auto,gate_up_qmv,gate_up_qmm,activation,down_auto,down_qmv,down_qmm
  --iters N
  --warmup-iters N
  --runs N
"
    );
}

fn measure_runs(runs: usize, warmup_iters: usize, iters: usize, mut run: impl FnMut()) -> Vec<f64> {
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        for _ in 0..warmup_iters {
            run();
        }
        let mut duration = Duration::ZERO;
        for _ in 0..iters {
            let start = Instant::now();
            run();
            duration += start.elapsed();
        }
        samples.push(duration.as_secs_f64() * 1_000_000.0 / iters as f64);
    }
    samples
}

fn print_perf(
    backend: &str,
    implementation: &str,
    num_tokens: u32,
    iters: usize,
    samples: &[f64],
    replay: &ReplayProgram,
) {
    let median_us = median(samples);
    let stats = replay.stats();
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=dense-mlp-real backend={backend} impl={implementation} tokens={num_tokens} iters={iters} \
         runs={} median_us={median_us:.3} command_count={} retained_buffers={} retained_pipelines={} \
         constant_bytes={} samples_us=[{sample_text}]",
        samples.len(),
        stats.command_count,
        stats.retained_buffer_count,
        stats.retained_pipeline_count,
        stats.parameter_buffer_bytes
    );
}

fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}
