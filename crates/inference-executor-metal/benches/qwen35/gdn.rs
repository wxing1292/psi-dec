use std::fs::File;
use std::mem::size_of;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::GDNCoreBuffers;
use inference_backend_metal::components::GDNCoreConfig;
use inference_backend_metal::components::GDNCoreKernels;
use inference_backend_metal::components::GDNCoreShape;
use inference_backend_metal::components::GDNProjectionSplitBuffers;
use inference_backend_metal::components::GDNProjectionSplitKernel;
use inference_backend_metal::components::GDNProjectionSplitShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::Operator;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_metal::attn::gdn::backend::GDN;
use inference_executor_metal::attn::gdn::backend::GDNInput;
use inference_executor_metal::attn::gdn::backend::GDNLayerStateBindings;
use inference_executor_metal::attn::gdn::backend::GDNMetalConfig;
use inference_executor_metal::attn::gdn::backend::GDNWeights;
use inference_executor_metal::attn::gdn::batch_metadata::GDNMetadataBuffers;
use inference_executor_metal::attn::gdn::scratch::GDNScratchBindings;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::def::replay_op::ReplayOp;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

const HIDDEN_DIM: usize = 2048;
const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

const TOKENS_PER_PAGE: u32 = 16;
const KV_TOKEN_TILE_SIZE: u32 = 256;
const NUM_THREADS_PER_THREADBLOCK: u32 = 256;
const Q_HEAD_TILE_SIZE_CAP: u32 = 8;
const Q_TOKEN_TILE_SIZE: u32 = 8;
const TILED_KV_TOKEN_TILE_SIZE: u32 = 16;

const GDN_LAYER: usize = 0;
const GDN_QK_HEADS: usize = 16;
const GDN_V_HEADS: usize = 32;
const GDN_QK_HEAD_DIM: usize = 128;
const GDN_V_HEAD_DIM: usize = 128;
const GDN_QK_DIM: usize = GDN_QK_HEADS * GDN_QK_HEAD_DIM;
const GDN_V_DIM: usize = GDN_V_HEADS * GDN_V_HEAD_DIM;
const GDN_CONV_DIM: usize = GDN_QK_DIM * 2 + GDN_V_DIM;
const GDN_QKVABZ_DIM: usize = GDN_CONV_DIM + GDN_V_HEADS * 2 + GDN_V_DIM;
const GDN_CONV_KERNEL_SIZE: usize = 4;
const GDN_EPS: f32 = 1.0e-6;

const GDN_SHARD: &str = "model-00001-of-00004.safetensors";

#[path = "gdn/fixture.rs"]
mod fixture;

fn main() {
    fixture::run(Args::parse());
}
struct Args {
    model_dir: PathBuf,
    tokens: Vec<u32>,
    contexts: Vec<u32>,
    num_reqs: Vec<u32>,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
    subcomponents: bool,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            tokens: vec![1, 2, 4, 8, 16, 32, 64],
            contexts: Vec::new(),
            num_reqs: vec![1],
            iters: 50,
            warmup_iters: 20,
            runs: 1,
            subcomponents: false,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--tokens" => args.tokens = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--contexts" => args.contexts = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--num-reqs" => args.num_reqs = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--subcomponents" => args.subcomponents = true,
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(
            !args.tokens.is_empty(),
            "--tokens must include at least one token count"
        );
        assert!(
            !args.num_reqs.is_empty(),
            "--num-reqs must include at least one request count"
        );
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        for &num_reqs in &args.num_reqs {
            assert!(num_reqs > 0, "--num-reqs entries must be positive");
        }
        args
    }
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{name} requires a value; pass --help for usage"))
}

fn parse_string_list(value: &str, name: &str) -> Vec<String> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert!(
        !values.is_empty(),
        "{name} must contain at least one comma-separated value"
    );
    values
}

fn parse_u32_list(value: &str, name: &str) -> Vec<u32> {
    parse_string_list(value, name)
        .into_iter()
        .map(|part| parse_u32(&part, name))
        .collect()
}

fn parse_u32(value: &str, name: &str) -> u32 {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {name} value {value:?}: {err}"))
}

fn parse_usize(value: &str, name: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {name} value {value:?}: {err}"))
}

fn print_help_and_exit() -> ! {
    println!(
        r#"qwen35_gdn bench
--model-dir PATH
--tokens 1,2,4,8,16,32,64
--contexts 0,128
--num-reqs 1,2,4
--subcomponents
--iters N
--warmup-iters N
--runs N"#
    );
    std::process::exit(0);
}

fn hidden_fixture(tokens: usize, hidden_dim: usize) -> Vec<u16> {
    (0..tokens * hidden_dim)
        .map(|index| bf16::from_f32(((index % 23) as f32 - 11.0) * 0.03125).to_bits())
        .collect()
}

fn valid_num_reqs(num_tokens: u32, num_reqs: u32) -> bool {
    num_reqs > 0 && num_reqs <= num_tokens
}

fn request_token_counts(num_tokens: u32, num_reqs: u32) -> Vec<u32> {
    assert!(
        valid_num_reqs(num_tokens, num_reqs),
        "request token counts require 1 <= num_reqs <= tokens"
    );
    let base = num_tokens / num_reqs;
    let remainder = num_tokens % num_reqs;
    (0..num_reqs)
        .map(|req_index| base + u32::from(req_index < remainder))
        .collect()
}

fn cu_tokens(num_tokens_per_req: &[u32]) -> Vec<i32> {
    let mut cu = Vec::with_capacity(num_tokens_per_req.len() + 1);
    cu.push(0);
    let mut total = 0u32;
    for &num_req_tokens in num_tokens_per_req {
        assert!(num_req_tokens > 0, "bench request segments must be non-empty");
        total = total
            .checked_add(num_req_tokens)
            .expect("bench cu_tokens total overflow");
        cu.push(total.try_into().expect("cu_tokens value must fit i32"));
    }
    cu
}

fn gdn_conv_state_fixture(existing_context_len: u32, num_reqs: usize, len: usize) -> Vec<f32> {
    gdn_state_fixture(
        existing_context_len,
        len,
        num_reqs,
        GDN_CONV_DIM * (GDN_CONV_KERNEL_SIZE - 1),
        0.00390625,
    )
}

fn gdn_recurrent_state_fixture(existing_context_len: u32, num_reqs: usize, len: usize) -> Vec<f32> {
    gdn_state_fixture(
        existing_context_len,
        len,
        num_reqs,
        GDN_V_HEADS * GDN_V_HEAD_DIM * GDN_QK_HEAD_DIM,
        0.000_244_140_63,
    )
}

fn gdn_state_fixture(existing_context_len: u32, len: usize, num_reqs: usize, slot_len: usize, scale: f32) -> Vec<f32> {
    let mut values = vec![0.0; len];
    if existing_context_len == 0 {
        return values;
    }
    let src_state_len = num_reqs * slot_len;
    assert!(
        src_state_len <= len,
        "GDN state fixture source state slot cannot exceed state arena"
    );
    for (index, value) in values.iter_mut().take(src_state_len).enumerate() {
        *value = ((index % 29) as f32 - 14.0) * scale;
    }
    values
}

fn flat_token_indices(num_tokens_per_req: &[u32], existing_context_len: u32) -> Vec<u32> {
    num_tokens_per_req
        .iter()
        .flat_map(|&num_req_tokens| existing_context_len..existing_context_len + num_req_tokens)
        .collect()
}

fn identity_u32(num_values: u32) -> Vec<u32> {
    (0..num_values).collect()
}

fn tensor_bytes(tensors: &SafeTensors<'_>, name: &str, dtype: safetensors::Dtype) -> Vec<u8> {
    let view = tensors
        .tensor(name)
        .unwrap_or_else(|err| panic!("missing safetensor {name}: {err:?}"));
    assert_eq!(view.dtype(), dtype, "unexpected dtype for tensor {name}");
    validate_tensor_shape(name, &view);
    view.data().to_vec()
}

fn bf16_tensor_as_f32(tensors: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    let bytes = tensor_bytes(tensors, name, safetensors::Dtype::BF16);
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn a_log_decay(tensors: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    bf16_tensor_as_f32(tensors, name)
        .into_iter()
        .map(|value| -value.exp())
        .collect()
}

fn validate_tensor_shape(name: &str, view: &TensorView<'_>) {
    let shape = view.shape();
    if name.ends_with("linear_attn.in_proj_qkv.weight") {
        assert_eq!(shape, &[GDN_CONV_DIM, packed_k_words(HIDDEN_DIM)]);
    } else if name.ends_with("linear_attn.in_proj_qkv.scales") || name.ends_with("linear_attn.in_proj_qkv.biases") {
        assert_eq!(shape, &[GDN_CONV_DIM, groups(HIDDEN_DIM)]);
    } else if name.ends_with("linear_attn.in_proj_a.weight") || name.ends_with("linear_attn.in_proj_b.weight") {
        assert_eq!(shape, &[GDN_V_HEADS, packed_k_words(HIDDEN_DIM)]);
    } else if name.ends_with("linear_attn.in_proj_a.scales")
        || name.ends_with("linear_attn.in_proj_a.biases")
        || name.ends_with("linear_attn.in_proj_b.scales")
        || name.ends_with("linear_attn.in_proj_b.biases")
    {
        assert_eq!(shape, &[GDN_V_HEADS, groups(HIDDEN_DIM)]);
    } else if name.ends_with("linear_attn.in_proj_z.weight") {
        assert_eq!(shape, &[GDN_V_DIM, packed_k_words(HIDDEN_DIM)]);
    } else if name.ends_with("linear_attn.in_proj_z.scales") || name.ends_with("linear_attn.in_proj_z.biases") {
        assert_eq!(shape, &[GDN_V_DIM, groups(HIDDEN_DIM)]);
    } else if name.ends_with("linear_attn.conv1d.weight") {
        assert_eq!(shape, &[GDN_CONV_DIM, GDN_CONV_KERNEL_SIZE, 1]);
    } else if name.ends_with("linear_attn.norm.weight") {
        assert_eq!(shape, &[GDN_V_HEAD_DIM]);
    } else if name.ends_with("linear_attn.A_log") || name.ends_with("linear_attn.dt_bias") {
        assert_eq!(shape, &[GDN_V_HEADS]);
    } else if name.ends_with("linear_attn.out_proj.weight") {
        assert_eq!(shape, &[HIDDEN_DIM, packed_k_words(GDN_V_DIM)]);
    } else if name.ends_with("linear_attn.out_proj.scales") || name.ends_with("linear_attn.out_proj.biases") {
        assert_eq!(shape, &[HIDDEN_DIM, groups(GDN_V_DIM)]);
    } else {
        panic!("unexpected GDN tensor name {name}");
    }
}

fn validate_qkvabz_sizes(weight: &[u8], scales: &[f32], biases: &[f32]) {
    assert_eq!(
        weight.len(),
        GDN_QKVABZ_DIM * packed_k_words(HIDDEN_DIM) * size_of::<u32>()
    );
    assert_eq!(scales.len(), GDN_QKVABZ_DIM * groups(HIDDEN_DIM));
    assert_eq!(biases.len(), scales.len());
}

fn concat_parts(parts: &[&[u8]]) -> Vec<u8> {
    let len = parts.iter().map(|part| part.len()).sum();
    let mut out = Vec::with_capacity(len);
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

fn concat_f32_parts(parts: &[&[f32]]) -> Vec<f32> {
    let len = parts.iter().map(|part| part.len()).sum();
    let mut out = Vec::with_capacity(len);
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

fn packed_k_words(k: usize) -> usize {
    k * BITS as usize / 32
}

fn groups(k: usize) -> usize {
    k / GROUP_SIZE as usize
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
    num_tokens: u32,
    num_reqs: u32,
    existing_context_len: Option<u32>,
    path: Option<&str>,
    iters: usize,
    samples: &[f64],
) {
    let median_us = median(samples);
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    let context_text = existing_context_len
        .map(|value| format!(" ctx={value}"))
        .unwrap_or_default();
    let path_text = path.map(|value| format!(" path={value}")).unwrap_or_default();
    println!(
        "perf component=gdn impl=full-forward-replay tokens={num_tokens} num_reqs={num_reqs}{context_text}{path_text} \
         iters={iters} runs={} median_us={median_us:.3} samples_us=[{sample_text}]",
        samples.len()
    );
}

fn print_skip(num_tokens: u32, num_reqs: u32, existing_context_len: Option<u32>, path: Option<&str>, reason: &str) {
    let context_text = existing_context_len
        .map(|value| format!(" ctx={value}"))
        .unwrap_or_default();
    let path_text = path.map(|value| format!(" path={value}")).unwrap_or_default();
    println!("skip component=gdn tokens={num_tokens} num_reqs={num_reqs}{context_text}{path_text} reason={reason}",);
}

fn print_named_perf(
    component: &str,
    num_tokens: u32,
    num_reqs: u32,
    existing_context_len: Option<u32>,
    iters: usize,
    samples: &[f64],
) {
    let median_us = median(samples);
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    let context_text = existing_context_len
        .map(|value| format!(" ctx={value}"))
        .unwrap_or_default();
    println!(
        "perf component={component} impl=subcomponent-replay tokens={num_tokens} num_reqs={num_reqs}{context_text} \
         iters={iters} runs={} median_us={median_us:.3} samples_us=[{sample_text}]",
        samples.len()
    );
}

fn build_single_invocation_replay<I>(stream: &Stream, invocation: I) -> ReplayProgram
where
    I: Operator,
{
    let mut builder = MetalReplayRuntime::new(stream).create_recorder();
    builder.record(ReplayOp::opaque(invocation));
    builder.build()
}

fn gdn_qkvabz_affine_shape(num_tokens: u32) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape {
        m: num_tokens.try_into().expect("GDN qkvabz m must fit i32"),
        n: GDN_QKVABZ_DIM.try_into().expect("GDN qkvabz n must fit i32"),
        k: HIDDEN_DIM.try_into().expect("GDN qkvabz k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("GDN group size must fit i32"),
        bits: BITS.try_into().expect("GDN bits must fit i32"),
        input_dtype: Dtype::Float32,
        output_dtype: Dtype::Float32,
        affine_dtype: Dtype::Float32,
    }
}

fn gdn_output_affine_shape(num_tokens: u32) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape {
        m: num_tokens.try_into().expect("GDN output m must fit i32"),
        n: HIDDEN_DIM.try_into().expect("GDN output n must fit i32"),
        k: GDN_V_DIM.try_into().expect("GDN output k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("GDN group size must fit i32"),
        bits: BITS.try_into().expect("GDN bits must fit i32"),
        input_dtype: Dtype::Float32,
        output_dtype: Dtype::Bfloat16,
        affine_dtype: Dtype::Bfloat16,
    }
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
