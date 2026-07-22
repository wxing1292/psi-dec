#[path = "gqa/fixture.rs"]
mod fixture;

fn main() {
    fixture::run(Args::parse());
}
use std::fs::File;
use std::mem::size_of;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::GQAActivationGateBuffers;
use inference_backend_metal::components::GQAActivationGateConfig;
use inference_backend_metal::components::GQAActivationGateKernel;
use inference_backend_metal::components::GQAActivationGateShape;
use inference_backend_metal::components::GQAKVPageUpdate;
use inference_backend_metal::components::GQAKVPageUpdateBuffers;
use inference_backend_metal::components::GQAKVPageUpdateConfig;
use inference_backend_metal::components::GQAKVPageUpdateShape;
use inference_backend_metal::components::GQANormRopeBuffers;
use inference_backend_metal::components::GQANormRopeConfig;
use inference_backend_metal::components::GQANormRopeKernel;
use inference_backend_metal::components::GQANormRopeShape;
use inference_backend_metal::components::GQAPageTableLayout as MetalGQAPageTableLayout;
use inference_backend_metal::components::GQAPagedSDPAConfig;
use inference_backend_metal::components::GQAPagedSDPAKernels;
use inference_backend_metal::components::GQAPagedSDPAMapBuffers;
use inference_backend_metal::components::GQAPagedSDPAReduceBuffers;
use inference_backend_metal::components::GQAPagedSDPAShape;
use inference_backend_metal::components::GQAProjectionSplitBuffers;
use inference_backend_metal::components::GQAProjectionSplitConfig;
use inference_backend_metal::components::GQAProjectionSplitKernel;
use inference_backend_metal::components::GQAProjectionSplitShape;
use inference_backend_metal::components::GQATiledSDPAKernels;
use inference_backend_metal::components::GQATiledSDPAMapBuffers;
use inference_backend_metal::components::GQATiledSDPAReduceBuffers;
use inference_backend_metal::components::GQATiledSDPAShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::Operator;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_metal::attn::gqa::backend::GQA;
use inference_executor_metal::attn::gqa::backend::GQAInput;
use inference_executor_metal::attn::gqa::backend::GQAKVCacheBindings;
use inference_executor_metal::attn::gqa::backend::GQAMetalConfig;
use inference_executor_metal::attn::gqa::backend::GQAWeights;
use inference_executor_metal::attn::gqa::batch_metadata::GQAMetadataBuffers;
use inference_executor_metal::attn::gqa::scratch::GQAScratchBindings;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::def::replay_op::ReplayOp;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

const HIDDEN_DIM: usize = 2048;
const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

const GQA_ROPE_DIM: u32 = 64;
const GQA_ROPE_THETA: f32 = 10_000_000.0;
const GQA_NORM_EPS: f32 = 1.0e-6;
const GQA_MAX_TOKENS: usize = 64;
const TOKENS_PER_PAGE: u32 = 16;
const KV_TOKEN_TILE_SIZE: u32 = 256;
const NUM_THREADS_PER_THREADBLOCK: u32 = 256;
const Q_HEAD_TILE_SIZE_CAP: u32 = 8;
const Q_TOKEN_TILE_SIZE: u32 = 8;
const TILED_KV_TOKEN_TILE_SIZE: u32 = 16;

#[derive(Clone, Copy)]
struct GQABenchParams {
    kv_token_tile_size: u32,
    num_threads_per_threadblock: u32,
    max_q_head_tile_size: u32,
    tiled_q_token_tile_size: u32,
    tiled_kv_token_tile_size: u32,
    tiled_q_head_tile_size: u32,
}

#[derive(Clone, Copy)]
struct GQAModelProfile {
    k: &'static str,
    shard: &'static str,
    hidden_dim: usize,
    model_layer_index: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl GQAModelProfile {
    const QWEN35_27B: Self = Self {
        k: "27b",
        shard: "model-00001-of-00003.safetensors",
        hidden_dim: 5120,
        model_layer_index: 3,
        num_q_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
    };

    const QWEN35_35B_A3B: Self = Self {
        k: "35b",
        shard: "model-00001-of-00004.safetensors",
        hidden_dim: 2048,
        model_layer_index: 3,
        num_q_heads: 16,
        num_kv_heads: 2,
        head_dim: 256,
    };

    fn parse(value: &str) -> Self {
        match value {
            "27b" => Self::QWEN35_27B,
            "35b" => Self::QWEN35_35B_A3B,
            _ => panic!("unknown GQA model profile {value:?}; expected 27b or 35b"),
        }
    }

    fn q_dim(self) -> usize {
        self.num_q_heads * self.head_dim
    }

    fn kv_dim(self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    fn qgkv_dim(self) -> usize {
        self.q_dim() * 2 + self.kv_dim() * 2
    }

    fn page_bytes(self) -> u32 {
        (2 * self.num_kv_heads * TOKENS_PER_PAGE as usize * self.head_dim * Dtype::Bfloat16.item_size())
            .try_into()
            .expect("GQA page bytes must fit u32")
    }
}

struct Args {
    model_dir: PathBuf,
    model: GQAModelProfile,
    tokens: Vec<u32>,
    contexts: Vec<u32>,
    num_reqs: Vec<u32>,
    tokens_per_req: Option<Vec<u32>>,
    paths: Vec<GQABenchPath>,
    selected_subcomponents: Vec<String>,
    params: GQABenchParams,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
    subcomponents: bool,
    validate_tiled: bool,
    print_limits: bool,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            model: GQAModelProfile::QWEN35_35B_A3B,
            tokens: vec![1, 2, 4, 8, 16, 32, 64],
            contexts: Vec::new(),
            num_reqs: vec![1],
            tokens_per_req: None,
            paths: vec![GQABenchPath::ContextParallel, GQABenchPath::Tiled],
            selected_subcomponents: default_gqa_subcomponents(),
            params: GQABenchParams {
                kv_token_tile_size: KV_TOKEN_TILE_SIZE,
                num_threads_per_threadblock: NUM_THREADS_PER_THREADBLOCK,
                max_q_head_tile_size: Q_HEAD_TILE_SIZE_CAP,
                tiled_q_token_tile_size: Q_TOKEN_TILE_SIZE,
                tiled_kv_token_tile_size: TILED_KV_TOKEN_TILE_SIZE,
                tiled_q_head_tile_size: 0,
            },
            iters: 50,
            warmup_iters: 20,
            runs: 1,
            subcomponents: false,
            validate_tiled: false,
            print_limits: false,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--gqa-model" => args.model = GQAModelProfile::parse(&next_arg(&mut values, &arg)),
                "--tokens" => args.tokens = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--contexts" => args.contexts = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--num-reqs" => args.num_reqs = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--gqa-tokens-per-req" => {
                    args.tokens_per_req = Some(parse_u32_list(&next_arg(&mut values, &arg), &arg))
                },
                "--gqa-paths" => args.paths = parse_gqa_paths(&next_arg(&mut values, &arg)),
                "--gqa-subcomponents" => {
                    args.selected_subcomponents = parse_string_list(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-kv-token-tile-size" => {
                    args.params.kv_token_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-num-threads-per-threadblock" => {
                    args.params.num_threads_per_threadblock = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-max-q-head-tile-size" => {
                    args.params.max_q_head_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-tiled-q-token-tile-size" => {
                    args.params.tiled_q_token_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-tiled-kv-token-tile-size" => {
                    args.params.tiled_kv_token_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-tiled-q-head-tile-size" => {
                    args.params.tiled_q_head_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--subcomponents" => args.subcomponents = true,
                "--validate-tiled" => args.validate_tiled = true,
                "--print-limits" => args.print_limits = true,
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
        assert!(!args.paths.is_empty(), "--gqa-paths must include at least one path");
        assert!(
            !args.selected_subcomponents.is_empty(),
            "--gqa-subcomponents must include at least one subcomponent"
        );
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        for &num_reqs in &args.num_reqs {
            assert!(num_reqs > 0, "--num-reqs entries must be positive");
        }
        if let Some(tokens_per_req) = &args.tokens_per_req {
            assert!(!tokens_per_req.is_empty(), "--gqa-tokens-per-req must not be empty");
            assert!(
                tokens_per_req.iter().all(|&count| count > 0),
                "--gqa-tokens-per-req entries must be positive"
            );
            assert!(
                tokens_per_req.iter().sum::<u32>() as usize <= GQA_MAX_TOKENS,
                "--gqa-tokens-per-req total exceeds the GQA bench token capacity"
            );
        }
        args
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GQABenchPath {
    ContextParallel,
    Tiled,
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{name} requires a value; pass --help for usage"))
}

fn parse_gqa_paths(value: &str) -> Vec<GQABenchPath> {
    let paths = parse_string_list(value, "--gqa-paths")
        .into_iter()
        .map(|part| {
            match part.as_str() {
                "context_parallel" => GQABenchPath::ContextParallel,
                "tiled" => GQABenchPath::Tiled,
                _ => {
                    panic!("invalid --gqa-paths entry {part:?}; expected context_parallel or tiled")
                },
            }
        })
        .collect::<Vec<_>>();
    assert!(!paths.is_empty(), "--gqa-paths must contain at least one path");
    paths
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

fn default_gqa_subcomponents() -> Vec<String> {
    vec![
        "qgkv-proj".to_string(),
        "split".to_string(),
        "q-norm-rope".to_string(),
        "k-norm-rope".to_string(),
        "kv-update".to_string(),
        "sdpa-context-parallel".to_string(),
        "sdpa-tiled".to_string(),
        "gate".to_string(),
        "output-proj".to_string(),
    ]
}

fn print_help_and_exit() -> ! {
    println!(
        r#"qwen35_gqa bench
--model-dir PATH
--gqa-model 27b|35b
--tokens 1,2,4,8,16,32,64
--contexts 0,128
--num-reqs 1,2,4
--gqa-tokens-per-req 1,2,4,8
--gqa-paths context_parallel,tiled
--subcomponents
--gqa-subcomponents qgkv-proj,split,q-norm-rope,k-norm-rope,kv-update,sdpa-context-parallel,sdpa-tiled,gate,output-proj
--validate-tiled
--gqa-kv-token-tile-size N
--gqa-num-threads-per-threadblock N
--gqa-max-q-head-tile-size N
--gqa-tiled-q-token-tile-size 8|16
--gqa-tiled-kv-token-tile-size 8|16
--gqa-tiled-q-head-tile-size N
--print-limits
--iters N
--warmup-iters N
--runs N"#
    );
    std::process::exit(0);
}

fn gqa_context_parallel_threadblock_memory_bytes(params: GQABenchParams) -> usize {
    (params.max_q_head_tile_size as usize * params.kv_token_tile_size as usize
        + params.num_threads_per_threadblock as usize)
        * size_of::<f32>()
}

fn print_gqa_kernel_limits(device: &Device, params: GQABenchParams) {
    let device_max = device.max_threadblock_memory_length();
    let threadblock_memory = gqa_context_parallel_threadblock_memory_bytes(params);
    println!(
        "metal-limits device={} max_threadblock_memory_bytes={} gqa_path=context_parallel gqa_kv_token_tile_size={} \
         gqa_num_threads_per_threadblock={} gqa_max_q_head_tile_size={} gqa_threadblock_memory_bytes={} gqa_valid={}",
        device.name(),
        device_max,
        params.kv_token_tile_size,
        params.num_threads_per_threadblock,
        params.max_q_head_tile_size,
        threadblock_memory,
        threadblock_memory <= device_max
    );
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

fn assert_bf16_close(expected: &Buffer, actual: &Buffer, num_values: usize, tolerance: f32) {
    let (max_abs_index, expected_value, actual_value, max_abs_diff) = max_bf16_diff(expected, actual, num_values);
    assert!(
        max_abs_diff <= tolerance,
        "GQA token-tiled mismatch at {max_abs_index}: expected={expected_value} actual={actual_value} \
         max_abs_diff={max_abs_diff} tolerance={tolerance}"
    );
}

fn max_bf16_diff(expected_buffer: &Buffer, actual_buffer: &Buffer, num_values: usize) -> (usize, f32, f32, f32) {
    let expected_values = expected_buffer.read_typed::<u16>(0, num_values);
    let actual_values = actual_buffer.read_typed::<u16>(0, num_values);
    let mut max_abs_diff = 0.0f32;
    let mut max_abs_index = 0usize;
    for (index, (&expected_value, &actual_value)) in expected_values.iter().zip(&actual_values).enumerate() {
        let diff = (bf16::from_bits(expected_value).to_f32() - bf16::from_bits(actual_value).to_f32()).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_abs_index = index;
        }
    }
    let expected_value = bf16::from_bits(expected_values[max_abs_index]).to_f32();
    let actual_value = bf16::from_bits(actual_values[max_abs_index]).to_f32();
    (max_abs_index, expected_value, actual_value, max_abs_diff)
}

fn gqa_attention_reference_at(
    q_buffer: &Buffer,
    kv_pages_buffer: &Buffer,
    page_ids_buffer: &Buffer,
    batch_metadata: &GQAMetadataBuffers,
    num_blocks: u32,
    output_index: usize,
    model: GQAModelProfile,
) -> f32 {
    let dim = output_index % model.head_dim;
    let q_head_index = (output_index / model.head_dim) % model.num_q_heads;
    let flat_token_index = output_index / model.q_dim();
    let req_slot = batch_metadata.req_slots().read_typed::<u32>(flat_token_index, 1)[0] as usize;
    let token_index = batch_metadata
        .flat_token_indices()
        .read_typed::<u32>(flat_token_index, 1)[0] as usize;
    let kv_head_index = q_head_index / (model.num_q_heads / model.num_kv_heads);
    let q_values = q_buffer.read_typed::<u16>(
        flat_token_index * model.q_dim() + q_head_index * model.head_dim,
        model.head_dim,
    );
    let kv_page_values = kv_pages_buffer.read_typed::<u16>(0, kv_pages_buffer.len_bytes() / size_of::<u16>());
    let page_id_values = page_ids_buffer.read_typed::<u32>(0, page_ids_buffer.len_bytes() / size_of::<u32>());
    let page_slots = 2 * model.num_kv_heads * TOKENS_PER_PAGE as usize * model.head_dim;
    let mut logits = Vec::with_capacity(token_index + 1);
    for context_token_index in 0..=token_index {
        let block_index = context_token_index / TOKENS_PER_PAGE as usize;
        let page_id = page_id_values[req_slot * num_blocks as usize + block_index] as usize;
        let page_token_index = context_token_index % TOKENS_PER_PAGE as usize;
        let k_start =
            page_id * page_slots + (kv_head_index * TOKENS_PER_PAGE as usize + page_token_index) * model.head_dim;
        let dot = q_values
            .iter()
            .zip(&kv_page_values[k_start..k_start + model.head_dim])
            .map(|(&q_value, &k_value)| bf16::from_bits(q_value).to_f32() * bf16::from_bits(k_value).to_f32())
            .sum::<f32>();
        logits.push(dot * (model.head_dim as f32).sqrt().recip());
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum = logits.iter().map(|logit| (logit - max_logit).exp()).sum::<f32>();
    logits
        .iter()
        .enumerate()
        .map(|(context_token_index, logit)| {
            let block_index = context_token_index / TOKENS_PER_PAGE as usize;
            let page_id = page_id_values[req_slot * num_blocks as usize + block_index] as usize;
            let page_token_index = context_token_index % TOKENS_PER_PAGE as usize;
            let v_index = page_id * page_slots
                + ((model.num_kv_heads + kv_head_index) * TOKENS_PER_PAGE as usize + page_token_index) * model.head_dim
                + dim;
            ((logit - max_logit).exp() / exp_sum) * bf16::from_bits(kv_page_values[v_index]).to_f32()
        })
        .sum()
}

fn page_table(num_reqs: u32, num_blocks: u32) -> Vec<u32> {
    identity_u32(num_reqs.checked_mul(num_blocks).expect("GQA page table size overflow"))
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

fn gqa_tensor_bytes(
    tensors: &SafeTensors<'_>,
    name: &str,
    dtype: safetensors::Dtype,
    model: GQAModelProfile,
) -> Vec<u8> {
    let view = tensors
        .tensor(name)
        .unwrap_or_else(|err| panic!("missing safetensor {name}: {err:?}"));
    assert_eq!(view.dtype(), dtype, "unexpected dtype for tensor {name}");
    validate_gqa_tensor_shape(name, &view, model);
    view.data().to_vec()
}

fn gqa_bf16_tensor_as_f32(tensors: &SafeTensors<'_>, name: &str, model: GQAModelProfile) -> Vec<f32> {
    let bytes = gqa_tensor_bytes(tensors, name, safetensors::Dtype::BF16, model);
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn validate_gqa_tensor_shape(name: &str, view: &TensorView<'_>, model: GQAModelProfile) {
    let shape = view.shape();
    if name.ends_with("self_attn.q_proj.weight") {
        assert_eq!(shape, &[model.q_dim() * 2, packed_k_words(model.hidden_dim)]);
    } else if name.ends_with("self_attn.q_proj.scales") || name.ends_with("self_attn.q_proj.biases") {
        assert_eq!(shape, &[model.q_dim() * 2, groups(model.hidden_dim)]);
    } else if name.ends_with("self_attn.k_proj.weight") || name.ends_with("self_attn.v_proj.weight") {
        assert_eq!(shape, &[model.kv_dim(), packed_k_words(model.hidden_dim)]);
    } else if name.ends_with("self_attn.k_proj.scales")
        || name.ends_with("self_attn.k_proj.biases")
        || name.ends_with("self_attn.v_proj.scales")
        || name.ends_with("self_attn.v_proj.biases")
    {
        assert_eq!(shape, &[model.kv_dim(), groups(model.hidden_dim)]);
    } else if name.ends_with("self_attn.o_proj.weight") {
        assert_eq!(shape, &[model.hidden_dim, packed_k_words(model.q_dim())]);
    } else if name.ends_with("self_attn.o_proj.scales") || name.ends_with("self_attn.o_proj.biases") {
        assert_eq!(shape, &[model.hidden_dim, groups(model.q_dim())]);
    } else if name.ends_with("self_attn.q_norm.weight") || name.ends_with("self_attn.k_norm.weight") {
        assert_eq!(shape, &[model.head_dim]);
    } else {
        panic!("unexpected GQA tensor name {name}");
    }
}

fn validate_qgkv_sizes(weight: &[u8], scales: &[u8], biases: &[u8], model: GQAModelProfile) {
    assert_eq!(
        weight.len(),
        model.qgkv_dim() * packed_k_words(model.hidden_dim) * size_of::<u32>()
    );
    assert_eq!(
        scales.len(),
        model.qgkv_dim() * groups(model.hidden_dim) * Dtype::Bfloat16.item_size()
    );
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
        "perf component=gqa impl=full-forward-replay tokens={num_tokens} num_reqs={num_reqs}{context_text}{path_text} \
         iters={iters} runs={} median_us={median_us:.3} samples_us=[{sample_text}]",
        samples.len()
    );
}

fn print_skip(num_tokens: u32, num_reqs: u32, existing_context_len: Option<u32>, path: Option<&str>, reason: &str) {
    let context_text = existing_context_len
        .map(|value| format!(" ctx={value}"))
        .unwrap_or_default();
    let path_text = path.map(|value| format!(" path={value}")).unwrap_or_default();
    println!("skip component=gqa tokens={num_tokens} num_reqs={num_reqs}{context_text}{path_text} reason={reason}",);
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

fn gqa_qgkv_affine_shape(num_tokens: u32, model: GQAModelProfile) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape {
        m: num_tokens.try_into().expect("GQA qgkv m must fit i32"),
        n: model.qgkv_dim().try_into().expect("GQA qgkv n must fit i32"),
        k: model.hidden_dim.try_into().expect("GQA qgkv k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("GQA group size must fit i32"),
        bits: BITS.try_into().expect("GQA bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        affine_dtype: Dtype::Bfloat16,
    }
}

fn gqa_output_affine_shape(num_tokens: u32, model: GQAModelProfile) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape {
        m: num_tokens.try_into().expect("GQA output m must fit i32"),
        n: model.hidden_dim.try_into().expect("GQA output n must fit i32"),
        k: model.q_dim().try_into().expect("GQA output k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("GQA group size must fit i32"),
        bits: BITS.try_into().expect("GQA bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        affine_dtype: Dtype::Bfloat16,
    }
}

fn gqa_projection_split_config(model: GQAModelProfile) -> GQAProjectionSplitConfig {
    GQAProjectionSplitConfig::bf16(
        model.num_q_heads.try_into().expect("GQA q heads must fit u32"),
        model.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        model.head_dim.try_into().expect("GQA head_dim must fit u32"),
    )
}

fn gqa_norm_rope_config(num_heads: usize, model: GQAModelProfile) -> GQANormRopeConfig {
    GQANormRopeConfig::bf16(
        num_heads.try_into().expect("GQA norm head count must fit u32"),
        model.head_dim.try_into().expect("GQA norm head_dim must fit u32"),
        GQA_ROPE_DIM,
        GQA_NORM_EPS,
        GQA_ROPE_THETA,
        1.0,
    )
}

fn gqa_norm_rope_shape(num_tokens: u32, _num_heads: usize, _model: GQAModelProfile) -> GQANormRopeShape {
    GQANormRopeShape { num_tokens }
}

fn gqa_kv_update_config(model: GQAModelProfile, page_bytes: u32) -> GQAKVPageUpdateConfig {
    GQAKVPageUpdateConfig {
        num_kv_heads: model.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        head_dim: model.head_dim.try_into().expect("GQA head_dim must fit u32"),
        page_bytes,
        dtype: Dtype::Bfloat16,
    }
}

fn gqa_activation_gate_config(model: GQAModelProfile) -> GQAActivationGateConfig {
    GQAActivationGateConfig::bf16(
        model.num_q_heads.try_into().expect("GQA q heads must fit u32"),
        model.head_dim.try_into().expect("GQA head_dim must fit u32"),
    )
}

fn gqa_sdpa_config(
    num_reqs: u32,
    end_context_len: u32,
    params: GQABenchParams,
    model: GQAModelProfile,
) -> GQAPagedSDPAConfig {
    GQAPagedSDPAConfig {
        num_q_heads: model.num_q_heads.try_into().expect("GQA q heads must fit u32"),
        num_kv_heads: model.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        head_dim: model.head_dim.try_into().expect("GQA head_dim must fit u32"),
        scale: (model.head_dim as f32).sqrt().recip(),
        page_bytes: model.page_bytes(),
        page_table_layout: gqa_page_table_layout(num_reqs, end_context_len),
        gqa_layer_index: 0,
        kv_token_tile_size: params.kv_token_tile_size,
        num_threads_per_threadblock: params.num_threads_per_threadblock,
        q_head_tile_size: u32::try_from(model.num_q_heads / model.num_kv_heads)
            .expect("GQA q heads per KV head must fit u32")
            .min(params.max_q_head_tile_size),
        dtype: Dtype::Bfloat16,
    }
}

fn gqa_sdpa_shape(replay_shape: GQAReplayShape) -> GQAPagedSDPAShape {
    GQAPagedSDPAShape {
        num_tokens: replay_shape.num_tokens,
        total_sdpa_map_task_templates: replay_shape.total_sdpa_map_task_templates,
    }
}

fn gqa_page_table_layout(num_reqs: u32, end_context_len: u32) -> MetalGQAPageTableLayout {
    MetalGQAPageTableLayout {
        num_req_slots: num_reqs,
        num_blocks: end_context_len.div_ceil(TOKENS_PER_PAGE).max(1),
        num_gqa_layers: 1,
        num_page_ids_per_block: 1,
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
