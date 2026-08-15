#[path = "gqa/fixture.rs"]
mod fixture;

fn main() {
    fixture::run(Args::parse());
}
use std::mem::size_of;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::GQAActivationGateConfig;
use inference_backend_metal::components::GQAKVPageWriteConfig;
use inference_backend_metal::components::GQAPageTableLayout as MetalGQAPageTableLayout;
use inference_backend_metal::components::GQAQGKVSplitConfig;
use inference_backend_metal::components::GQASplitKVSingleQConfig;
use inference_backend_metal::components::GQASplitKVSingleQShape;
use inference_backend_metal::components::GQASplitKVTiledQConfig;
use inference_backend_metal::components::RMSNormRopeConfig;
use inference_backend_metal::components::RMSNormRopeShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::Operator;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_metal::attn::gqa::batch_metadata::GQAMetadataBuffers;
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
const SPLIT_KV_SINGLE_Q_KV_TOKEN_TILE_SIZE: u32 = 256;
const SPLIT_KV_SINGLE_Q_NUM_THREADS_PER_THREADBLOCK: u32 = 256;
const SPLIT_KV_SINGLE_Q_Q_HEAD_TILE_SIZE_CAP: u32 = 8;
const SPLIT_KV_TILED_Q_TOKEN_TILE_SIZE: u32 = 8;
const SPLIT_KV_TILED_Q_KV_TOKEN_TILE_SIZE: u32 = 16;

#[derive(Clone, Copy)]
struct GQABenchParams {
    split_kv_single_q_kv_token_tile_size: u32,
    split_kv_single_q_num_threads_per_threadblock: u32,
    split_kv_single_q_max_q_head_tile_size: u32,
    split_kv_tiled_q_token_tile_size: u32,
    split_kv_tiled_q_kv_token_tile_size: u32,
    split_kv_tiled_q_head_tile_size: u32,
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
    split_kv_variants: Vec<GQASplitKVBenchVariant>,
    selected_subcomponents: Vec<String>,
    params: GQABenchParams,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
    subcomponents: bool,
    validate_split_kv_tiled_q: bool,
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
            split_kv_variants: vec![GQASplitKVBenchVariant::SingleQ, GQASplitKVBenchVariant::TiledQ],
            selected_subcomponents: default_gqa_subcomponents(),
            params: GQABenchParams {
                split_kv_single_q_kv_token_tile_size: SPLIT_KV_SINGLE_Q_KV_TOKEN_TILE_SIZE,
                split_kv_single_q_num_threads_per_threadblock: SPLIT_KV_SINGLE_Q_NUM_THREADS_PER_THREADBLOCK,
                split_kv_single_q_max_q_head_tile_size: SPLIT_KV_SINGLE_Q_Q_HEAD_TILE_SIZE_CAP,
                split_kv_tiled_q_token_tile_size: SPLIT_KV_TILED_Q_TOKEN_TILE_SIZE,
                split_kv_tiled_q_kv_token_tile_size: SPLIT_KV_TILED_Q_KV_TOKEN_TILE_SIZE,
                split_kv_tiled_q_head_tile_size: 0,
            },
            iters: 50,
            warmup_iters: 20,
            runs: 1,
            subcomponents: false,
            validate_split_kv_tiled_q: false,
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
                "--gqa-split-kv-variants" => {
                    args.split_kv_variants = parse_split_kv_variants(&next_arg(&mut values, &arg))
                },
                "--gqa-subcomponents" => {
                    args.selected_subcomponents = parse_string_list(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-split-kv-single-q-kv-token-tile-size" => {
                    args.params.split_kv_single_q_kv_token_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-split-kv-single-q-num-threads-per-threadblock" => {
                    args.params.split_kv_single_q_num_threads_per_threadblock =
                        parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-split-kv-single-q-max-q-head-tile-size" => {
                    args.params.split_kv_single_q_max_q_head_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-split-kv-tiled-q-token-tile-size" => {
                    args.params.split_kv_tiled_q_token_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-split-kv-tiled-q-kv-token-tile-size" => {
                    args.params.split_kv_tiled_q_kv_token_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--gqa-split-kv-tiled-q-head-tile-size" => {
                    args.params.split_kv_tiled_q_head_tile_size = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--subcomponents" => args.subcomponents = true,
                "--validate-split-kv-tiled-q" => args.validate_split_kv_tiled_q = true,
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
        assert!(
            !args.split_kv_variants.is_empty(),
            "--gqa-split-kv-variants must include at least one variant"
        );
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
enum GQASplitKVBenchVariant {
    SingleQ,
    TiledQ,
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{name} requires a value; pass --help for usage"))
}

fn parse_split_kv_variants(value: &str) -> Vec<GQASplitKVBenchVariant> {
    let variants = parse_string_list(value, "--gqa-split-kv-variants")
        .into_iter()
        .map(|part| {
            match part.as_str() {
                "single_q" => GQASplitKVBenchVariant::SingleQ,
                "tiled_q" => GQASplitKVBenchVariant::TiledQ,
                _ => {
                    panic!("invalid --gqa-split-kv-variants entry {part:?}; expected single_q or tiled_q")
                },
            }
        })
        .collect::<Vec<_>>();
    assert!(
        !variants.is_empty(),
        "--gqa-split-kv-variants must contain at least one variant"
    );
    variants
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
        "qgkv".to_string(),
        "qgkv-to-q-g-k-v".to_string(),
        "q-norm-rope".to_string(),
        "k-norm-rope".to_string(),
        "kv-page-write".to_string(),
        "split-kv-single-q".to_string(),
        "split-kv-tiled-q".to_string(),
        "gate".to_string(),
        "output".to_string(),
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
--gqa-split-kv-variants single_q,tiled_q
--subcomponents
--gqa-subcomponents qgkv,qgkv-to-q-g-k-v,q-norm-rope,k-norm-rope,kv-page-write,split-kv-single-q,split-kv-tiled-q,gate,output
--validate-split-kv-tiled-q
--gqa-split-kv-single-q-kv-token-tile-size N
--gqa-split-kv-single-q-num-threads-per-threadblock N
--gqa-split-kv-single-q-max-q-head-tile-size N
--gqa-split-kv-tiled-q-token-tile-size 8|16
--gqa-split-kv-tiled-q-kv-token-tile-size 8|16
--gqa-split-kv-tiled-q-head-tile-size N
--print-limits
--iters N
--warmup-iters N
--runs N"#
    );
    std::process::exit(0);
}

fn split_kv_single_q_threadblock_memory_bytes(params: GQABenchParams) -> usize {
    (params.split_kv_single_q_max_q_head_tile_size as usize * params.split_kv_single_q_kv_token_tile_size as usize
        + params.split_kv_single_q_num_threads_per_threadblock as usize)
        * size_of::<f32>()
}

fn print_gqa_kernel_limits(device: &Device, params: GQABenchParams) {
    let device_max = device.max_threadblock_memory_length();
    let threadblock_memory = split_kv_single_q_threadblock_memory_bytes(params);
    println!(
        "metal-limits device={} max_threadblock_memory_bytes={} gqa_split_kv_variant=single_q \
         gqa_split_kv_single_q_kv_token_tile_size={} gqa_split_kv_single_q_num_threads_per_threadblock={} \
         gqa_split_kv_single_q_max_q_head_tile_size={} gqa_threadblock_memory_bytes={} gqa_valid={}",
        device.name(),
        device_max,
        params.split_kv_single_q_kv_token_tile_size,
        params.split_kv_single_q_num_threads_per_threadblock,
        params.split_kv_single_q_max_q_head_tile_size,
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
    split_kv_variant: Option<&str>,
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
    let split_kv_variant_text = split_kv_variant
        .map(|value| format!(" split_kv_variant={value}"))
        .unwrap_or_default();
    println!(
        "perf component=gqa impl=full-forward-replay tokens={num_tokens} \
         num_reqs={num_reqs}{context_text}{split_kv_variant_text} iters={iters} runs={} median_us={median_us:.3} \
         samples_us=[{sample_text}]",
        samples.len()
    );
}

fn print_skip(
    num_tokens: u32,
    num_reqs: u32,
    existing_context_len: Option<u32>,
    split_kv_variant: Option<&str>,
    reason: &str,
) {
    let context_text = existing_context_len
        .map(|value| format!(" ctx={value}"))
        .unwrap_or_default();
    let split_kv_variant_text = split_kv_variant
        .map(|value| format!(" split_kv_variant={value}"))
        .unwrap_or_default();
    println!(
        "skip component=gqa tokens={num_tokens} num_reqs={num_reqs}{context_text}{split_kv_variant_text} \
         reason={reason}",
    );
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
    let mut recorder = MetalReplayRuntime::new(stream).create_recorder();
    recorder.record(ReplayOp::opaque(invocation));
    recorder.build()
}

fn gqa_qgkv_affine_config(model: GQAModelProfile) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig {
        n: model.qgkv_dim().try_into().expect("GQA qgkv n must fit i32"),
        k: model.hidden_dim.try_into().expect("GQA qgkv k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("GQA group size must fit i32"),
        bits: BITS.try_into().expect("GQA bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    }
}

fn gqa_output_affine_config(model: GQAModelProfile) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig {
        n: model.hidden_dim.try_into().expect("GQA output n must fit i32"),
        k: model.q_dim().try_into().expect("GQA output k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("GQA group size must fit i32"),
        bits: BITS.try_into().expect("GQA bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    }
}

fn gqa_qgkv_to_q_g_k_v_config(model: GQAModelProfile) -> GQAQGKVSplitConfig {
    GQAQGKVSplitConfig::bf16(
        model.num_q_heads.try_into().expect("GQA q heads must fit u32"),
        model.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        model.head_dim.try_into().expect("GQA head_dim must fit u32"),
    )
}

fn norm_rope_config(num_heads: usize, model: GQAModelProfile) -> RMSNormRopeConfig {
    RMSNormRopeConfig::bf16(
        num_heads.try_into().expect("GQA norm head count must fit u32"),
        model.head_dim.try_into().expect("GQA norm head_dim must fit u32"),
        GQA_ROPE_DIM,
        GQA_NORM_EPS,
        GQA_ROPE_THETA,
    )
}

fn norm_rope_shape(num_tokens: u32, _num_heads: usize, _model: GQAModelProfile) -> RMSNormRopeShape {
    RMSNormRopeShape {
        num_total_tokens: num_tokens,
    }
}

fn gqa_kv_page_write_config(model: GQAModelProfile, page_bytes: u32) -> GQAKVPageWriteConfig {
    GQAKVPageWriteConfig {
        num_kv_heads: model.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        head_dim: model.head_dim.try_into().expect("GQA head_dim must fit u32"),
        page_bytes,
        dtype: Dtype::Bfloat16,
    }
}

fn gqa_gate_config(model: GQAModelProfile) -> GQAActivationGateConfig {
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
) -> GQASplitKVSingleQConfig {
    GQASplitKVSingleQConfig {
        num_q_heads: model.num_q_heads.try_into().expect("GQA q heads must fit u32"),
        num_kv_heads: model.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        head_dim: model.head_dim.try_into().expect("GQA head_dim must fit u32"),
        scale: (model.head_dim as f32).sqrt().recip(),
        page_bytes: model.page_bytes(),
        page_table_layout: gqa_page_table_layout(num_reqs, end_context_len),
        kv_token_tile_size: params.split_kv_single_q_kv_token_tile_size,
        num_threads_per_threadblock: params.split_kv_single_q_num_threads_per_threadblock,
        q_head_tile_size: u32::try_from(model.num_q_heads / model.num_kv_heads)
            .expect("GQA q heads per KV head must fit u32")
            .min(params.split_kv_single_q_max_q_head_tile_size),
        dtype: Dtype::Bfloat16,
    }
}

fn gqa_sdpa_shape(replay_shape: GQAReplayShape) -> GQASplitKVSingleQShape {
    GQASplitKVSingleQShape {
        num_total_tokens: replay_shape.num_tokens,
        num_total_sdpa_map_task_templates: replay_shape.num_total_sdpa_map_task_templates,
    }
}

fn gqa_split_kv_tiled_q_config(
    page_table_layout: MetalGQAPageTableLayout,
    params: GQABenchParams,
    model: GQAModelProfile,
) -> GQASplitKVTiledQConfig {
    GQASplitKVTiledQConfig {
        num_q_heads: model.num_q_heads.try_into().expect("GQA q heads must fit u32"),
        num_kv_heads: model.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        head_dim: model.head_dim.try_into().expect("GQA head_dim must fit u32"),
        q_head_tile_size: params.split_kv_tiled_q_head_tile_size,
        q_token_tile_size: params.split_kv_tiled_q_token_tile_size,
        kv_token_tile_size: params.split_kv_tiled_q_kv_token_tile_size,
        scale: (model.head_dim as f32).sqrt().recip(),
        page_bytes: model.page_bytes(),
        dtype: Dtype::Bfloat16,
        page_table_layout,
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
