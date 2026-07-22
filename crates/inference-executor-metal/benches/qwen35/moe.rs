use std::fs::File;
use std::mem::size_of;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::MoECombineWithCommonShape;
use inference_backend_metal::components::MoEExpertMajorShape;
use inference_backend_metal::components::MoERoutingShape;
use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPScratch;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::components::QuantizedSparseMLPConfig;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorShape;
use inference_backend_metal::components::QuantizedSparseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::mlp::moe::GatedMoECore;
use inference_executor_core::mlp::moe::GatedMoEReplayShape;
use inference_executor_core::mlp::moe::MoEExecutionPolicy;
use inference_executor_core::mlp::moe::MoEExecutionPolicyConfig;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::mlp::dense::scratch::DenseMLPScratchBindings;
use inference_executor_metal::mlp::moe::backend::GatedMoE;
use inference_executor_metal::mlp::moe::backend::GatedMoECommonExpertReplayInput;
use inference_executor_metal::mlp::moe::backend::GatedMoECommonExpertWeights;
use inference_executor_metal::mlp::moe::backend::GatedMoEMetalConfig;
use inference_executor_metal::mlp::moe::backend::GatedMoEReplayInput;
use inference_executor_metal::mlp::moe::backend::GatedMoEWeights;
use inference_executor_metal::mlp::moe::scratch::CommonExpertScratchBindings;
use inference_executor_metal::mlp::moe::scratch::MoERoutingScratchBindings;
use inference_executor_metal::mlp::moe::scratch::MoEScratchBindings;
use inference_executor_metal::mlp::moe::scratch::TopKExpertsScratchBindings;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

const SHARD: &str = "model-00001-of-00004.safetensors";
const NUM_EXPERTS: u32 = 256;
const TOPK_EXPERTS: u32 = 8;
const HIDDEN_DIM: u32 = 2048;
const INTERMEDIATE_DIM: u32 = 512;
const GROUP_SIZE: u32 = 64;
const EXPERT_BITS: u32 = 4;
const ROUTER_BITS: u32 = 8;

fn main() {
    let args = Args::parse();

    let device = Device::system_default();
    let weights = RealMoEWeights::load(&device, &args.model_dir, args.model_layer_index);
    for num_tokens in args.num_tokens {
        if args.check_parity {
            let token_fixture = RealMoEFixture::new(
                &device,
                args.model_layer_index,
                num_tokens,
                &weights,
                MoERealImpl::TokenMajor,
            );
            let expert_fixture = RealMoEFixture::new(
                &device,
                args.model_layer_index,
                num_tokens,
                &weights,
                MoERealImpl::ExpertMajor,
            );
            token_fixture.run_replay();
            expert_fixture.run_replay();
            let token_bits = token_fixture.output_bits();
            let expert_bits = expert_fixture.output_bits();
            print_bitwise(
                args.model_layer_index,
                num_tokens,
                "token_major/expert_major",
                &token_bits,
                &expert_bits,
            );
        }

        for implementation in &args.implementations {
            let fixture = RealMoEFixture::new(&device, args.model_layer_index, num_tokens, &weights, *implementation);
            let samples = fixture.measure(*implementation, args.warmup_iters, args.iters, args.runs);
            print_perf(
                args.model_layer_index,
                implementation.key(),
                num_tokens,
                args.iters,
                &samples,
            );
        }
    }
}

struct Args {
    model_dir: PathBuf,
    model_layer_index: usize,
    num_tokens: Vec<u32>,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
    implementations: Vec<MoERealImpl>,
    check_parity: bool,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            model_layer_index: 0,
            num_tokens: vec![1, 2, 4, 8, 16, 32, 64],
            iters: 50,
            warmup_iters: 20,
            runs: 1,
            implementations: vec![MoERealImpl::TokenMajor, MoERealImpl::ExpertMajor],
            check_parity: false,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut iter, &arg)),
                "--layer" => args.model_layer_index = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--tokens" => args.num_tokens = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--impls" => args.implementations = parse_implementations(&next_arg(&mut iter, &arg)),
                "--check-parity" => args.check_parity = true,
                "--bench" => {},
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                },
                _ => panic!("unknown argument {arg:?}; pass --help for usage"),
            }
        }
        assert!(!args.num_tokens.is_empty(), "--tokens must include at least one value");
        assert!(
            !args.implementations.is_empty(),
            "--impls must select token_major, expert_major, or both"
        );
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MoERealImpl {
    TokenMajor,
    ExpertMajor,
}

impl MoERealImpl {
    fn key(self) -> &'static str {
        match self {
            Self::TokenMajor => "token_major",
            Self::ExpertMajor => "expert_major",
        }
    }

    fn policy(self) -> MoEExecutionPolicy {
        match self {
            Self::TokenMajor => MoEExecutionPolicy::TokenMajor,
            Self::ExpertMajor => MoEExecutionPolicy::ExpertMajor,
        }
    }
}

struct RealMoEFixture<'a> {
    stream: Stream,
    output: Buffer,
    replay: ReplayProgram,
    _input: Buffer,
    _router_logits: Buffer,
    _router_probs: Buffer,
    _expert_indices: Buffer,
    _expert_probs: Buffer,
    _token_indices: Buffer,
    _route_indices: Buffer,
    _expert_counts: Buffer,
    _expert_offsets: Buffer,
    _expert_cursors: Buffer,
    _routes_by_expert: Buffer,
    _routes_by_token: Buffer,
    _experts_by_route: Buffer,
    _packed_input: Buffer,
    _routed_hidden: Buffer,
    _sparse_activation: Buffer,
    _common_hidden: Buffer,
    _common_gate_logits: Buffer,
    _common_scratch: DenseMLPScratch,
    _weights: &'a RealMoEWeights,
}

impl<'a> RealMoEFixture<'a> {
    fn new(
        device: &Device,
        model_layer_index: usize,
        num_tokens: u32,
        weights: &'a RealMoEWeights,
        implementation: MoERealImpl,
    ) -> Self {
        let stream = Stream::new(device);
        let router_projection_shape = AffineQuantizedMatmulShape {
            m: num_tokens as i32,
            n: NUM_EXPERTS as i32,
            k: HIDDEN_DIM as i32,
            group_size: GROUP_SIZE as i32,
            bits: ROUTER_BITS as i32,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let common_gate_shape = AffineQuantizedMatmulShape {
            m: num_tokens as i32,
            n: 1,
            k: HIDDEN_DIM as i32,
            group_size: GROUP_SIZE as i32,
            bits: ROUTER_BITS as i32,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let routing_shape = MoERoutingShape {
            num_tokens,
            num_experts: NUM_EXPERTS,
            num_experts_per_token: TOPK_EXPERTS,
            norm_topk_prob: true,
        };
        let sparse_config = sparse_config();
        let dense_config = dense_config();
        let sparse_shape = QuantizedSparseMLPTokenMajorShape {
            num_routes: num_tokens * TOPK_EXPERTS,
            num_tokens,
        };
        let expert_major_shape = MoEExpertMajorShape::bf16(num_tokens, NUM_EXPERTS, TOPK_EXPERTS, HIDDEN_DIM);
        let expert_major_sparse_shape = QuantizedSparseMLPTokenMajorShape {
            num_routes: expert_major_shape.num_routes(),
            num_tokens: expert_major_shape.num_routes(),
        };
        let selected_sparse_shape = match implementation {
            MoERealImpl::TokenMajor => sparse_shape,
            MoERealImpl::ExpertMajor => expert_major_sparse_shape,
        };
        let dense_shape = QuantizedDenseMLPShape { num_tokens };
        let combine_shape = MoECombineWithCommonShape::bf16(num_tokens, TOPK_EXPERTS, HIDDEN_DIM);
        let num_routes = num_tokens as usize * TOPK_EXPERTS as usize;
        let input = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM as usize));
        let router_logits = Buffer::new_zeroed(device, router_projection_shape.output_bytes());
        let router_probs = Buffer::new_zeroed(device, router_projection_shape.output_bytes());
        let expert_indices = Buffer::new_zeroed(device, routing_shape.expert_indices_bytes());
        let expert_probs = Buffer::new_zeroed(device, routing_shape.expert_probs_bytes());
        let token_indices =
            Buffer::from_slice(device, &token_route_indices(num_tokens as usize, TOPK_EXPERTS as usize));
        let route_indices = Buffer::from_slice(device, &identity_indices(num_routes));
        let expert_counts = Buffer::new_zeroed(device, expert_major_shape.expert_counts_bytes());
        let expert_offsets = Buffer::new_zeroed(device, expert_major_shape.expert_offsets_bytes());
        let expert_cursors = Buffer::new_zeroed(device, expert_major_shape.expert_counts_bytes());
        let routes_by_expert = Buffer::new_zeroed(device, expert_major_shape.route_indices_bytes());
        let routes_by_token = Buffer::new_zeroed(device, expert_major_shape.route_indices_bytes());
        let experts_by_route = Buffer::new_zeroed(device, expert_major_shape.route_indices_bytes());
        let packed_input = Buffer::new_zeroed(device, expert_major_shape.route_hidden_bytes());
        let routed_hidden = Buffer::new_zeroed(device, sparse_config.token_major_output_bytes(selected_sparse_shape));
        let sparse_activation =
            Buffer::new_zeroed(device, sparse_config.activation_bytes(selected_sparse_shape.num_routes));
        let common_hidden = Buffer::new_zeroed(device, dense_config.down_shape(dense_shape).output_bytes());
        let common_gate_logits = Buffer::new_zeroed(device, common_gate_shape.output_bytes());
        let output = Buffer::new_zeroed(device, combine_shape.output_bytes());
        let common_scratch = DenseMLPScratch {
            gate_up_proj: Buffer::new_zeroed(
                device,
                num_tokens as usize * INTERMEDIATE_DIM as usize * 2 * Dtype::Bfloat16.item_size(),
            ),
            activation: Buffer::new_zeroed(device, dense_config.activation_shape(dense_shape).bytes()),
        };
        let core = GatedMoECore {
            model_layer_index,
            hidden_dim: HIDDEN_DIM as usize,
            intermediate_dim: INTERMEDIATE_DIM as usize,
            common_expert_intermediate_dim: Some(INTERMEDIATE_DIM as usize),
            num_experts: NUM_EXPERTS as usize,
            num_experts_per_token: TOPK_EXPERTS as usize,
            norm_topk_prob: true,
        };
        let backend = GatedMoE::new(device, core, moe_backend_config(implementation.policy()));
        let replay_shape = GatedMoEReplayShape { num_tokens };
        let mut builder = MetalReplayRuntime::new(&stream).create_recorder();
        let _ = <GatedMoE as ReplayLayer>::record(
            &backend,
            &mut builder,
            GatedMoEReplayInput {
                shape: replay_shape,
                hidden_state: &input,
                next_hidden_state: &output,
                scratch: MoEScratchBindings {
                    routing: MoERoutingScratchBindings {
                        router_logits: &router_logits,
                        router_probs: &router_probs,
                        expert_indices: &expert_indices,
                        expert_probs: &expert_probs,
                    },
                    topk_experts: TopKExpertsScratchBindings {
                        token_indices: &token_indices,
                        route_indices: &route_indices,
                        routed_hidden: &routed_hidden,
                        sparse_activation: &sparse_activation,
                        expert_counts: &expert_counts,
                        expert_offsets: &expert_offsets,
                        expert_cursors: &expert_cursors,
                        routes_by_expert: &routes_by_expert,
                        routes_by_token: &routes_by_token,
                        experts_by_route: &experts_by_route,
                        packed_input: &packed_input,
                    },
                },
                weights: weights.as_moe_weights(),
                common_expert: Some(GatedMoECommonExpertReplayInput {
                    scratch: common_scratch.as_common_scratch(&common_hidden, &common_gate_logits),
                    weights: weights.as_common_expert_weights(),
                }),
            },
        );
        let replay = builder.build();
        Self {
            stream,
            output,
            replay,
            _input: input,
            _router_logits: router_logits,
            _router_probs: router_probs,
            _expert_indices: expert_indices,
            _expert_probs: expert_probs,
            _token_indices: token_indices,
            _route_indices: route_indices,
            _expert_counts: expert_counts,
            _expert_offsets: expert_offsets,
            _expert_cursors: expert_cursors,
            _routes_by_expert: routes_by_expert,
            _routes_by_token: routes_by_token,
            _experts_by_route: experts_by_route,
            _packed_input: packed_input,
            _routed_hidden: routed_hidden,
            _sparse_activation: sparse_activation,
            _common_hidden: common_hidden,
            _common_gate_logits: common_gate_logits,
            _common_scratch: common_scratch,
            _weights: weights,
        }
    }

    fn run_replay(&self) {
        MetalReplayRuntime::new(&self.stream).submit_replay(&self.replay).wait();
    }

    fn measure(&self, implementation: MoERealImpl, warmup_iters: usize, iters: usize, runs: usize) -> Vec<f64> {
        measure_runs(runs, warmup_iters, iters, || self.run_impl(implementation))
    }

    fn run_impl(&self, implementation: MoERealImpl) {
        match implementation {
            MoERealImpl::TokenMajor | MoERealImpl::ExpertMajor => self.run_replay(),
        }
    }

    fn output_bits(&self) -> Vec<u16> {
        self.output
            .read_typed::<u16>(0, self.output.len_bytes() / size_of::<u16>())
    }
}

struct RealMoEWeights {
    router_weight: Buffer,
    router_scales: Buffer,
    router_biases: Buffer,
    common_gate_weight: Buffer,
    common_gate_scales: Buffer,
    common_gate_biases: Buffer,
    sparse: SparseMLPWeights,
    common: DenseMLPWeights,
}

impl RealMoEWeights {
    fn load(device: &Device, model_dir: &Path, model_layer_index: usize) -> Self {
        let shard_path = model_dir.join(SHARD);
        let mapped = MappedFile::open(&shard_path);
        let tensors = SafeTensors::deserialize(mapped.as_bytes()).unwrap_or_else(|err| {
            panic!(
                "unable to deserialize safetensors shard {}: {err:?}",
                shard_path.display()
            )
        });
        let router_weight = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "gate.weight"),
            safetensors::Dtype::U32,
        );
        let router_scales = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "gate.scales"),
            safetensors::Dtype::BF16,
        );
        let router_biases = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "gate.biases"),
            safetensors::Dtype::BF16,
        );
        let common_gate_weight = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert_gate.weight"),
            safetensors::Dtype::U32,
        );
        let common_gate_scales = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert_gate.scales"),
            safetensors::Dtype::BF16,
        );
        let common_gate_biases = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert_gate.biases"),
            safetensors::Dtype::BF16,
        );
        let sparse = SparseMLPWeights {
            gate_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.gate_proj.weight"),
                    safetensors::Dtype::U32,
                ),
            ),
            gate_scales: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.gate_proj.scales"),
                    safetensors::Dtype::BF16,
                ),
            ),
            gate_biases: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.gate_proj.biases"),
                    safetensors::Dtype::BF16,
                ),
            ),
            up_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.up_proj.weight"),
                    safetensors::Dtype::U32,
                ),
            ),
            up_scales: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.up_proj.scales"),
                    safetensors::Dtype::BF16,
                ),
            ),
            up_biases: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.up_proj.biases"),
                    safetensors::Dtype::BF16,
                ),
            ),
            down_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.down_proj.weight"),
                    safetensors::Dtype::U32,
                ),
            ),
            down_scales: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.down_proj.scales"),
                    safetensors::Dtype::BF16,
                ),
            ),
            down_biases: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.down_proj.biases"),
                    safetensors::Dtype::BF16,
                ),
            ),
        };
        let common_gate_weight_dense = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.gate_proj.weight"),
            safetensors::Dtype::U32,
        );
        let common_up_weight = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.up_proj.weight"),
            safetensors::Dtype::U32,
        );
        let common_gate_scales_dense = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.gate_proj.scales"),
            safetensors::Dtype::BF16,
        );
        let common_up_scales = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.up_proj.scales"),
            safetensors::Dtype::BF16,
        );
        let common_gate_biases_dense = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.gate_proj.biases"),
            safetensors::Dtype::BF16,
        );
        let common_up_biases = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.up_proj.biases"),
            safetensors::Dtype::BF16,
        );
        let common = DenseMLPWeights {
            gate_up_weight: Buffer::from_slice(device, &concat_bytes(&common_gate_weight_dense, &common_up_weight)),
            gate_up_scales: Buffer::from_slice(device, &concat_bytes(&common_gate_scales_dense, &common_up_scales)),
            gate_up_biases: Buffer::from_slice(device, &concat_bytes(&common_gate_biases_dense, &common_up_biases)),
            down_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "shared_expert.down_proj.weight"),
                    safetensors::Dtype::U32,
                ),
            ),
            down_scales: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "shared_expert.down_proj.scales"),
                    safetensors::Dtype::BF16,
                ),
            ),
            down_biases: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "shared_expert.down_proj.biases"),
                    safetensors::Dtype::BF16,
                ),
            ),
        };
        validate_weight_sizes(
            &router_weight,
            &router_scales,
            &router_biases,
            &common_gate_weight,
            &common_gate_scales,
            &common_gate_biases,
            &sparse,
            &common,
        );
        Self {
            router_weight: Buffer::from_slice(device, &router_weight),
            router_scales: Buffer::from_slice(device, &router_scales),
            router_biases: Buffer::from_slice(device, &router_biases),
            common_gate_weight: Buffer::from_slice(device, &common_gate_weight),
            common_gate_scales: Buffer::from_slice(device, &common_gate_scales),
            common_gate_biases: Buffer::from_slice(device, &common_gate_biases),
            sparse,
            common,
        }
    }

    fn as_moe_weights(&self) -> GatedMoEWeights<'_> {
        GatedMoEWeights {
            router_weight: &self.router_weight,
            router_scales: &self.router_scales,
            router_biases: &self.router_biases,
            topk_experts: self.sparse.as_borrowed(),
        }
    }

    fn as_common_expert_weights(&self) -> GatedMoECommonExpertWeights<'_> {
        GatedMoECommonExpertWeights {
            common_gate_weight: &self.common_gate_weight,
            common_gate_scales: &self.common_gate_scales,
            common_gate_biases: &self.common_gate_biases,
            common_expert: self.common.as_borrowed(),
        }
    }
}

struct SparseMLPWeights {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    up_weight: Buffer,
    up_scales: Buffer,
    up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl SparseMLPWeights {
    fn as_borrowed(&self) -> QuantizedSparseMLPWeights<'_> {
        QuantizedSparseMLPWeights {
            gate_weight: &self.gate_weight,
            gate_scales: &self.gate_scales,
            gate_biases: &self.gate_biases,
            up_weight: &self.up_weight,
            up_scales: &self.up_scales,
            up_biases: &self.up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

struct DenseMLPWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl DenseMLPWeights {
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

struct DenseMLPScratch {
    gate_up_proj: Buffer,
    activation: Buffer,
}

impl DenseMLPScratch {
    fn as_borrowed(&self) -> QuantizedDenseMLPScratch<'_> {
        QuantizedDenseMLPScratch {
            gate_up_proj: &self.gate_up_proj,
            activation: &self.activation,
        }
    }

    fn as_common_scratch<'a>(
        &'a self,
        common_hidden: &'a Buffer,
        common_gate_logits: &'a Buffer,
    ) -> CommonExpertScratchBindings<'a> {
        CommonExpertScratchBindings {
            hidden: common_hidden,
            gate_logits: common_gate_logits,
            dense_mlp: DenseMLPScratchBindings {
                gate_up_proj: &self.gate_up_proj,
                activation: &self.activation,
            },
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

fn tensor_name(model_layer_index: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{model_layer_index}.mlp.{suffix}")
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
    if name.ends_with("mlp.gate.weight") {
        assert_eq!(shape, &[NUM_EXPERTS as usize, packed_k_words(HIDDEN_DIM, ROUTER_BITS)]);
    } else if name.ends_with("mlp.gate.scales") || name.ends_with("mlp.gate.biases") {
        assert_eq!(shape, &[NUM_EXPERTS as usize, groups(HIDDEN_DIM)]);
    } else if name.ends_with("shared_expert_gate.weight") {
        assert_eq!(shape, &[1, packed_k_words(HIDDEN_DIM, ROUTER_BITS)]);
    } else if name.ends_with("shared_expert_gate.scales") || name.ends_with("shared_expert_gate.biases") {
        assert_eq!(shape, &[1, groups(HIDDEN_DIM)]);
    } else if name.ends_with("switch_mlp.gate_proj.weight") || name.ends_with("switch_mlp.up_proj.weight") {
        assert_eq!(
            shape,
            &[
                NUM_EXPERTS as usize,
                INTERMEDIATE_DIM as usize,
                packed_k_words(HIDDEN_DIM, EXPERT_BITS)
            ]
        );
    } else if name.ends_with("switch_mlp.gate_proj.scales")
        || name.ends_with("switch_mlp.gate_proj.biases")
        || name.ends_with("switch_mlp.up_proj.scales")
        || name.ends_with("switch_mlp.up_proj.biases")
    {
        assert_eq!(
            shape,
            &[NUM_EXPERTS as usize, INTERMEDIATE_DIM as usize, groups(HIDDEN_DIM)]
        );
    } else if name.ends_with("switch_mlp.down_proj.weight") {
        assert_eq!(
            shape,
            &[
                NUM_EXPERTS as usize,
                HIDDEN_DIM as usize,
                packed_k_words(INTERMEDIATE_DIM, EXPERT_BITS)
            ]
        );
    } else if name.ends_with("switch_mlp.down_proj.scales") || name.ends_with("switch_mlp.down_proj.biases") {
        assert_eq!(
            shape,
            &[NUM_EXPERTS as usize, HIDDEN_DIM as usize, groups(INTERMEDIATE_DIM)]
        );
    } else if name.ends_with("shared_expert.gate_proj.weight") || name.ends_with("shared_expert.up_proj.weight") {
        assert_eq!(
            shape,
            &[INTERMEDIATE_DIM as usize, packed_k_words(HIDDEN_DIM, EXPERT_BITS)]
        );
    } else if name.ends_with("shared_expert.gate_proj.scales")
        || name.ends_with("shared_expert.gate_proj.biases")
        || name.ends_with("shared_expert.up_proj.scales")
        || name.ends_with("shared_expert.up_proj.biases")
    {
        assert_eq!(shape, &[INTERMEDIATE_DIM as usize, groups(HIDDEN_DIM)]);
    } else if name.ends_with("shared_expert.down_proj.weight") {
        assert_eq!(
            shape,
            &[HIDDEN_DIM as usize, packed_k_words(INTERMEDIATE_DIM, EXPERT_BITS)]
        );
    } else if name.ends_with("shared_expert.down_proj.scales") || name.ends_with("shared_expert.down_proj.biases") {
        assert_eq!(shape, &[HIDDEN_DIM as usize, groups(INTERMEDIATE_DIM)]);
    } else {
        panic!("unexpected MoE tensor name {name}");
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_weight_sizes(
    router_weight: &[u8],
    router_scales: &[u8],
    router_biases: &[u8],
    common_gate_weight: &[u8],
    common_gate_scales: &[u8],
    common_gate_biases: &[u8],
    sparse: &SparseMLPWeights,
    common: &DenseMLPWeights,
) {
    let router_shape = AffineQuantizedMatmulShape {
        m: 1,
        n: NUM_EXPERTS as i32,
        k: HIDDEN_DIM as i32,
        group_size: GROUP_SIZE as i32,
        bits: ROUTER_BITS as i32,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        affine_dtype: Dtype::Bfloat16,
    };
    let common_gate_shape = AffineQuantizedMatmulShape {
        m: 1,
        n: 1,
        k: HIDDEN_DIM as i32,
        group_size: GROUP_SIZE as i32,
        bits: ROUTER_BITS as i32,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        affine_dtype: Dtype::Bfloat16,
    };
    assert_eq!(router_weight.len(), router_shape.weight_bytes());
    assert_eq!(router_scales.len(), router_shape.affine_param_bytes());
    assert_eq!(router_biases.len(), router_shape.affine_param_bytes());
    assert_eq!(common_gate_weight.len(), common_gate_shape.weight_bytes());
    assert_eq!(common_gate_scales.len(), common_gate_shape.affine_param_bytes());
    assert_eq!(common_gate_biases.len(), common_gate_shape.affine_param_bytes());
    let sparse_config = sparse_config();
    let sparse_shape = QuantizedSparseMLPTokenMajorShape {
        num_routes: TOPK_EXPERTS,
        num_tokens: 1,
    };
    let gate_up_shape = sparse_config.token_major_fused_gate_up_silu_shape(sparse_shape);
    let down_shape = sparse_config.token_major_down_shape(sparse_shape);
    assert_eq!(
        sparse.gate_weight.len_bytes(),
        NUM_EXPERTS as usize * gate_up_shape.weight_bytes_per_expert()
    );
    assert_eq!(
        sparse.gate_scales.len_bytes(),
        NUM_EXPERTS as usize * gate_up_shape.affine_param_bytes_per_expert()
    );
    assert_eq!(
        sparse.gate_biases.len_bytes(),
        NUM_EXPERTS as usize * gate_up_shape.affine_param_bytes_per_expert()
    );
    assert_eq!(sparse.up_weight.len_bytes(), sparse.gate_weight.len_bytes());
    assert_eq!(sparse.up_scales.len_bytes(), sparse.gate_scales.len_bytes());
    assert_eq!(sparse.up_biases.len_bytes(), sparse.gate_biases.len_bytes());
    assert_eq!(
        sparse.down_weight.len_bytes(),
        NUM_EXPERTS as usize * down_shape.weight_bytes_per_expert()
    );
    assert_eq!(
        sparse.down_scales.len_bytes(),
        NUM_EXPERTS as usize * down_shape.affine_param_bytes_per_expert()
    );
    assert_eq!(
        sparse.down_biases.len_bytes(),
        NUM_EXPERTS as usize * down_shape.affine_param_bytes_per_expert()
    );
    let dense_config = dense_config();
    let dense_shape = QuantizedDenseMLPShape { num_tokens: 1 };
    let dense_gate_up_shape = dense_config.gate_up_shape(dense_shape);
    let dense_down_shape = dense_config.down_shape(dense_shape);
    assert_eq!(common.gate_up_weight.len_bytes(), dense_gate_up_shape.weight_bytes());
    assert_eq!(
        common.gate_up_scales.len_bytes(),
        dense_gate_up_shape.affine_param_bytes()
    );
    assert_eq!(
        common.gate_up_biases.len_bytes(),
        dense_gate_up_shape.affine_param_bytes()
    );
    assert_eq!(common.down_weight.len_bytes(), dense_down_shape.weight_bytes());
    assert_eq!(common.down_scales.len_bytes(), dense_down_shape.affine_param_bytes());
    assert_eq!(common.down_biases.len_bytes(), dense_down_shape.affine_param_bytes());
}

fn sparse_config() -> QuantizedSparseMLPConfig {
    QuantizedSparseMLPConfig {
        hidden_dim: HIDDEN_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        group_size: GROUP_SIZE,
        bits: EXPERT_BITS,
        dtype: Dtype::Bfloat16,
    }
}

fn dense_config() -> QuantizedDenseMLPConfig {
    QuantizedDenseMLPConfig {
        hidden_dim: HIDDEN_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        group_size: GROUP_SIZE,
        bits: EXPERT_BITS,
        dtype: Dtype::Bfloat16,
    }
}

fn moe_backend_config(policy: MoEExecutionPolicy) -> GatedMoEMetalConfig {
    GatedMoEMetalConfig {
        group_size: GROUP_SIZE,
        bits: EXPERT_BITS,
        router_bits: ROUTER_BITS,
        common_gate_bits: ROUTER_BITS,
        dtype: Dtype::Bfloat16,
        execution_policy: MoEExecutionPolicyConfig::new(policy),
    }
}

fn concat_bytes(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend_from_slice(left);
    out.extend_from_slice(right);
    out
}

fn hidden_fixture(num_tokens: usize, hidden_dim: usize) -> Vec<u16> {
    (0..num_tokens * hidden_dim)
        .map(|index| bf16::from_f32(((index % 23) as f32 - 11.0) * 0.03125).to_bits())
        .collect()
}

fn token_route_indices(num_tokens: usize, topk_experts: usize) -> Vec<u32> {
    (0..num_tokens * topk_experts)
        .map(|route| u32::try_from(route / topk_experts).expect("token route index must fit u32"))
        .collect()
}

fn identity_indices(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| u32::try_from(index).expect("identity index must fit u32"))
        .collect()
}

fn packed_k_words(k: u32, bits: u32) -> usize {
    (k as usize * bits as usize) / 32
}

fn groups(k: u32) -> usize {
    (k / GROUP_SIZE) as usize
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

fn parse_usize(value: &str, flag: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {flag} value {value:?}: {err}"))
}

fn parse_implementations(value: &str) -> Vec<MoERealImpl> {
    let mut impls = Vec::new();
    for part in value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
    {
        match part {
            "both" => {
                impls.push(MoERealImpl::TokenMajor);
                impls.push(MoERealImpl::ExpertMajor);
            },
            "token_major" => impls.push(MoERealImpl::TokenMajor),
            "expert_major" => impls.push(MoERealImpl::ExpertMajor),
            other => panic!("unknown --impls entry {other:?}; use token_major, expert_major, or both"),
        }
    }
    impls.dedup();
    assert!(
        !impls.is_empty(),
        "--impls must select token_major, expert_major, or both"
    );
    impls
}

fn parse_list(value: &str) -> Vec<&str> {
    let values = value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    assert!(!values.is_empty(), "list argument must contain at least one value");
    values
}

fn print_help() {
    println!(
        "\
moe real-weight replay bench

Options:
  --model-dir PATH
  --layer N
  --tokens 1,2,4,8,16,32,64
  --impls token_major,expert_major,both
  --check-parity
  --iters N
  --warmup-iters N
  --runs N
"
    );
}

fn print_bitwise(model_layer_index: usize, num_tokens: u32, implementation: &str, baseline: &[u16], actual: &[u16]) {
    let first_mismatch = baseline
        .iter()
        .zip(actual.iter())
        .position(|(left, right)| left != right)
        .unwrap_or(0);
    let (lhs, rhs) = baseline
        .get(first_mismatch)
        .zip(actual.get(first_mismatch))
        .map(|(left, right)| (*left, *right))
        .unwrap_or((0, 0));
    println!(
        "bitwise component=moe-real impl={implementation} layer={model_layer_index} num_tokens={num_tokens} \
         num_values={} equal={} first_mismatch={first_mismatch} lhs=0x{lhs:04x} rhs=0x{rhs:04x}",
        baseline.len(),
        baseline == actual,
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

fn print_perf(model_layer_index: usize, implementation: &str, num_tokens: u32, iters: usize, samples: &[f64]) {
    let median_us = median(samples);
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=moe-real impl={implementation} layer={model_layer_index} num_tokens={num_tokens} \
         iters={iters} runs={} median_us={median_us:.3} samples_us=[{sample_text}]",
        samples.len()
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
