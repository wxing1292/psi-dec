use std::mem::size_of;

use inference_backend_metal::components::GQAActivationGateBuffers;
use inference_backend_metal::components::GQAActivationGateKernel;
use inference_backend_metal::components::GQAActivationGateShape;
use inference_backend_metal::components::GQAComputePath;
use inference_backend_metal::components::GQAKVPageWrite;
use inference_backend_metal::components::GQAKVPageWriteBuffers;
use inference_backend_metal::components::GQAKVPageWriteShape;
use inference_backend_metal::components::GQAPagedSDPAKernels;
use inference_backend_metal::components::GQAPagedSDPAMapBuffers;
use inference_backend_metal::components::GQAPagedSDPAReduceBuffers;
use inference_backend_metal::components::GQAQGKVSplitBuffers;
use inference_backend_metal::components::GQAQGKVSplitKernel;
use inference_backend_metal::components::GQAQGKVSplitShape;
use inference_backend_metal::components::GQATiledSDPAKernels;
use inference_backend_metal::components::GQATiledSDPAMapBuffers;
use inference_backend_metal::components::GQATiledSDPAReduceBuffers;
use inference_backend_metal::components::GQATiledSDPAShape;
use inference_backend_metal::components::RMSNormRopeBuffers;
use inference_backend_metal::components::RMSNormRopeKernel;
use inference_backend_metal::components::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_metal::attn::gqa::backend::GQA;
use inference_executor_metal::attn::gqa::backend::GQAInput;
use inference_executor_metal::attn::gqa::backend::GQAKVCacheBindings;
use inference_executor_metal::attn::gqa::backend::GQAMetalConfig;
use inference_executor_metal::attn::gqa::backend::GQAReplayMode;
use inference_executor_metal::attn::gqa::backend::GQAWeights;
use inference_executor_metal::attn::gqa::batch_metadata::GQAMetadataBuffers;
use inference_executor_metal::attn::gqa::scratch::GQAScratchBindings;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::def::replay_op::ReplayOp;
use safetensors::SafeTensors;

use crate::Args;
use crate::BITS;
use crate::GQA_MAX_TOKENS;
use crate::GQA_NORM_EPS;
use crate::GQA_ROPE_DIM;
use crate::GQA_ROPE_THETA;
use crate::GQABenchParams;
use crate::GQABenchPath;
use crate::GQAModelProfile;
use crate::GROUP_SIZE;
use crate::TOKENS_PER_PAGE;
use crate::assert_bf16_close;
use crate::build_single_invocation_replay;
use crate::cu_tokens;
use crate::gqa_attention_reference_at;
use crate::gqa_gate_config;
use crate::gqa_kv_page_write_config;
use crate::gqa_output_affine_config;
use crate::gqa_page_table_layout;
use crate::gqa_qgkv_affine_config;
use crate::gqa_qgkv_to_q_g_k_v_config;
use crate::gqa_sdpa_config;
use crate::gqa_sdpa_shape;
use crate::gqa_tiled_sdpa_config;
use crate::hidden_fixture;
use crate::max_bf16_diff;
use crate::measure_runs;
use crate::norm_rope_config;
use crate::norm_rope_shape;
use crate::page_table;
use crate::print_gqa_kernel_limits;
use crate::print_named_perf;
use crate::print_perf;
use crate::print_skip;
use crate::request_token_counts;
use crate::valid_num_reqs;

mod weights;
use weights::MappedFile;
use weights::RealGQAWeights;

pub fn run(args: Args) {
    let device = Device::system_default();
    if args.print_limits {
        print_gqa_kernel_limits(&device, args.params);
    }
    let mapped = MappedFile::open(&args.model_dir.join(args.model.shard));
    let tensors = SafeTensors::deserialize(mapped.as_bytes()).unwrap_or_else(|err| {
        panic!(
            "unable to deserialize safetensors shard {}: {err:?}",
            args.model_dir.join(args.model.shard).display()
        )
    });
    let weights = RealGQAWeights::load(&device, &tensors, args.model);
    let contexts = if args.contexts.is_empty() {
        vec![0]
    } else {
        args.contexts.clone()
    };
    let token_counts = args
        .tokens_per_req
        .as_ref()
        .map(|tokens_per_req| vec![tokens_per_req.iter().sum()])
        .unwrap_or_else(|| args.tokens.clone());

    for num_tokens in token_counts {
        if let Some(tokens_per_req) = &args.tokens_per_req {
            for &existing_context_len in &contexts {
                let fixture = RealGQAFixture::new(
                    &device,
                    tokens_per_req.clone(),
                    existing_context_len,
                    &weights,
                    args.params,
                    args.model,
                );
                if args.validate_tiled_q_tokens {
                    fixture.validate_tiled_q_tokens_attention();
                }
                fixture.measure(&args.paths, args.warmup_iters, args.iters, args.runs);
                if args.subcomponents {
                    fixture.measure_subcomponents(
                        &args.selected_subcomponents,
                        args.warmup_iters,
                        args.iters,
                        args.runs,
                    );
                }
            }
            continue;
        }

        for &num_reqs in &args.num_reqs {
            if !valid_num_reqs(num_tokens, num_reqs) {
                print_skip(num_tokens, num_reqs, None, None, "num_reqs_exceeds_tokens");
                continue;
            }
            for &existing_context_len in &contexts {
                let fixture = RealGQAFixture::new(
                    &device,
                    request_token_counts(num_tokens, num_reqs),
                    existing_context_len,
                    &weights,
                    args.params,
                    args.model,
                );
                if args.validate_tiled_q_tokens {
                    fixture.validate_tiled_q_tokens_attention();
                }
                fixture.measure(&args.paths, args.warmup_iters, args.iters, args.runs);
                if args.subcomponents {
                    fixture.measure_subcomponents(
                        &args.selected_subcomponents,
                        args.warmup_iters,
                        args.iters,
                        args.runs,
                    );
                }
            }
        }
    }
}

struct RealGQAFixture<'a> {
    device: Device,
    stream: Stream,
    model: GQAModelProfile,
    params: GQABenchParams,
    num_tokens: u32,
    num_reqs: u32,
    num_tokens_per_req: Vec<u32>,
    existing_context_len: u32,
    end_context_len: u32,
    next_hidden_state: Buffer,
    replay: ReplayProgram,
    tiled_next_hidden_state: Buffer,
    tiled_replay: ReplayProgram,
    _hidden_state: Buffer,
    _kv_pages: Buffer,
    batch_metadata: GQAMetadataBuffers,
    tiled_batch_metadata: GQAMetadataBuffers,
    _page_ids: Buffer,
    _tiled_partial_output: Buffer,
    _tiled_partial_exp_sums: Buffer,
    _tiled_partial_max_logits: Buffer,
    _qgkv: Buffer,
    _q: Buffer,
    _g: Buffer,
    _k: Buffer,
    _v: Buffer,
    _q_norm_rope: Buffer,
    _k_norm_rope: Buffer,
    _sdpa_partial_exp_sums: Buffer,
    _sdpa_partial_max_logits: Buffer,
    _sdpa_partial_output: Buffer,
    _attention_output: Buffer,
    _tiled_attention_output: Buffer,
    _gated_attention_output: Buffer,
    _weights: &'a RealGQAWeights,
}

impl<'a> RealGQAFixture<'a> {
    fn new(
        device: &Device,
        num_tokens_per_req: Vec<u32>,
        existing_context_len: u32,
        weights: &'a RealGQAWeights,
        mut params: GQABenchParams,
        model: GQAModelProfile,
    ) -> Self {
        assert!(
            !num_tokens_per_req.is_empty(),
            "GQA bench requires at least one request"
        );
        assert!(
            num_tokens_per_req.iter().all(|&num_req_tokens| num_req_tokens > 0),
            "GQA bench token counts per request must be positive"
        );
        let num_tokens = num_tokens_per_req.iter().sum::<u32>();
        let num_reqs = num_tokens_per_req
            .len()
            .try_into()
            .expect("GQA request count must fit u32");
        assert!(
            num_tokens as usize <= GQA_MAX_TOKENS,
            "GQA bench token capacity exceeded"
        );
        let stream = Stream::new(device);
        let core = GQACore::new(
            model.model_layer_index,
            model.hidden_dim,
            model.head_dim,
            model.num_q_heads,
            model.num_kv_heads,
            (model.head_dim as f32).sqrt().recip(),
        );
        let config = GQAMetalConfig {
            group_size: GROUP_SIZE,
            bits: BITS,
            page_bytes: model.page_bytes(),
            rope_dim: GQA_ROPE_DIM,
            norm_eps: GQA_NORM_EPS,
            rope_theta: GQA_ROPE_THETA,
            rope_scaling: RopeScaling::Default,
            io_dtype: Dtype::Bfloat16,
        };
        let backend = GQA::new(device, core, config);
        let max_num_req_tokens = num_tokens_per_req.iter().copied().max().unwrap_or(0);
        let end_context_len = existing_context_len + max_num_req_tokens;
        let num_blocks = end_context_len.div_ceil(TOKENS_PER_PAGE).max(1);
        let page_table_layout = GQAPageTableLayout {
            num_req_slots: num_reqs,
            num_blocks,
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        };
        let hidden_state = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, model.hidden_dim));
        let kv_pages = Buffer::new_zeroed(
            device,
            num_reqs as usize * num_blocks as usize * config.page_bytes as usize,
        );
        assert!(num_tokens as usize <= GQA_MAX_TOKENS);
        let q_heads_per_kv_head = model.num_q_heads / model.num_kv_heads;
        let single_q_head_tile_size = q_heads_per_kv_head
            .min(params.single_q_token_max_q_head_tile_size as usize)
            .try_into()
            .expect("GQA Q-head tile size must fit u32");
        let num_q_token_tiles = num_tokens_per_req
            .iter()
            .map(|&count| count.div_ceil(params.tiled_q_token_tile_size))
            .sum::<u32>();
        if params.tiled_q_head_tile_size == 0 {
            let desired_q_head_tile_size = if (num_tokens as u64) < 4 * num_q_token_tiles as u64 {
                q_heads_per_kv_head.div_ceil(2)
            } else {
                q_heads_per_kv_head
            };
            let max_q_head_tile_size = 256 / (params.tiled_q_token_tile_size / 8 * 32);
            params.tiled_q_head_tile_size = desired_q_head_tile_size
                .min(max_q_head_tile_size as usize)
                .try_into()
                .expect("GQA Q-head tile size must fit u32");
        }
        let batch_metadata = GQAMetadataBuffers::new(device, GQA_MAX_TOKENS);
        let shape = batch_metadata.update(
            &(0..num_reqs).collect::<Vec<_>>(),
            &vec![existing_context_len; num_reqs as usize],
            &cu_tokens(&num_tokens_per_req)
                .into_iter()
                .map(|value| value as u32)
                .collect::<Vec<_>>(),
            GQAComputePath::SingleQueryToken {
                kv_token_tile_size: params.single_q_token_kv_token_tile_size,
                num_threads_per_threadblock: params.single_q_token_num_threads_per_threadblock,
                q_head_tile_size: single_q_head_tile_size,
            },
        );
        let page_ids = Buffer::from_slice(device, &page_table(num_reqs, num_blocks));
        let tiled_batch_metadata = GQAMetadataBuffers::new(device, GQA_MAX_TOKENS);
        let tiled_replay_shape = tiled_batch_metadata.update(
            &(0..num_reqs).collect::<Vec<_>>(),
            &vec![existing_context_len; num_reqs as usize],
            &cu_tokens(&num_tokens_per_req)
                .into_iter()
                .map(|value| value as u32)
                .collect::<Vec<_>>(),
            GQAComputePath::TiledQueryTokens {
                q_token_tile_size: params.tiled_q_token_tile_size,
                kv_token_tile_size: params.tiled_kv_token_tile_size,
                q_head_tile_size: params.tiled_q_head_tile_size,
            },
        );
        let qgkv = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.qgkv_dim() * Dtype::Bfloat16.item_size(),
        );
        let q = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let g = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let k = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.kv_dim() * Dtype::Bfloat16.item_size(),
        );
        let v = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.kv_dim() * Dtype::Bfloat16.item_size(),
        );
        let q_norm_rope = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let k_norm_rope = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.kv_dim() * Dtype::Bfloat16.item_size(),
        );
        let num_sdpa_partial_outputs = shape.num_total_sdpa_map_task_templates as usize * model.num_q_heads;
        let sdpa_partial_exp_sums = Buffer::new_zeroed(device, num_sdpa_partial_outputs * size_of::<f32>());
        let sdpa_partial_max_logits = Buffer::new_zeroed(device, num_sdpa_partial_outputs * size_of::<f32>());
        let sdpa_partial_output = Buffer::new_zeroed(
            device,
            num_sdpa_partial_outputs * model.head_dim * Dtype::Bfloat16.item_size(),
        );
        let attention_output = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let tiled_attention_output = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let num_sdpa_partial_output_tokens =
            tiled_replay_shape.num_total_sdpa_map_task_templates as usize * params.tiled_q_token_tile_size as usize;
        let tiled_partial_output = Buffer::new_zeroed(
            device,
            num_sdpa_partial_output_tokens * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let tiled_partial_exp_sums = Buffer::new_zeroed(
            device,
            num_sdpa_partial_output_tokens * model.num_q_heads * size_of::<f32>(),
        );
        let tiled_partial_max_logits = Buffer::new_zeroed(
            device,
            num_sdpa_partial_output_tokens * model.num_q_heads * size_of::<f32>(),
        );
        let gated_attention_output = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let next_hidden_state = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.hidden_dim * Dtype::Bfloat16.item_size(),
        );
        let tiled_next_hidden_state = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.hidden_dim * Dtype::Bfloat16.item_size(),
        );
        let mut recorder = MetalReplayRuntime::new(&stream).create_recorder();
        let _ = <GQA as ReplayLayer>::record(
            &backend,
            &mut recorder,
            GQAInput {
                page_table_layout,
                gqa_layer_index: ReplayU32::Fixed(0),
                batch_metadata: &batch_metadata,
                hidden_state: &hidden_state,
                next_hidden_state: &next_hidden_state,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: &kv_pages,
                    page_ids: &page_ids,
                },
                weights: GQAWeights {
                    qgkv_weight: &weights.qgkv_weight,
                    qgkv_scales: &weights.qgkv_scales,
                    qgkv_biases: &weights.qgkv_biases,
                    q_norm_weight: &weights.q_norm_weight,
                    k_norm_weight: &weights.k_norm_weight,
                    output_weight: &weights.output_weight,
                    output_scales: &weights.output_scales,
                    output_biases: &weights.output_biases,
                },
                scratch: GQAScratchBindings {
                    qgkv: &qgkv,
                    q: &q,
                    g: &g,
                    k: &k,
                    v: &v,
                    q_norm_rope: &q_norm_rope,
                    k_norm_rope: &k_norm_rope,
                    sdpa_partial_exp_sums: &sdpa_partial_exp_sums,
                    sdpa_partial_max_logits: &sdpa_partial_max_logits,
                    sdpa_partial_output: &sdpa_partial_output,
                    attention_output: &attention_output,
                    gated_attention_output: &gated_attention_output,
                },
                replay_mode: GQAReplayMode::Exact,
            },
        );
        let replay = recorder.build();
        let qgkv_matmul = AffineQuantizedMatmul::new(device, gqa_qgkv_affine_config(model));
        let qgkv_to_q_g_k_v = GQAQGKVSplitKernel::new(device, gqa_qgkv_to_q_g_k_v_config(model));
        let q_norm_rope_kernel = RMSNormRopeKernel::new(device, norm_rope_config(model.num_q_heads, model));
        let k_norm_rope_kernel = RMSNormRopeKernel::new(device, norm_rope_config(model.num_kv_heads, model));
        let kv_page_write = GQAKVPageWrite::new(device, gqa_kv_page_write_config(model, config.page_bytes));
        let gate = GQAActivationGateKernel::new(device, gqa_gate_config(model));
        let output = AffineQuantizedMatmul::new(device, gqa_output_affine_config(model));
        let metal_page_table_layout = gqa_page_table_layout(num_reqs, end_context_len);
        let tiled_config = gqa_tiled_sdpa_config(metal_page_table_layout, params, model);
        let tiled_shape = GQATiledSDPAShape {
            num_total_tokens: num_tokens,
            num_total_q_token_tiles: tiled_replay_shape.num_q_token_tiles,
            num_total_sdpa_map_task_templates: tiled_replay_shape.num_total_sdpa_map_task_templates,
        };
        let tiled_kernel = GQATiledSDPAKernels::new(device, tiled_config, tiled_shape);
        let mut tiled_builder = MetalReplayRuntime::new(&stream).create_recorder();
        tiled_builder.record_with_barrier_before(ReplayOp::opaque(qgkv_matmul.invoke(
            num_tokens.try_into().expect("GQA token count must fit i32"),
            &qgkv,
            0,
            &hidden_state,
            0,
            &weights.qgkv_weight,
            0,
            &weights.qgkv_scales,
            0,
            &weights.qgkv_biases,
            0,
        )));
        tiled_builder.record_with_barrier_before(ReplayOp::opaque(qgkv_to_q_g_k_v.invoke(
            GQAQGKVSplitShape {
                num_total_tokens: num_tokens,
            },
            GQAQGKVSplitBuffers {
                qgkv: &qgkv,
                q: &q,
                g: &g,
                k: &k,
                v: &v,
            },
        )));
        tiled_builder.record_with_barrier_before(ReplayOp::opaque(q_norm_rope_kernel.invoke(
            norm_rope_shape(num_tokens, model.num_q_heads, model),
            RMSNormRopeBuffers {
                input: &q,
                norm_weight: &weights.q_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: &q_norm_rope,
            },
        )));
        tiled_builder.record(ReplayOp::opaque(k_norm_rope_kernel.invoke(
            norm_rope_shape(num_tokens, model.num_kv_heads, model),
            RMSNormRopeBuffers {
                input: &k,
                norm_weight: &weights.k_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: &k_norm_rope,
            },
        )));
        tiled_builder.record_with_barrier_before(ReplayOp::opaque(kv_page_write.invoke(
            GQAKVPageWriteShape {
                num_total_token_writes: num_tokens,
                page_table_layout: metal_page_table_layout,
            },
            GQAKVPageWriteBuffers {
                pages: &kv_pages,
                flat_k: &k_norm_rope,
                flat_v: &v,
                req_slots: tiled_batch_metadata.req_slots(),
                flat_token_indices: batch_metadata.flat_token_indices(),
                page_ids: &page_ids,
            },
            ReplayU32::Fixed(0),
        )));
        tiled_builder.record_with_barrier_before(ReplayOp::opaque(tiled_kernel.invoke_map(
            GQATiledSDPAMapBuffers {
                q: &q_norm_rope,
                kv_pages: &kv_pages,
                req_slots: batch_metadata.req_slots(),
                page_ids: &page_ids,
                flat_token_indices: tiled_batch_metadata.flat_token_indices(),
                q_token_tiles: tiled_batch_metadata.q_token_tiles(),
                sdpa_map_task_templates: tiled_batch_metadata.sdpa_map_task_templates(),
                partial_output: &tiled_partial_output,
                partial_exp_sums: &tiled_partial_exp_sums,
                partial_max_logits: &tiled_partial_max_logits,
            },
            ReplayU32::Fixed(0),
        )));
        tiled_builder.record_with_barrier_before(ReplayOp::opaque(tiled_kernel.invoke_reduce(
            GQATiledSDPAReduceBuffers {
                partial_output: &tiled_partial_output,
                partial_exp_sums: &tiled_partial_exp_sums,
                partial_max_logits: &tiled_partial_max_logits,
                q_token_tiles: tiled_batch_metadata.q_token_tiles(),
                cu_sdpa_partial_outputs: tiled_batch_metadata.cu_sdpa_partial_outputs(),
                output: &tiled_attention_output,
            },
        )));
        tiled_builder.record_with_barrier_before(ReplayOp::opaque(gate.invoke(
            GQAActivationGateShape {
                num_total_tokens: num_tokens,
            },
            GQAActivationGateBuffers {
                attention_output: &tiled_attention_output,
                g: &g,
                output: &gated_attention_output,
            },
        )));
        tiled_builder.record_with_barrier_before(ReplayOp::opaque(output.invoke(
            num_tokens.try_into().expect("GQA token count must fit i32"),
            &tiled_next_hidden_state,
            0,
            &gated_attention_output,
            0,
            &weights.output_weight,
            0,
            &weights.output_scales,
            0,
            &weights.output_biases,
            0,
        )));
        let tiled_replay = tiled_builder.build();
        Self {
            device: device.clone(),
            stream,
            model,
            params,
            num_tokens,
            num_reqs,
            num_tokens_per_req,
            existing_context_len,
            end_context_len,
            next_hidden_state,
            replay,
            tiled_next_hidden_state,
            tiled_replay,
            _hidden_state: hidden_state,
            _kv_pages: kv_pages,
            batch_metadata,
            tiled_batch_metadata,
            _page_ids: page_ids,
            _tiled_partial_output: tiled_partial_output,
            _tiled_partial_exp_sums: tiled_partial_exp_sums,
            _tiled_partial_max_logits: tiled_partial_max_logits,
            _qgkv: qgkv,
            _q: q,
            _g: g,
            _k: k,
            _v: v,
            _q_norm_rope: q_norm_rope,
            _k_norm_rope: k_norm_rope,
            _sdpa_partial_exp_sums: sdpa_partial_exp_sums,
            _sdpa_partial_max_logits: sdpa_partial_max_logits,
            _sdpa_partial_output: sdpa_partial_output,
            _attention_output: attention_output,
            _tiled_attention_output: tiled_attention_output,
            _gated_attention_output: gated_attention_output,
            _weights: weights,
        }
    }

    fn run_single_q_token(&self) {
        MetalReplayRuntime::new(&self.stream).submit_replay(&self.replay).wait();
    }

    fn run_tiled_q_tokens(&self) {
        MetalReplayRuntime::new(&self.stream)
            .submit_replay(&self.tiled_replay)
            .wait();
    }

    fn validate_tiled_q_tokens_attention(&self) {
        let sdpa_config = gqa_sdpa_config(self.num_reqs, self.end_context_len, self.params, self.model);
        let sdpa_shape = gqa_sdpa_shape(self.batch_metadata.replay_shape());
        let sdpa = GQAPagedSDPAKernels::new(&self.device, sdpa_config, sdpa_shape);
        let single_q_token_replay = {
            let mut recorder = MetalReplayRuntime::new(&self.stream).create_recorder();
            recorder.record(ReplayOp::opaque(sdpa.invoke_map(
                GQAPagedSDPAMapBuffers {
                    q: &self._q_norm_rope,
                    kv_pages: &self._kv_pages,
                    req_slots: self.batch_metadata.req_slots(),
                    page_ids: &self._page_ids,
                    sdpa_map_task_templates: self.batch_metadata.sdpa_map_task_templates(),
                    partial_exp_sums: &self._sdpa_partial_exp_sums,
                    partial_max_logits: &self._sdpa_partial_max_logits,
                    partial_output: &self._sdpa_partial_output,
                },
                ReplayU32::Fixed(0),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(GQAPagedSDPAReduceBuffers {
                partial_exp_sums: &self._sdpa_partial_exp_sums,
                partial_max_logits: &self._sdpa_partial_max_logits,
                partial_output: &self._sdpa_partial_output,
                cu_sdpa_partial_outputs: self.batch_metadata.cu_sdpa_partial_outputs(),
                output: &self._attention_output,
            })));
            recorder.build()
        };
        let tiled_config = gqa_tiled_sdpa_config(
            gqa_page_table_layout(self.num_reqs, self.end_context_len),
            self.params,
            self.model,
        );
        let tiled_shape = GQATiledSDPAShape {
            num_total_tokens: self.num_tokens,
            num_total_q_token_tiles: self.tiled_batch_metadata.replay_shape().num_q_token_tiles,
            num_total_sdpa_map_task_templates: self
                .tiled_batch_metadata
                .replay_shape()
                .num_total_sdpa_map_task_templates,
        };
        let tiled = GQATiledSDPAKernels::new(&self.device, tiled_config, tiled_shape);
        let tiled_replay = {
            let mut recorder = MetalReplayRuntime::new(&self.stream).create_recorder();
            recorder.record(ReplayOp::opaque(tiled.invoke_map(
                GQATiledSDPAMapBuffers {
                    q: &self._q_norm_rope,
                    kv_pages: &self._kv_pages,
                    req_slots: self.tiled_batch_metadata.req_slots(),
                    page_ids: &self._page_ids,
                    flat_token_indices: self.tiled_batch_metadata.flat_token_indices(),
                    q_token_tiles: self.tiled_batch_metadata.q_token_tiles(),
                    sdpa_map_task_templates: self.tiled_batch_metadata.sdpa_map_task_templates(),
                    partial_output: &self._tiled_partial_output,
                    partial_exp_sums: &self._tiled_partial_exp_sums,
                    partial_max_logits: &self._tiled_partial_max_logits,
                },
                ReplayU32::Fixed(0),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(tiled.invoke_reduce(GQATiledSDPAReduceBuffers {
                partial_output: &self._tiled_partial_output,
                partial_exp_sums: &self._tiled_partial_exp_sums,
                partial_max_logits: &self._tiled_partial_max_logits,
                q_token_tiles: self.tiled_batch_metadata.q_token_tiles(),
                cu_sdpa_partial_outputs: self.tiled_batch_metadata.cu_sdpa_partial_outputs(),
                output: &self._tiled_attention_output,
            })));
            recorder.build()
        };
        let runtime = MetalReplayRuntime::new(&self.stream);
        runtime.submit_replay(&single_q_token_replay).wait();
        runtime.submit_replay(&tiled_replay).wait();

        let single_q_token_output = &self._attention_output;
        let num_q_slots = self.num_tokens as usize * self.model.q_dim();
        let (max_diff_index, expected, actual, max_abs_diff) =
            max_bf16_diff(single_q_token_output, &self._tiled_attention_output, num_q_slots);
        if max_abs_diff > 0.0625 {
            let reference = gqa_attention_reference_at(
                &self._q_norm_rope,
                &self._kv_pages,
                &self._page_ids,
                &self.batch_metadata,
                self.end_context_len.div_ceil(TOKENS_PER_PAGE).max(1),
                max_diff_index,
                self.model,
            );
            panic!(
                "GQA tiled-Q-tokens mismatch at {max_diff_index}: single_q_token={expected} tiled_q_tokens={actual} \
                 cpu_reference={reference} max_abs_diff={max_abs_diff} tolerance=0.0625"
            );
        }
    }

    fn measure(&self, paths: &[GQABenchPath], warmup_iters: usize, iters: usize, runs: usize) {
        let num_tokens_per_req = self
            .num_tokens_per_req
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "case component=gqa model={} num_tokens_per_req=[{num_tokens_per_req}] tiled_q_token_tile_size={} \
             tiled_kv_token_tile_size={} tiled_q_head_tile_size={}",
            self.model.k,
            self.params.tiled_q_token_tile_size,
            self.params.tiled_kv_token_tile_size,
            self.params.tiled_q_head_tile_size,
        );
        if paths.contains(&GQABenchPath::SingleQToken) {
            let samples = measure_runs(runs, warmup_iters, iters, || self.run_single_q_token());
            let _ = self.next_hidden_state.len_bytes();
            print_perf(
                self.num_tokens,
                self.num_reqs,
                Some(self.existing_context_len),
                Some("single_q_token"),
                iters,
                &samples,
            );
        }
        if paths.contains(&GQABenchPath::TiledQTokens) {
            let tiled_samples = measure_runs(runs, warmup_iters, iters, || self.run_tiled_q_tokens());
            let _ = self.tiled_next_hidden_state.len_bytes();
            print_perf(
                self.num_tokens,
                self.num_reqs,
                Some(self.existing_context_len),
                Some("tiled_q_tokens"),
                iters,
                &tiled_samples,
            );
        }
    }

    fn measure_subcomponents(&self, selected_subcomponents: &[String], warmup_iters: usize, iters: usize, runs: usize) {
        let qgkv_matmul = AffineQuantizedMatmul::new(&self.device, gqa_qgkv_affine_config(self.model));
        let qgkv_to_q_g_k_v = GQAQGKVSplitKernel::new(&self.device, gqa_qgkv_to_q_g_k_v_config(self.model));
        let q_norm_rope = RMSNormRopeKernel::new(&self.device, norm_rope_config(self.model.num_q_heads, self.model));
        let k_norm_rope = RMSNormRopeKernel::new(&self.device, norm_rope_config(self.model.num_kv_heads, self.model));
        let kv_page_write = GQAKVPageWrite::new(
            &self.device,
            gqa_kv_page_write_config(self.model, self.model.page_bytes()),
        );
        let gate = GQAActivationGateKernel::new(&self.device, gqa_gate_config(self.model));
        let output = AffineQuantizedMatmul::new(&self.device, gqa_output_affine_config(self.model));
        let sdpa_config = gqa_sdpa_config(self.num_reqs, self.end_context_len, self.params, self.model);
        let sdpa_shape = gqa_sdpa_shape(self.batch_metadata.replay_shape());
        let sdpa = GQAPagedSDPAKernels::new(&self.device, sdpa_config, sdpa_shape);

        let qgkv_replay = build_single_invocation_replay(
            &self.stream,
            qgkv_matmul.invoke(
                self.num_tokens.try_into().expect("GQA token count must fit i32"),
                &self._qgkv,
                0,
                &self._hidden_state,
                0,
                &self._weights.qgkv_weight,
                0,
                &self._weights.qgkv_scales,
                0,
                &self._weights.qgkv_biases,
                0,
            ),
        );
        let split_replay = build_single_invocation_replay(
            &self.stream,
            qgkv_to_q_g_k_v.invoke(
                GQAQGKVSplitShape {
                    num_total_tokens: self.num_tokens,
                },
                GQAQGKVSplitBuffers {
                    qgkv: &self._qgkv,
                    q: &self._q,
                    g: &self._g,
                    k: &self._k,
                    v: &self._v,
                },
            ),
        );
        let q_norm_rope_replay = build_single_invocation_replay(
            &self.stream,
            q_norm_rope.invoke(
                norm_rope_shape(self.num_tokens, self.model.num_q_heads, self.model),
                RMSNormRopeBuffers {
                    input: &self._q,
                    norm_weight: &self._weights.q_norm_weight,
                    flat_token_indices: self.batch_metadata.flat_token_indices(),
                    output: &self._q_norm_rope,
                },
            ),
        );
        let k_norm_rope_replay = build_single_invocation_replay(
            &self.stream,
            k_norm_rope.invoke(
                norm_rope_shape(self.num_tokens, self.model.num_kv_heads, self.model),
                RMSNormRopeBuffers {
                    input: &self._k,
                    norm_weight: &self._weights.k_norm_weight,
                    flat_token_indices: self.batch_metadata.flat_token_indices(),
                    output: &self._k_norm_rope,
                },
            ),
        );
        let page_table_layout = gqa_page_table_layout(self.num_reqs, self.end_context_len);
        let kv_page_write_replay = build_single_invocation_replay(
            &self.stream,
            kv_page_write.invoke(
                GQAKVPageWriteShape {
                    num_total_token_writes: self.num_tokens,
                    page_table_layout,
                },
                GQAKVPageWriteBuffers {
                    pages: &self._kv_pages,
                    flat_k: &self._k_norm_rope,
                    flat_v: &self._v,
                    req_slots: self.batch_metadata.req_slots(),
                    flat_token_indices: self.batch_metadata.flat_token_indices(),
                    page_ids: &self._page_ids,
                },
                ReplayU32::Fixed(0),
            ),
        );
        let single_q_token_replay = {
            let mut recorder = MetalReplayRuntime::new(&self.stream).create_recorder();
            recorder.record(ReplayOp::opaque(sdpa.invoke_map(
                GQAPagedSDPAMapBuffers {
                    q: &self._q_norm_rope,
                    kv_pages: &self._kv_pages,
                    req_slots: self.batch_metadata.req_slots(),
                    page_ids: &self._page_ids,
                    sdpa_map_task_templates: self.batch_metadata.sdpa_map_task_templates(),
                    partial_exp_sums: &self._sdpa_partial_exp_sums,
                    partial_max_logits: &self._sdpa_partial_max_logits,
                    partial_output: &self._sdpa_partial_output,
                },
                ReplayU32::Fixed(0),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(GQAPagedSDPAReduceBuffers {
                partial_exp_sums: &self._sdpa_partial_exp_sums,
                partial_max_logits: &self._sdpa_partial_max_logits,
                partial_output: &self._sdpa_partial_output,
                cu_sdpa_partial_outputs: self.batch_metadata.cu_sdpa_partial_outputs(),
                output: &self._attention_output,
            })));
            recorder.build()
        };
        let tiled_config = gqa_tiled_sdpa_config(page_table_layout, self.params, self.model);
        let tiled_shape = GQATiledSDPAShape {
            num_total_tokens: self.num_tokens,
            num_total_q_token_tiles: self.tiled_batch_metadata.replay_shape().num_q_token_tiles,
            num_total_sdpa_map_task_templates: self
                .tiled_batch_metadata
                .replay_shape()
                .num_total_sdpa_map_task_templates,
        };
        let tiled_kernel = GQATiledSDPAKernels::new(&self.device, tiled_config, tiled_shape);
        let tiled_replay = {
            let mut recorder = MetalReplayRuntime::new(&self.stream).create_recorder();
            recorder.record(ReplayOp::opaque(tiled_kernel.invoke_map(
                GQATiledSDPAMapBuffers {
                    q: &self._q_norm_rope,
                    kv_pages: &self._kv_pages,
                    req_slots: self.tiled_batch_metadata.req_slots(),
                    page_ids: &self._page_ids,
                    flat_token_indices: self.tiled_batch_metadata.flat_token_indices(),
                    q_token_tiles: self.tiled_batch_metadata.q_token_tiles(),
                    sdpa_map_task_templates: self.tiled_batch_metadata.sdpa_map_task_templates(),
                    partial_output: &self._tiled_partial_output,
                    partial_exp_sums: &self._tiled_partial_exp_sums,
                    partial_max_logits: &self._tiled_partial_max_logits,
                },
                ReplayU32::Fixed(0),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(tiled_kernel.invoke_reduce(
                GQATiledSDPAReduceBuffers {
                    partial_output: &self._tiled_partial_output,
                    partial_exp_sums: &self._tiled_partial_exp_sums,
                    partial_max_logits: &self._tiled_partial_max_logits,
                    q_token_tiles: self.tiled_batch_metadata.q_token_tiles(),
                    cu_sdpa_partial_outputs: self.tiled_batch_metadata.cu_sdpa_partial_outputs(),
                    output: &self._tiled_attention_output,
                },
            )));
            recorder.build()
        };
        let runtime = MetalReplayRuntime::new(&self.stream);
        runtime.submit_replay(&single_q_token_replay).wait();
        runtime.submit_replay(&tiled_replay).wait();
        let single_q_token_output = &self._attention_output;
        assert_bf16_close(
            single_q_token_output,
            &self._tiled_attention_output,
            self.num_tokens as usize * self.model.q_dim(),
            0.0625,
        );
        let gate_replay = build_single_invocation_replay(
            &self.stream,
            gate.invoke(
                GQAActivationGateShape {
                    num_total_tokens: self.num_tokens,
                },
                GQAActivationGateBuffers {
                    attention_output: &self._attention_output,
                    g: &self._g,
                    output: &self._gated_attention_output,
                },
            ),
        );
        let output_replay = build_single_invocation_replay(
            &self.stream,
            output.invoke(
                self.num_tokens.try_into().expect("GQA token count must fit i32"),
                &self.next_hidden_state,
                0,
                &self._gated_attention_output,
                0,
                &self._weights.output_weight,
                0,
                &self._weights.output_scales,
                0,
                &self._weights.output_biases,
                0,
            ),
        );

        self.measure_subcomponent(selected_subcomponents, "qgkv", &qgkv_replay, warmup_iters, iters, runs);
        self.measure_subcomponent(
            selected_subcomponents,
            "qgkv-to-q-g-k-v",
            &split_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "q-norm-rope",
            &q_norm_rope_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "k-norm-rope",
            &k_norm_rope_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "kv-page-write",
            &kv_page_write_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "sdpa-single-q-token",
            &single_q_token_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "sdpa-tiled-q-tokens",
            &tiled_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(selected_subcomponents, "gate", &gate_replay, warmup_iters, iters, runs);
        self.measure_subcomponent(
            selected_subcomponents,
            "output",
            &output_replay,
            warmup_iters,
            iters,
            runs,
        );
    }

    fn measure_subcomponent(
        &self,
        selected_subcomponents: &[String],
        name: &str,
        replay: &ReplayProgram,
        warmup_iters: usize,
        iters: usize,
        runs: usize,
    ) {
        if !selected_subcomponents.iter().any(|selected| selected == name) {
            return;
        }
        let samples = measure_runs(runs, warmup_iters, iters, || {
            MetalReplayRuntime::new(&self.stream).submit_replay(replay).wait();
        });
        print_named_perf(
            &format!("gqa.{name}"),
            self.num_tokens,
            self.num_reqs,
            Some(self.existing_context_len),
            iters,
            &samples,
        );
    }
}
