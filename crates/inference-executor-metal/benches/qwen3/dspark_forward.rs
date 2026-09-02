use std::hint::black_box;
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_backend_metal::metal::ReplayProgram;
use inference_executor_core::attn::BiDiBlockCapacity;
use inference_executor_core::attn::BiDiBlockGQAMetadata;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::checkpoint::SafeTensorStore;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_core::model::qwen::v3::QWEN3_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;
use inference_executor_core::model::qwen::v3::init_qwen3_model_config;
use inference_executor_core::model::qwen::v3::weight_layout::resolve_qwen3_model_weight_bindings;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkWeightBindings;
use inference_executor_core::model::qwen::v3_x::dspark::init_qwen3x_dspark_config;
use inference_executor_core::model::qwen::v3_x::dspark::resolve_qwen3x_dspark_weight_bindings;
use inference_executor_metal::attn::bidi_block_gqa::state::BiDiBlockGQAState;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::model::embedding::Embed;
use inference_executor_metal::model::embedding::EmbedConfig;
use inference_executor_metal::model::page_arena::PageArena;
use inference_executor_metal::model::qwen::v3::executor::Qwen3Executor;
use inference_executor_metal::model::qwen::v3::executor::Qwen3ExecutorConfig;
use inference_executor_metal::model::qwen::v3::executor::init_qwen_3_model;
use inference_executor_metal::model::qwen::v3::executor::init_qwen_3_model_with_dspark;
use inference_executor_metal::model::qwen::v3_x::dspark::attention::qwen3x_dspark_gqa_core;
use inference_executor_metal::model::qwen::v3_x::dspark::attention::qwen3x_dspark_gqa_sdpa_config;
use inference_executor_metal::model::qwen::v3_x::dspark::embed::Qwen3xDSparkEmbed;
use inference_executor_metal::model::qwen::v3_x::dspark::embed::Qwen3xDSparkEmbedArgs;
use inference_executor_metal::model::qwen::v3_x::dspark::model::Qwen3xDSparkBody;
use inference_executor_metal::model::qwen::v3_x::dspark::model::Qwen3xDSparkBodyArgs;
use inference_executor_metal::model::qwen::v3_x::dspark::model::Qwen3xDSparkModel;
use inference_executor_metal::replay::ReplayComponent;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::Token;

fn main() {
    let args = Args::parse();
    let dspark_config =
        init_qwen3x_dspark_config(&args.dspark_model_dir).expect("unable to load Qwen3 DSpark benchmark config");
    let measurement = Measurement {
        warmup_iters: args.warmup_iters,
        iters: args.iters,
        runs: args.runs,
    };

    {
        let mut main = MainFixture::new(
            &args.model_dir,
            args.num_requests,
            args.context,
            dspark_config.block_size,
        );
        measure_and_print(
            ForwardCase {
                name: "main",
                num_tokens: main.num_tokens,
                num_requests: args.num_requests,
                context: args.context,
            },
            main.num_layers,
            measurement,
            || main.run(),
        );
    }
    {
        let mut main_verification = MainFixture::new_with_dspark(
            &args.model_dir,
            &args.dspark_model_dir,
            args.num_requests,
            args.context,
            dspark_config
                .block_size
                .checked_add(1)
                .expect("Main verification row count must fit usize"),
        );
        measure_segment_and_print(
            ForwardCase {
                name: "main-verification",
                num_tokens: main_verification.num_tokens,
                num_requests: args.num_requests,
                context: args.context,
            },
            "embed-forward-prefill",
            measurement,
            || main_verification.run(),
        );
    }
    let dspark = DSparkFixture::new(&args.model_dir, &args.dspark_model_dir, args.num_requests, args.context);
    assert_eq!(
        args.num_requests
            .checked_mul(dspark_config.block_size)
            .expect("Main and DSpark comparison row count must fit usize"),
        dspark.num_tokens,
        "Main and DSpark comparison rows must match"
    );
    measure_segment_and_print(
        ForwardCase {
            name: "dspark",
            num_tokens: dspark.num_tokens,
            num_requests: args.num_requests,
            context: args.context,
        },
        "embed",
        measurement,
        || dspark.run_embed(),
    );
    measure_segment_and_print(
        ForwardCase {
            name: "dspark",
            num_tokens: dspark.num_tokens,
            num_requests: args.num_requests,
            context: args.context,
        },
        "forward",
        measurement,
        || dspark.run_forward(),
    );
    measure_and_print(
        ForwardCase {
            name: "dspark",
            num_tokens: dspark.num_tokens,
            num_requests: args.num_requests,
            context: args.context,
        },
        dspark.num_layers,
        measurement,
        || dspark.run(),
    );
}

struct Args {
    model_dir: PathBuf,
    dspark_model_dir: PathBuf,
    num_requests: usize,
    context: u32,
    warmup_iters: usize,
    iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            dspark_model_dir: PathBuf::new(),
            num_requests: 1,
            context: 128,
            warmup_iters: 10,
            iters: 50,
            runs: 5,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--dspark-model-dir" => args.dspark_model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--num-requests" => args.num_requests = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--context" => args.context = parse_u32(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(
            !args.dspark_model_dir.as_os_str().is_empty(),
            "--dspark-model-dir is required"
        );
        assert!(args.num_requests > 0, "--num-requests must be positive");
        assert!(args.context > 0, "--context must be positive");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args
    }
}

struct MainFixture {
    model: Qwen3Executor,
    num_requests: usize,
    context: usize,
    tokens_per_request: usize,
    num_tokens: usize,
    num_layers: usize,
    num_cache_pages: usize,
    num_page_ids_per_block: usize,
    next_sequence: u64,
}

impl MainFixture {
    fn new(model_dir: &Path, num_requests: usize, context: u32, tokens_per_request: usize) -> Self {
        Self::load(model_dir, None, num_requests, context, tokens_per_request)
    }

    fn new_with_dspark(
        model_dir: &Path,
        dspark_model_dir: &Path,
        num_requests: usize,
        context: u32,
        tokens_per_request: usize,
    ) -> Self {
        Self::load(
            model_dir,
            Some(dspark_model_dir),
            num_requests,
            context,
            tokens_per_request,
        )
    }

    fn load(
        model_dir: &Path,
        dspark_model_dir: Option<&Path>,
        num_requests: usize,
        context: u32,
        tokens_per_request: usize,
    ) -> Self {
        let config = init_qwen3_model_config(model_dir).expect("unable to load Qwen3 benchmark config");
        let num_layers = config.text_config.num_hidden_layers;
        let num_tokens = num_requests
            .checked_mul(tokens_per_request)
            .expect("Main comparison token count must fit usize");
        let mut num_page_ids_per_block = qwen3_main_page_ids_per_cache_block(&config);
        if let Some(dspark_model_dir) = dspark_model_dir {
            let dspark_config =
                init_qwen3x_dspark_config(dspark_model_dir).expect("unable to load Qwen3 DSpark benchmark config");
            let split_kv_config = qwen3x_dspark_gqa_sdpa_config(&dspark_config, QWEN3_PAGE_SIZE_BYTES)
                .expect("unable to build Qwen3 DSpark GQA SplitKV config");
            let tokens_per_page = split_kv_config.tokens_per_page as usize;
            num_page_ids_per_block = num_page_ids_per_block
                .checked_add(
                    dspark_config
                        .num_hidden_layers
                        .checked_mul(pages_per_cache_block(tokens_per_page))
                        .expect("DSpark page IDs per cache block must fit usize"),
                )
                .expect("Main and DSpark page IDs per cache block must fit usize");
        }
        let num_cache_pages = num_requests
            .checked_mul(num_page_ids_per_block)
            .expect("Main comparison page count must fit usize");
        let executor_config = Qwen3ExecutorConfig {
            max_requests: num_requests,
            max_tokens: num_tokens,
            max_tokens_per_request: 1024,
            num_cache_pages,
            num_tokens_per_block: 1024,
        };
        let model = match dspark_model_dir {
            Some(dspark_model_dir) => init_qwen_3_model_with_dspark(model_dir, dspark_model_dir, executor_config),
            None => init_qwen_3_model(model_dir, executor_config),
        }
        .expect("unable to initialize Main comparison executor");
        let actual_num_page_ids_per_block = model.num_kv_page_ids_per_block();
        assert!(
            num_requests
                .checked_mul(actual_num_page_ids_per_block)
                .is_some_and(|pages| pages <= num_cache_pages),
            "Main comparison page IDs exceed the configured page arena"
        );
        Self {
            model,
            num_requests,
            context: context as usize,
            tokens_per_request,
            num_tokens,
            num_layers,
            num_cache_pages,
            num_page_ids_per_block: actual_num_page_ids_per_block,
            next_sequence: 0,
        }
    }

    fn run(&mut self) -> Duration {
        let batch = self.batch_request();
        let prepared = self.model.prepare_batch(&batch);
        let mut recorder = self.model.begin_ops_recording(&prepared);
        let hidden = self.model.embed_main(&mut recorder, &prepared);
        let hidden = self.model.forward_main(&mut recorder, &prepared, hidden);
        let output = self.model.unembed_main(&mut recorder, &prepared, &hidden);
        self.model.sample_main(&mut recorder, &prepared, &output);
        let replay_start = Instant::now();
        let submission = self.model.submit_main(&recorder);
        submission.wait();
        let replay_elapsed = replay_start.elapsed();
        let gpu_timestamp_durations = submission.gpu_timestamp_durations();
        drop(submission);
        let sampled = self
            .model
            .read_main(&recorder, &prepared, replay_elapsed, gpu_timestamp_durations.as_deref());
        assert_eq!(self.model.sampled_output_len(&sampled), 0);
        let spec_replay_elapsed = if self.model.run_spec_prefill(&prepared) {
            self.model.prefill_spec(&mut recorder, &prepared, &sampled);
            let replay_start = Instant::now();
            self.model.submit_spec(&recorder).wait();
            replay_start.elapsed()
        } else {
            Duration::ZERO
        };
        drop(recorder);
        black_box(self.model.commit_batch(batch, sampled));
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .expect("Main comparison sequence must fit u64");
        replay_elapsed + spec_replay_elapsed
    }

    fn batch_request(&self) -> BatchDeviceRequest {
        let requests = (0..self.num_requests)
            .map(|request_index| {
                let req_slot = request_index
                    .try_into()
                    .expect("Main comparison request slot must fit u32");
                let first_page = request_index
                    .checked_mul(self.num_page_ids_per_block)
                    .expect("Main comparison page offset must fit usize");
                let page_end = first_page
                    .checked_add(self.num_page_ids_per_block)
                    .expect("Main comparison page end must fit usize");
                assert!(page_end <= self.num_cache_pages);
                DeviceRequest::new(
                    request_index,
                    req_slot,
                    QueryTokens::Prefill {
                        epoch: 0,
                        token_index: self.context,
                        tokens: vec![Token::new(11); self.tokens_per_request],
                        window: self.tokens_per_request,
                    },
                    DecoderSyncBlocks::new(
                        0,
                        vec![vec![
                            (first_page..page_end)
                                .map(|page_id| page_id.try_into().expect("Main comparison page ID must fit u32"))
                                .collect(),
                        ]],
                        Vec::new(),
                    ),
                    None,
                    vec![],
                    SamplingConfig {
                        max_sampled_tokens: usize::MAX,
                        temperature: 0.0,
                        top_k: 1,
                        top_p: 1.0,
                        seed: Some(req_slot),
                        stop_sequences: Vec::new(),
                    },
                )
            })
            .collect::<Vec<_>>();
        BatchDeviceRequest::new(self.next_sequence, requests)
    }
}

struct DSparkFixture {
    runtime: MetalRuntime,
    embed_replay: ReplayProgram,
    forward_replay: ReplayProgram,
    empty_arguments: ReplayArguments,
    output: Buffer,
    num_tokens: usize,
    num_layers: usize,
    _embed: Qwen3xDSparkEmbed,
    _body: Qwen3xDSparkBody,
    _model: Rc<Qwen3xDSparkModel>,
    _gqa_state: BiDiBlockGQAState,
    _token_ids: Buffer,
    hidden_input: Buffer,
    _pages: PageArena,
}

impl DSparkFixture {
    fn new(main_model_dir: &Path, dspark_model_dir: &Path, num_requests: usize, context: u32) -> Self {
        let config = init_qwen3x_dspark_config(dspark_model_dir).expect("unable to load Qwen3 DSpark benchmark config");
        let runtime = MetalRuntime::system_default();
        let device = runtime.device();
        let mut store =
            SafeTensorStore::from_model_dir(dspark_model_dir).expect("unable to open Qwen3 DSpark comparison weights");
        let bindings = resolve_qwen3x_dspark_weight_bindings(&config, store.index().tensor_names())
            .expect("unable to resolve Qwen3 DSpark comparison weights");
        let attention_core = qwen3x_dspark_gqa_core(&config, config.block_size, 0);
        let attention_split_kv_config = qwen3x_dspark_gqa_sdpa_config(&config, QWEN3_PAGE_SIZE_BYTES)
            .expect("unable to build Qwen3 DSpark GQA SplitKV config");
        let num_layers = config.num_hidden_layers;
        let capacity = BiDiBlockCapacity::new(num_requests, config.block_size);
        let tokens_per_page = attention_split_kv_config.tokens_per_page as usize;
        let num_page_ids_per_block = pages_per_cache_block(tokens_per_page);
        let page_table_layout = GQAPageTableLayout {
            num_req_slots: num_requests
                .try_into()
                .expect("DSpark comparison request count must fit u32"),
            num_blocks: 1,
            num_gqa_layers: num_layers
                .try_into()
                .expect("DSpark comparison layer count must fit u32"),
            num_page_ids_per_block: num_page_ids_per_block
                .try_into()
                .expect("DSpark comparison page count must fit u32"),
        };
        let num_cache_pages = num_requests
            .checked_mul(num_layers)
            .and_then(|pages| pages.checked_mul(num_page_ids_per_block))
            .expect("DSpark comparison page count must fit usize");
        let gqa_state = BiDiBlockGQAState::new(
            device,
            attention_core,
            attention_split_kv_config,
            page_table_layout,
            capacity,
            capacity.max_tokens,
            num_cache_pages,
        );
        let flat_query_token_indices = (0..num_requests)
            .flat_map(|_| context..context + config.block_size as u32)
            .collect::<Vec<_>>();
        let block = BiDiBlockGQAMetadata::new(
            &request_slots(num_requests),
            &flat_query_token_indices,
            &vec![0..context; num_requests * config.block_size],
            config.block_size,
        );
        gqa_state.prepare_bidi_block(&block);
        let request_page_table = gqa_state.request_page_table();
        for req_slot in 0..num_requests {
            for layer_index in 0..num_layers {
                let first_page = req_slot
                    .checked_mul(num_layers)
                    .and_then(|index| index.checked_add(layer_index))
                    .and_then(|index| index.checked_mul(num_page_ids_per_block))
                    .expect("DSpark comparison page offset must fit usize");
                request_page_table.write_page_ids(
                    req_slot
                        .try_into()
                        .expect("DSpark comparison request slot must fit u32"),
                    layer_index,
                    0,
                    &(first_page..first_page + num_page_ids_per_block)
                        .map(|page_id| page_id.try_into().expect("DSpark comparison page ID must fit u32"))
                        .collect::<Vec<_>>(),
                );
            }
        }

        let Qwen3xDSparkWeightBindings {
            embed: embed_bindings,
            main_feature: main_feature_bindings,
            layers: layer_bindings,
            final_norm_weight,
            ..
        } = bindings;
        let embed = Qwen3xDSparkEmbed::new(load_dspark_embed(
            device,
            main_model_dir,
            &mut store,
            &config,
            capacity.max_tokens,
            embed_bindings,
        ));
        let mut model = Qwen3xDSparkModel::new(
            device,
            &config,
            config.block_size,
            QWEN3_PAGE_SIZE_BYTES,
            &main_feature_bindings,
            &layer_bindings,
            &gqa_state,
            capacity.max_tokens,
            capacity.max_tokens,
        )
        .expect("unable to construct Qwen3 DSpark comparison model");
        model
            .load_weights(
                device,
                &mut store,
                &config,
                &main_feature_bindings,
                layer_bindings,
                final_norm_weight,
            )
            .expect("unable to load Qwen3 DSpark comparison model");
        let model = Rc::new(model);
        let body = Qwen3xDSparkBody::new(Rc::clone(&model));
        let hidden_elements = capacity
            .max_tokens
            .checked_mul(config.hidden_size)
            .expect("DSpark comparison hidden size must fit usize");
        let token_ids = Buffer::from_slice(
            device,
            &(0..capacity.max_tokens)
                .map(|index| {
                    if index % config.block_size == 0 {
                        11
                    } else {
                        config
                            .mask_token_id
                            .try_into()
                            .expect("DSpark MASK token ID must fit i32")
                    }
                })
                .collect::<Vec<i32>>(),
        );
        let hidden_input = Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16);
        let output = Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16);
        let pages = PageArena::new(device, num_cache_pages, QWEN3_PAGE_SIZE_BYTES);
        let replay_runtime = MetalReplayRuntime::new(runtime.stream());

        let mut embed_recorder = replay_runtime.create_recorder();
        embed.record(
            &mut embed_recorder,
            &Qwen3xDSparkEmbedArgs {
                num_tokens: capacity
                    .max_tokens
                    .try_into()
                    .expect("DSpark comparison token count must fit u32"),
                token_ids: &token_ids,
                hidden_output: &hidden_input,
            },
        );
        let embed_replay = embed_recorder.build();

        let mut forward_recorder = replay_runtime.create_recorder();
        body.record(
            &mut forward_recorder,
            &Qwen3xDSparkBodyArgs {
                num_tokens: capacity
                    .max_tokens
                    .try_into()
                    .expect("DSpark comparison token count must fit u32"),
                metadata: gqa_state.metadata(),
                hidden_input: &hidden_input,
                hidden_output: &output,
                pages: pages.buffer(),
            },
        );
        let forward_replay = forward_recorder.build();
        Self {
            runtime,
            embed_replay,
            forward_replay,
            empty_arguments: ReplayArguments::new(),
            output,
            num_tokens: capacity.max_tokens,
            num_layers: config.num_hidden_layers,
            _embed: embed,
            _body: body,
            _model: model,
            _gqa_state: gqa_state,
            _token_ids: token_ids,
            hidden_input,
            _pages: pages,
        }
    }

    fn run_embed(&self) -> Duration {
        let replay_start = Instant::now();
        self.runtime.stream().submit_replay(&self.embed_replay).wait();
        black_box(&self.hidden_input);
        replay_start.elapsed()
    }

    fn run_forward(&self) -> Duration {
        let replay_start = Instant::now();
        self.runtime.stream().submit_replay(&self.forward_replay).wait();
        black_box(&self.output);
        replay_start.elapsed()
    }

    fn run(&self) -> Duration {
        let replay_start = Instant::now();
        self.runtime
            .stream()
            .submit_replay_sequence(&[
                ReplayExecution::new(&self.embed_replay, &self.empty_arguments),
                ReplayExecution::new(&self.forward_replay, &self.empty_arguments),
            ])
            .wait();
        black_box(&self.output);
        replay_start.elapsed()
    }
}

fn load_main_text_embed(
    device: &inference_backend_metal::metal::Device,
    store: &mut SafeTensorStore,
    config: &Qwen3ModelConfig,
    max_tokens: u32,
    bindings: inference_executor_core::checkpoint::QuantizedTensorBindings,
) -> Embed {
    let quantization = config
        .quantization
        .as_ref()
        .expect("Main comparison requires quantization");
    let mut embed = Embed::new(
        device,
        EmbedConfig {
            max_tokens,
            vocab_size: config
                .text_config
                .vocab_size
                .try_into()
                .expect("Main vocabulary must fit u32"),
            hidden_dim: config
                .text_config
                .hidden_size
                .try_into()
                .expect("Main hidden dimension must fit u32"),
            group_size: quantization
                .group_size
                .try_into()
                .expect("Main group size must fit u32"),
            bits: quantization.bits.try_into().expect("Main bit count must fit u32"),
            scale_bias_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        },
    );
    embed
        .load_weights(device, store, bindings)
        .expect("unable to load Main comparison embedding");
    embed
}

fn load_dspark_embed(
    device: &inference_backend_metal::metal::Device,
    main_model_dir: &Path,
    dspark_store: &mut SafeTensorStore,
    config: &Qwen3xDSparkConfig,
    max_tokens: usize,
    bindings: Option<inference_executor_core::checkpoint::QuantizedTensorBindings>,
) -> Rc<Embed> {
    if let Some(bindings) = bindings {
        let quantization = config
            .quantization
            .as_ref()
            .expect("DSpark comparison requires quantization")
            .resolve_for_tensor(&bindings.weight);
        let mut embed = Embed::new(
            device,
            EmbedConfig {
                max_tokens: max_tokens.try_into().expect("DSpark embed token capacity must fit u32"),
                vocab_size: config
                    .vocab_size
                    .try_into()
                    .expect("DSpark embedding vocabulary must fit u32"),
                hidden_dim: config
                    .hidden_size
                    .try_into()
                    .expect("DSpark embedding dimension must fit u32"),
                group_size: quantization
                    .group_size
                    .try_into()
                    .expect("DSpark embedding group size must fit u32"),
                bits: quantization
                    .bits
                    .try_into()
                    .expect("DSpark embedding bits must fit u32"),
                scale_bias_dtype: Dtype::Bfloat16,
                output_dtype: Dtype::Bfloat16,
            },
        );
        embed
            .load_weights(device, dspark_store, bindings)
            .expect("unable to load DSpark comparison embedding");
        return Rc::new(embed);
    }

    let main_config = init_qwen3_model_config(main_model_dir).expect("unable to load Main embedding config");
    let mut main_store =
        SafeTensorStore::from_model_dir(main_model_dir).expect("unable to open Main embedding weights");
    let main_bindings = resolve_qwen3_model_weight_bindings(&main_config, main_store.index().tensor_names())
        .expect("unable to resolve Main embedding weights");
    Rc::new(load_main_text_embed(
        device,
        &mut main_store,
        &main_config,
        max_tokens
            .try_into()
            .expect("Main fallback embed token count must fit u32"),
        main_bindings.embed,
    ))
}

fn pages_per_cache_block(tokens_per_page: usize) -> usize {
    const TOKENS_PER_CACHE_BLOCK: usize = 1024;
    assert!(tokens_per_page > 0);
    assert!(TOKENS_PER_CACHE_BLOCK.is_multiple_of(tokens_per_page));
    TOKENS_PER_CACHE_BLOCK / tokens_per_page
}

fn qwen3_main_page_ids_per_cache_block(config: &Qwen3ModelConfig) -> usize {
    let text = &config.text_config;
    let kv_token_bytes = text
        .num_key_value_heads
        .checked_mul(text.head_dim)
        .and_then(|values| values.checked_mul(2))
        .and_then(|values| values.checked_mul(size_of::<u8>()))
        .expect("Main comparison K/V token size must fit usize");
    assert!(QWEN3_PAGE_SIZE_BYTES.is_multiple_of(kv_token_bytes));
    let tokens_per_page = QWEN3_PAGE_SIZE_BYTES / kv_token_bytes;
    text.num_hidden_layers
        .checked_mul(pages_per_cache_block(tokens_per_page))
        .expect("Main comparison page IDs per cache block must fit usize")
}

fn request_slots(num_requests: usize) -> Vec<u32> {
    (0..num_requests)
        .map(|slot| slot.try_into().expect("comparison request slot must fit u32"))
        .collect()
}

#[derive(Clone, Copy)]
struct ForwardCase {
    name: &'static str,
    num_tokens: usize,
    num_requests: usize,
    context: u32,
}

#[derive(Clone, Copy)]
struct Measurement {
    warmup_iters: usize,
    iters: usize,
    runs: usize,
}

fn measure_and_print(
    case: ForwardCase,
    num_layers: usize,
    measurement: Measurement,
    mut run: impl FnMut() -> Duration,
) {
    let cache_miss = run();
    for _ in 0..measurement.warmup_iters {
        let _ = run();
    }
    let samples = (0..measurement.runs)
        .map(|_| (0..measurement.iters).map(|_| run()).sum())
        .collect::<Vec<_>>();
    let median = median_duration(samples);
    println!(
        "perf component=qwen3-dspark-forward case={} num_requests={} num_tokens={} context={} num_layers={num_layers} \
         operation=embed-forward cache_miss_us={:.3} warmup_iters={} iters={} runs={} median_us={:.3} \
         per_iter_us={:.3} per_layer_us={:.3}",
        case.name,
        case.num_requests,
        case.num_tokens,
        case.context,
        cache_miss.as_secs_f64() * 1.0e6,
        measurement.warmup_iters,
        measurement.iters,
        measurement.runs,
        median.as_secs_f64() * 1.0e6,
        median.as_secs_f64() * 1.0e6 / measurement.iters as f64,
        median.as_secs_f64() * 1.0e6 / measurement.iters as f64 / num_layers as f64,
    );
}

fn measure_segment_and_print(
    case: ForwardCase,
    operation: &str,
    measurement: Measurement,
    mut run: impl FnMut() -> Duration,
) {
    let cache_miss = run();
    for _ in 0..measurement.warmup_iters {
        let _ = run();
    }
    let samples = (0..measurement.runs)
        .map(|_| (0..measurement.iters).map(|_| run()).sum())
        .collect::<Vec<_>>();
    let median = median_duration(samples);
    println!(
        "perf component=qwen3-dspark-forward case={} num_requests={} num_tokens={} context={} operation={operation} \
         cache_miss_us={:.3} warmup_iters={} iters={} runs={} median_us={:.3} per_iter_us={:.3}",
        case.name,
        case.num_requests,
        case.num_tokens,
        case.context,
        cache_miss.as_secs_f64() * 1.0e6,
        measurement.warmup_iters,
        measurement.iters,
        measurement.runs,
        median.as_secs_f64() * 1.0e6,
        median.as_secs_f64() * 1.0e6 / measurement.iters as f64,
    );
}

fn median_duration(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2
    } else {
        values[mid]
    }
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a usize"))
}

fn parse_u32(value: &str, flag: &str) -> u32 {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a u32"))
}

fn next_arg(values: &mut impl Iterator<Item = String>, flag: &str) -> String {
    values.next().unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn print_help_and_exit() -> ! {
    println!(
        "qwen3_dspark_forward bench\n--model-dir PATH\n--dspark-model-dir PATH\n--num-requests N\n--context \
         N\n--warmup-iters N\n--iters N\n--runs N"
    );
    std::process::exit(0);
}
