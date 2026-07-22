use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::checkpoint::SafeTensorStore;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::init_model_config;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35ModelWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_model_weight_bindings;
use inference_executor_metal::attn::gdn::backend::GDN;
use inference_executor_metal::attn::gdn::batch_metadata::GDNMetadataBuffers;
use inference_executor_metal::attn::gdn::scratch::GDNScratch;
use inference_executor_metal::attn::gdn::state_table::GDNRequestStateTable;
use inference_executor_metal::attn::gqa::backend::GQA;
use inference_executor_metal::attn::gqa::batch_metadata::GQAMetadataBuffers;
use inference_executor_metal::attn::gqa::request_page_table::GQARequestPageTable;
use inference_executor_metal::attn::gqa::scratch::GQAScratch;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::mlp::dense::scratch::DenseMLPScratch;
use inference_executor_metal::mlp::moe::scratch::MoEScratch;
use inference_executor_metal::model::page_arena::PageArena;
use inference_executor_metal::model::qwen::v3_5::layer::Qwen35Layer;
use inference_executor_metal::model::qwen::v3_5::layer::Qwen35LayerInput;
use inference_executor_metal::model::qwen::v3_5::layer::scratch::Qwen35LayerScratch;
use inference_executor_metal::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use inference_executor_metal::model::qwen::v3_5::plan::qwen35_dense_mlp_core_and_metal;
use inference_executor_metal::model::qwen::v3_5::plan::qwen35_gdn_core_and_metal;
use inference_executor_metal::model::qwen::v3_5::plan::qwen35_gqa_core_and_metal;
use inference_executor_metal::model::qwen::v3_5::plan::qwen35_layer_counts;
use inference_executor_metal::model::qwen::v3_5::plan::qwen35_moe_core_and_metal;

const DEFAULT_TOKENS: u32 = 1;
const DEFAULT_CONTEXT: u32 = 32;
const CACHE_BLOCK_TOKENS: usize = 2048;

#[derive(Clone, Copy, Debug)]
struct BenchShape {
    num_tokens: u32,
    context: u32,
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Layer(usize),
    FirstLayers(usize),
    MainAll,
}

impl Case {
    fn layer_indices(self, num_main_layers: usize) -> Vec<usize> {
        match self {
            Self::Layer(model_layer_index) => vec![model_layer_index],
            Self::FirstLayers(count) => (0..count).collect(),
            Self::MainAll => (0..num_main_layers).collect(),
        }
    }

    fn key(self) -> String {
        match self {
            Self::Layer(model_layer_index) => format!("layer{model_layer_index}"),
            Self::FirstLayers(count) => format!("first{count}"),
            Self::MainAll => "main_all".to_string(),
        }
    }
}

struct BenchArgs {
    model_dir: PathBuf,
    cases: Vec<Case>,
    shapes: Vec<BenchShape>,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl BenchArgs {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            cases: selected_cases(),
            shapes: vec![BenchShape {
                num_tokens: DEFAULT_TOKENS,
                context: DEFAULT_CONTEXT,
            }],
            iters: 200,
            warmup_iters: 50,
            runs: 3,
        };
        let mut num_tokens = None;
        let mut contexts = None;
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut iter, &arg)),
                "--cases" => args.cases = parse_cases(&next_arg(&mut iter, &arg)),
                "--tokens" => num_tokens = Some(parse_u32_list(&next_arg(&mut iter, &arg), &arg)),
                "--contexts" => contexts = Some(parse_u32_list(&next_arg(&mut iter, &arg), &arg)),
                "--iters" => args.iters = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        let num_tokens = num_tokens.unwrap_or_else(|| vec![DEFAULT_TOKENS]);
        let contexts = contexts.unwrap_or_else(|| vec![DEFAULT_CONTEXT]);
        assert!(!num_tokens.is_empty(), "--tokens must include at least one value");
        assert!(!contexts.is_empty(), "--contexts must include at least one value");
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        args.shapes = num_tokens
            .iter()
            .flat_map(|&num_tokens| {
                assert!(num_tokens > 0, "--tokens entries must be positive");
                contexts.iter().map(move |&context| BenchShape { num_tokens, context })
            })
            .collect();
        args
    }
}

struct BlockFixture {
    stream: Stream,
    input: Buffer,
    layers: Vec<Qwen35Layer>,
    gqa_metadata: GQAMetadataBuffers,
    gdn_metadata: GDNMetadataBuffers,
    pages: PageArena,
    shape: BenchShape,
}

impl BlockFixture {
    fn new(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        weight_bindings: &Qwen35ModelWeightBindings,
        case: Case,
        shape: BenchShape,
    ) -> Self {
        let defaults = Qwen35MetalDefaults::from_quantization(config.quantization.as_ref())
            .expect("qwen3.5 layer bench requires supported quantization");
        let counts = qwen35_layer_counts(config).expect("qwen3.5 layer bench requires a valid layer schedule");
        assert!(counts.gqa > 0, "qwen3.5 layer bench requires GQA layers");
        assert!(counts.gdn > 0, "qwen3.5 layer bench requires GDN layers");
        let max_tokens = shape.num_tokens as usize;

        let first_gqa_layer = (0..config.text_config.num_hidden_layers)
            .find(|&index| {
                config
                    .layer_type_at(index)
                    .is_ok_and(|kind| kind == LayerType::FullAttention)
            })
            .expect("qwen3.5 layer bench requires a GQA layer");
        let (gqa_core, gqa_metal) = qwen35_gqa_core_and_metal(first_gqa_layer, &config.text_config, defaults)
            .expect("qwen3.5 layer bench requires valid GQA geometry");
        let num_page_ids_per_block = CACHE_BLOCK_TOKENS.div_ceil(gqa_metal.num_tokens_per_page(&gqa_core) as usize);
        let context_end = (shape.context as usize)
            .checked_add(max_tokens)
            .expect("qwen3.5 layer bench context length must fit usize");
        let num_blocks = context_end.div_ceil(CACHE_BLOCK_TOKENS).max(1);
        let num_cache_pages = counts
            .gqa
            .checked_mul(num_blocks)
            .and_then(|value| value.checked_mul(num_page_ids_per_block))
            .expect("qwen3.5 layer bench cache page count must fit usize");
        let gqa_backend = Rc::new(GQA::new(device, gqa_core.clone(), gqa_metal));
        let gqa_scratch = Rc::new(GQAScratch::new(device, &gqa_core, gqa_metal, max_tokens));
        let gqa_page_table = Rc::new(GQARequestPageTable::new(
            device,
            GQAPageTableLayout {
                num_req_slots: 1,
                num_blocks: num_blocks
                    .try_into()
                    .expect("qwen3.5 layer bench block count must fit u32"),
                num_gqa_layers: counts
                    .gqa
                    .try_into()
                    .expect("qwen3.5 layer bench GQA layer count must fit u32"),
                num_page_ids_per_block: num_page_ids_per_block
                    .try_into()
                    .expect("qwen3.5 layer bench page count must fit u32"),
            },
        ));
        let mut next_page_id = 0u32;
        for layer_index in 0..counts.gqa {
            for block_index in 0..num_blocks {
                let page_ids = (0..num_page_ids_per_block)
                    .map(|_| {
                        let page_id = next_page_id;
                        next_page_id = next_page_id
                            .checked_add(1)
                            .expect("qwen3.5 layer bench page ID must fit u32");
                        page_id
                    })
                    .collect::<Vec<_>>();
                gqa_page_table.write_page_ids(0, layer_index, block_index, &page_ids);
            }
        }
        assert_eq!(
            next_page_id as usize, num_cache_pages,
            "qwen3.5 layer bench must initialize every cache page ID"
        );
        let gqa_metadata = GQAMetadataBuffers::new(device, max_tokens);
        gqa_backend.prepare(&gqa_metadata, &[0], &[shape.context], &[0, shape.num_tokens]);

        let gdn_layers = (0..config.text_config.num_hidden_layers)
            .filter(|&index| config.layer_type_at(index).is_ok_and(|kind| kind == LayerType::GDN))
            .collect::<Vec<_>>();
        let gdn_cores = gdn_layers
            .iter()
            .map(|&index| {
                qwen35_gdn_core_and_metal(index, &config.text_config, defaults)
                    .expect("qwen3.5 layer bench requires valid GDN geometry")
                    .0
            })
            .collect::<Vec<_>>();
        let gdn_metal = qwen35_gdn_core_and_metal(gdn_layers[0], &config.text_config, defaults)
            .expect("qwen3.5 layer bench requires valid GDN geometry")
            .1;
        let gdn_backend = Rc::new(GDN::new(device, gdn_cores[0].clone(), gdn_metal));
        let gdn_scratch = Rc::new(GDNScratch::new(device, &gdn_cores[0], gdn_metal, max_tokens));
        let gdn_state_table = Rc::new(GDNRequestStateTable::new(
            device,
            &gdn_cores,
            1,
            0,
            max_tokens,
            CACHE_BLOCK_TOKENS,
            QWEN35_PAGE_SIZE_BYTES,
        ));
        let gdn_metadata = GDNMetadataBuffers::new(device, 1, max_tokens);
        gdn_metadata.update(&[0, shape.num_tokens], &[0], &[0], &vec![0; max_tokens]);

        let layer_scratch = Rc::new(Qwen35LayerScratch::new(
            device,
            max_tokens,
            config.text_config.hidden_size,
        ));
        let dense_scratch = counts.has_dense_mlp.then(|| {
            let index = (0..config.text_config.num_hidden_layers)
                .find(|&index| !config.layer_uses_moe(index))
                .expect("qwen3.5 layer bench dense schedule must contain a dense layer");
            let (core, metal) = qwen35_dense_mlp_core_and_metal(index, &config.text_config, defaults)
                .expect("qwen3.5 layer bench requires valid dense MLP geometry");
            Rc::new(DenseMLPScratch::new(device, &core, metal, max_tokens))
        });
        let moe_scratch = counts.has_moe.then(|| {
            let index = (0..config.text_config.num_hidden_layers)
                .find(|&index| config.layer_uses_moe(index))
                .expect("qwen3.5 layer bench MoE schedule must contain an MoE layer");
            let (core, metal) = qwen35_moe_core_and_metal(&format!("layers.{index}"), index, config, defaults)
                .expect("qwen3.5 layer bench requires valid MoE geometry");
            Rc::new(MoEScratch::new(device, &core, metal, max_tokens))
        });

        let layer_indices = case.layer_indices(config.text_config.num_hidden_layers);
        assert!(!layer_indices.is_empty(), "qwen3.5 layer bench requires layers");
        let mut layers = Vec::with_capacity(layer_indices.len());
        for model_layer_index in layer_indices {
            let bindings = weight_bindings
                .main
                .layers
                .get(model_layer_index)
                .unwrap_or_else(|| panic!("qwen3.5 layer bench missing bindings for layer {model_layer_index}"))
                .clone();
            layers.push(load_layer(
                device,
                store,
                config,
                defaults,
                model_layer_index,
                bindings,
                &gqa_backend,
                &gqa_scratch,
                &gqa_page_table,
                &gdn_backend,
                &gdn_scratch,
                &gdn_state_table,
                Rc::clone(&layer_scratch),
                dense_scratch.as_ref(),
                moe_scratch.as_ref(),
            ));
            store.unload_all();
        }

        Self {
            stream: Stream::new(device),
            input: Buffer::from_slice(device, &hidden_fixture(max_tokens, config.text_config.hidden_size)),
            layers,
            gqa_metadata,
            gdn_metadata,
            pages: PageArena::new(device, num_cache_pages, QWEN35_PAGE_SIZE_BYTES),
            shape,
        }
    }

    fn build_replay(&self) -> ReplayProgram {
        let mut recorder = MetalReplayRuntime::new(&self.stream).create_recorder();
        let mut hidden = &self.input;
        for layer in &self.layers {
            hidden = <Qwen35Layer as ReplayLayer>::record(
                layer,
                &mut recorder,
                Qwen35LayerInput {
                    gdn: Some(&self.gdn_metadata),
                    gqa: &self.gqa_metadata,
                    input: hidden,
                    output: layer.output(),
                    num_tokens: self.shape.num_tokens,
                    pages: self.pages.buffer(),
                },
            );
        }
        recorder.build()
    }

    fn run(&self, replay: &ReplayProgram) {
        MetalReplayRuntime::new(&self.stream).submit_replay(replay).wait();
    }
}

#[allow(clippy::too_many_arguments)]
fn load_layer(
    device: &Device,
    store: &mut SafeTensorStore,
    config: &Qwen35ModelConfig,
    defaults: Qwen35MetalDefaults,
    model_layer_index: usize,
    bindings: Qwen35LayerWeightBindings,
    gqa_backend: &Rc<GQA>,
    gqa_scratch: &Rc<GQAScratch>,
    gqa_page_table: &Rc<GQARequestPageTable>,
    gdn_backend: &Rc<GDN>,
    gdn_scratch: &Rc<GDNScratch>,
    gdn_state_table: &Rc<GDNRequestStateTable>,
    layer_scratch: Rc<Qwen35LayerScratch>,
    dense_scratch: Option<&Rc<DenseMLPScratch>>,
    moe_scratch: Option<&Rc<MoEScratch>>,
) -> Qwen35Layer {
    let Qwen35LayerWeightBindings {
        input_norm_weight,
        post_attention_norm_weight,
        attention,
        mlp,
    } = bindings;
    let mut compact_gqa_layer_index = (0..model_layer_index)
        .filter(|&index| {
            config
                .layer_type_at(index)
                .is_ok_and(|kind| kind == LayerType::FullAttention)
        })
        .count();
    let mut compact_gdn_layer_index = model_layer_index - compact_gqa_layer_index;
    let attention = Qwen35Layer::load_attention(
        device,
        store,
        config,
        defaults,
        model_layer_index,
        &mut compact_gqa_layer_index,
        &mut compact_gdn_layer_index,
        attention,
        gqa_backend,
        gqa_scratch,
        gqa_page_table,
        gdn_backend,
        gdn_scratch,
        gdn_state_table,
    )
    .unwrap_or_else(|err| panic!("unable to load attention for layer {model_layer_index}: {err}"));
    let mlp = Qwen35Layer::load_mlp(
        device,
        store,
        config,
        defaults,
        model_layer_index,
        mlp,
        dense_scratch,
        moe_scratch,
    )
    .unwrap_or_else(|err| panic!("unable to load MLP for layer {model_layer_index}: {err}"));
    Qwen35Layer::load(
        device,
        store,
        config,
        model_layer_index,
        input_norm_weight,
        post_attention_norm_weight,
        attention,
        mlp,
        layer_scratch,
    )
    .unwrap_or_else(|err| panic!("unable to load layer {model_layer_index}: {err}"))
}

fn main() {
    let args = BenchArgs::parse();
    assert!(args.iters > 0, "--iters must be positive");
    assert!(args.runs > 0, "--runs must be positive");
    let device = Device::system_default();
    let model_config = init_model_config(&args.model_dir)
        .unwrap_or_else(|err| panic!("unable to init Qwen3.5 config from {}: {err}", args.model_dir.display()));
    let mut store = SafeTensorStore::from_model_dir(&args.model_dir).unwrap_or_else(|err| {
        panic!(
            "unable to load safetensors store from {}: {err}",
            args.model_dir.display()
        )
    });
    let weight_bindings = resolve_qwen35_model_weight_bindings(&model_config, store.index().tensor_names())
        .unwrap_or_else(|err| {
            panic!(
                "unable to resolve Qwen3.5 weight layout from {}: {err}",
                args.model_dir.display()
            )
        });
    for shape in args.shapes {
        for &case in &args.cases {
            let fixture = BlockFixture::new(&device, &mut store, &model_config, &weight_bindings, case, shape);
            store.unload_all();
            let replay = fixture.build_replay();
            fixture.run(&replay);
            let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || fixture.run(&replay));
            print_perf(&args.model_dir, case, shape, args.iters, &replay, &samples);
        }
    }
}

fn selected_cases() -> Vec<Case> {
    vec![Case::Layer(0), Case::Layer(4), Case::FirstLayers(4)]
}

fn parse_cases(value: &str) -> Vec<Case> {
    let cases = value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            match part {
                "layer0" => Case::Layer(0),
                "layer4" => Case::Layer(4),
                "first4" => Case::FirstLayers(4),
                "main_all" => Case::MainAll,
                _ => panic!("invalid case {part:?}; expected layer0, layer4, first4, or main_all"),
            }
        })
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "--cases must include at least one case");
    cases
}

fn hidden_fixture(num_tokens: usize, hidden_dim: usize) -> Vec<u16> {
    (0..num_tokens * hidden_dim)
        .map(|index| bf16::from_f32(((index % 23) as f32 - 11.0) * 0.03125).to_bits())
        .collect()
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{name} requires a value; pass --help for usage"))
}

fn parse_u32_list(value: &str, name: &str) -> Vec<u32> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse()
                .unwrap_or_else(|err| panic!("invalid {name} value {part:?}: {err}"))
        })
        .collect::<Vec<_>>();
    assert!(!values.is_empty(), "{name} must include at least one value");
    values
}

fn parse_usize_arg(value: &str, name: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {name} value {value:?}: {err}"))
}

fn print_help_and_exit() -> ! {
    println!("qwen35_layers bench");
    println!();
    println!("Usage: cargo bench -p inference-executor-metal --bench qwen35_layers -- [options]");
    println!();
    println!("Options:");
    println!("--model-dir PATH");
    println!("--cases layer0,layer4,first4,main_all");
    println!("--tokens 1,2,4");
    println!("--contexts 0,32,128");
    println!("--iters N");
    println!("--warmup-iters N");
    println!("--runs N");
    std::process::exit(0);
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
    model_dir: &std::path::Path,
    case: Case,
    shape: BenchShape,
    iters: usize,
    replay: &ReplayProgram,
    samples: &[f64],
) {
    let median_us = median(samples);
    let stats = replay.stats();
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=qwen35-layer impl=layer-forward-replay model_dir={} case={} num_tokens={} ctx={} commands={} \
         retained_buffers={} retained_pipelines={} constant_bytes={} iters={iters} runs={} median_us={median_us:.3} \
         samples_us=[{sample_text}]",
        model_dir.display(),
        case.key(),
        shape.num_tokens,
        shape.context,
        stats.command_count,
        stats.retained_buffer_count,
        stats.retained_pipeline_count,
        stats.parameter_buffer_bytes,
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
