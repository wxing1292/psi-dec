use std::mem::size_of;
use std::num::NonZeroUsize;
use std::path::Path;
use std::rc::Rc;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3::QWEN3_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;
use inference_executor_core::model::qwen::v3::init_qwen3_model_config;
use inference_executor_core::model::qwen::v3::weight_layout::Qwen3ModelWeightBindings;
use inference_executor_core::model::qwen::v3::weight_layout::resolve_qwen3_model_weight_bindings;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkMainConfig;
use inference_executor_core::model::qwen::v3_x::dspark::init_qwen3x_dspark_config;
use inference_executor_core::sampling::HFGenerationConfig;
use inference_executor_core::sampling::MAX_TOP_K;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_runtime_core::runtime::Token;

use crate::checkpoint::SafeTensorStore;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedConfig;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::page_arena::PageArena;
use crate::model::qwen::v3::executor::Qwen3DSparkSpeculator;
use crate::model::qwen::v3::executor::Qwen3Executor;
use crate::model::qwen::v3::executor::Qwen3PendingTransactions;
use crate::model::qwen::v3::executor::Qwen3Speculator;
use crate::model::qwen::v3::executor::Qwen3WeightSource;
use crate::model::qwen::v3::executor::compact_target_distribution_indices;
use crate::model::qwen::v3::executor::num_page_ids_per_block;
use crate::model::qwen::v3::main::Qwen3Main;
use crate::model::qwen::v3::main::embed::Qwen3MainEmbed;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQAState;
use crate::model::qwen::v3::main::layer::Qwen3MainLayerScratch;
use crate::model::qwen::v3::main::output::Qwen3GatherUnembed;
use crate::model::qwen::v3::main::plan::qwen3_dense_mlp_core_and_metal;
use crate::model::qwen::v3::main::plan::qwen3_gqa_core_and_metal;
use crate::model::qwen::v3_x::dspark::execution::Qwen3xDSparkExecution;
use crate::model::qwen::v3_x::dspark::load::Qwen3xDSparkLoadConfig;
use crate::model::qwen::v3_x::dspark::load::Qwen3xDSparkLoaded;
use crate::model::qwen::v3_x::dspark::load::load_qwen3x_dspark;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedConfig;
use crate::replay::Replay;
use crate::sampling::rejection_replay::RejectionSampler;
use crate::sampling::rejection_replay::RejectionSampling;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3ExecutorConfig {
    pub max_requests: usize,
    pub max_tokens: usize,
    pub max_tokens_per_request: usize,
    pub num_cache_pages: usize,
    pub num_tokens_per_block: usize,
}

impl Qwen3ExecutorConfig {
    pub fn validate(self) {
        assert!(self.max_requests > 0, "qwen3 replay executor requires max_requests > 0");
        assert!(self.max_tokens > 0, "qwen3 replay executor requires max_tokens > 0");
        assert!(
            self.max_tokens_per_request > 0,
            "qwen3 replay executor requires max_tokens_per_request > 0"
        );
        assert!(
            self.num_cache_pages > 0,
            "qwen3 replay executor requires num_cache_pages > 0"
        );
        assert!(
            self.num_tokens_per_block > 0,
            "qwen3 replay executor requires num_tokens_per_block > 0"
        );
        assert!(
            u32::try_from(self.max_requests).is_ok(),
            "qwen3 max_requests must fit the u32 request-slot domain"
        );
        assert!(
            i32::try_from(self.max_tokens).is_ok(),
            "qwen3 max_tokens must fit the i32 flattened-token domain"
        );
        assert!(
            u32::try_from(self.max_tokens_per_request).is_ok(),
            "qwen3 max_tokens_per_request must fit the u32 position domain"
        );
        assert!(
            u32::try_from(self.num_tokens_per_block).is_ok(),
            "qwen3 num_tokens_per_block must fit the u32 cache-block domain"
        );
        assert!(
            u32::try_from(self.num_cache_pages - 1).is_ok(),
            "qwen3 cache page IDs must fit u32"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen3ModelLayout {
    pub max_tokens: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub scale_bias_dtype: Dtype,
    pub hidden_dtype: Dtype,
    pub rms_norm_eps: f32,
}

impl Qwen3ModelLayout {
    fn from_model_config(model_config: &Qwen3ModelConfig, max_tokens: usize) -> Result<Self, ModelExecutorError> {
        let text = &model_config.text_config;
        let quant = model_config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("qwen3 replay model requires quantization config"))?;
        Ok(Self {
            max_tokens: max_tokens
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3 max_tokens must fit u32"))?,
            vocab_size: text
                .vocab_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3 vocab_size must fit u32"))?,
            hidden_dim: text
                .hidden_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3 hidden_size must fit u32"))?,
            group_size: quant
                .group_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3 quantization group_size must fit u32"))?,
            bits: quant
                .bits
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3 quantization bits must fit u32"))?,
            scale_bias_dtype: Dtype::Bfloat16,
            hidden_dtype: Dtype::Bfloat16,
            rms_norm_eps: text.rms_norm_eps,
        })
    }

    fn validate(self) {
        assert!(self.max_tokens > 0, "qwen3 replay model requires positive max_tokens");
        assert!(self.vocab_size > 0, "qwen3 replay model requires positive vocab_size");
        assert!(self.hidden_dim > 0, "qwen3 replay model requires positive hidden_dim");
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.hidden_dim % self.group_size, 0);
        assert!(matches!(self.scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert_eq!(self.hidden_dtype, Dtype::Bfloat16);
        assert!(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0);
        i32::try_from(self.vocab_size).expect("qwen3 vocab index must fit i32");
        i32::try_from(self.hidden_dim).expect("qwen3 hidden dimension must fit i32");
        self.max_tokens
            .checked_mul(self.hidden_dim)
            .expect("qwen3 flattened hidden tensor index must fit u32");
    }

    fn embedding_config(self) -> EmbedConfig {
        EmbedConfig {
            max_tokens: self.max_tokens,
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            scale_bias_dtype: self.scale_bias_dtype,
            output_dtype: self.hidden_dtype,
        }
    }

    fn unembed_config(self) -> UnembedConfig {
        UnembedConfig {
            max_tokens: self.max_tokens,
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.hidden_dtype,
            output_dtype: self.hidden_dtype,
            scale_bias_dtype: self.scale_bias_dtype,
        }
    }

    fn hidden_bytes(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .and_then(|elements| elements.checked_mul(self.hidden_dtype.item_size()))
            .expect("qwen3 hidden buffer byte length must fit usize")
    }

    fn token_id_bytes(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(size_of::<i32>())
            .expect("qwen3 token ID buffer byte length must fit usize")
    }
}

enum Qwen3InitMode<'a> {
    Vanilla,
    DSpark {
        model_dir: &'a Path,
        config: Box<Qwen3xDSparkConfig>,
        num_spec_tokens: NonZeroUsize,
    },
}

enum Qwen3SpecLoad {
    Vanilla,
    DSpark(Box<Qwen3xDSparkLoaded>),
}

pub fn init_qwen_3_model(
    model_dir: impl AsRef<Path>,
    config: Qwen3ExecutorConfig,
) -> Result<Qwen3Executor, ModelExecutorError> {
    init_qwen_3_model_inner(model_dir.as_ref(), Qwen3InitMode::Vanilla, config)
}

pub fn init_qwen_3_model_with_dspark(
    model_dir: impl AsRef<Path>,
    dspark_model_dir: impl AsRef<Path>,
    requested_num_spec_tokens: Option<NonZeroUsize>,
    config: Qwen3ExecutorConfig,
) -> Result<Qwen3Executor, ModelExecutorError> {
    let dspark_model_dir = dspark_model_dir.as_ref();
    let dspark_config = init_qwen3x_dspark_config(dspark_model_dir)?;
    let num_spec_tokens = dspark_config.resolve_num_spec_tokens(requested_num_spec_tokens)?;
    init_qwen_3_model_inner(
        model_dir.as_ref(),
        Qwen3InitMode::DSpark {
            model_dir: dspark_model_dir,
            config: Box::new(dspark_config),
            num_spec_tokens,
        },
        config,
    )
}

fn init_qwen_3_model_inner(
    model_dir: &Path,
    init_mode: Qwen3InitMode<'_>,
    config: Qwen3ExecutorConfig,
) -> Result<Qwen3Executor, ModelExecutorError> {
    config.validate();
    let model_config = init_qwen3_model_config(model_dir)?;
    if let Qwen3InitMode::DSpark {
        config: dspark_config, ..
    } = &init_mode
    {
        let text = &model_config.text_config;
        dspark_config.validate_main(Qwen3xDSparkMainConfig {
            hidden_size: text.hidden_size,
            num_hidden_layers: text.num_hidden_layers,
            vocab_size: text.vocab_size,
            max_position_embeddings: text.max_position_embeddings,
            rope_theta: text.rope_theta,
        })?;
    }
    let generation_config = HFGenerationConfig::load(model_dir)?;
    let eos_token_ids = if generation_config.eos_token_ids().is_empty() {
        model_config.eos_token_ids()
    } else {
        generation_config.eos_token_ids()
    };
    let default_stop_sequences = eos_token_ids
        .iter()
        .map(|&token_id| vec![Token::new(token_id)])
        .collect();

    let layout = Qwen3ModelLayout::from_model_config(&model_config, config.max_tokens)?;
    layout.validate();

    let text = &model_config.text_config;
    let (main_gqa_core, main_gqa_metal) = qwen3_gqa_core_and_metal(0, &model_config)?;
    let gqa_tokens_per_page = main_gqa_metal.num_ungated_tokens_per_page(&main_gqa_core) as usize;
    let main_page_ids_per_block = num_page_ids_per_block(config.num_tokens_per_block, gqa_tokens_per_page);
    let gqa_page_table_layout = GQAPageTableLayout {
        num_req_slots: config.max_requests.try_into().expect("qwen3 max_requests must fit u32"),
        num_blocks: text
            .max_position_embeddings
            .div_ceil(config.num_tokens_per_block)
            .max(1)
            .try_into()
            .expect("qwen3 GQA block capacity must fit u32"),
        num_gqa_layers: text
            .num_hidden_layers
            .try_into()
            .expect("qwen3 GQA layer count must fit u32"),
        num_page_ids_per_block: main_page_ids_per_block
            .try_into()
            .expect("qwen3 GQA pages per block must fit u32"),
    };
    let device = Device::system_default();
    let runtime = MetalRuntime::new(device.clone());
    let mut store = SafeTensorStore::from_model_dir(model_dir)?;
    let weight_bindings = resolve_qwen3_model_weight_bindings(&model_config, store.index().tensor_names())?;
    let main_gqa_state = Qwen3MainGQAState::new(
        &device,
        main_gqa_core,
        main_gqa_metal,
        gqa_page_table_layout,
        config.max_tokens,
        config.num_cache_pages,
    );
    let layer_scratch = Rc::new(Qwen3MainLayerScratch::new(
        &device,
        config.max_tokens,
        layout.hidden_dim as usize,
    ));
    let (dense_core, dense_metal) = qwen3_dense_mlp_core_and_metal(0, &model_config)?;
    let dense_scratch = Rc::new(DenseMLPScratch::new(
        &device,
        &dense_core,
        dense_metal.io_dtype,
        config.max_tokens,
    ));

    let Qwen3ModelWeightBindings {
        embed: embed_bindings,
        main: main_bindings,
        unembed: unembed_bindings,
    } = weight_bindings;
    let mut embed = Embed::new(&device, layout.embedding_config());
    embed.load_weights(&device, &mut store, embed_bindings)?;
    let embed = Rc::new(embed);
    let token_hidden_input = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    let hidden_output = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    let unembed_config = layout.unembed_config();
    let mut unembed = Unembed::new(&device, unembed_config);
    unembed.load_weights(&device, &mut store, unembed_bindings)?;
    let gather_unembed = Qwen3GatherUnembed::new(&device, unembed_config.hidden_dim, Rc::new(unembed));
    let sampler_bounds = TopKSamplingBounds {
        max_sampling_inputs: layout.max_tokens,
        vocab_size: layout.vocab_size,
        top_k: MAX_TOP_K.try_into().expect("qwen3 sampler top_k must fit u32"),
    };
    sampler_bounds.validate();
    let spec_load = match &init_mode {
        Qwen3InitMode::Vanilla => Qwen3SpecLoad::Vanilla,
        Qwen3InitMode::DSpark {
            model_dir: dspark_model_dir,
            config: dspark_config,
            num_spec_tokens,
        } => {
            Qwen3SpecLoad::DSpark(Box::new(load_qwen3x_dspark(
                &device,
                dspark_model_dir,
                dspark_config,
                Qwen3xDSparkLoadConfig {
                    num_spec_tokens: *num_spec_tokens,
                    page_size_bytes: QWEN3_PAGE_SIZE_BYTES,
                    max_position_embeddings: text.max_position_embeddings,
                    max_requests: config.max_requests,
                    max_tokens: config.max_tokens,
                    num_cache_pages: config.num_cache_pages,
                    num_tokens_per_block: config.num_tokens_per_block,
                },
                Rc::clone(&embed),
                gather_unembed.unembed(),
                sampler_bounds,
            )?))
        },
    };
    let residual_capture = match &spec_load {
        Qwen3SpecLoad::Vanilla => None,
        Qwen3SpecLoad::DSpark(loaded) => {
            let capture: Rc<dyn MainResidualCapture> = loaded.model.main_feature_projector();
            Some(capture)
        },
    };
    let mut main = Qwen3Main::new(
        &device,
        &model_config,
        &main_gqa_state,
        residual_capture,
        layer_scratch,
        &dense_scratch,
    )?;
    main.load_weights(&device, &mut store, &model_config, main_bindings)?;
    drop(store);

    let sampler = Rc::new(TopKSampling::new(&device, sampler_bounds));
    let (speculator, num_dspark_gqa_page_ids_per_block) = match spec_load {
        Qwen3SpecLoad::Vanilla => (Qwen3Speculator::Vanilla, 0),
        Qwen3SpecLoad::DSpark(loaded) => {
            let execution = Qwen3xDSparkExecution::new(&device, *loaded, config.max_requests, unembed_config);
            let num_spec_tokens = execution.num_spec_tokens();
            let num_gqa_page_ids_per_block = execution.num_gqa_page_ids_per_block();
            let max_target_distributions = config
                .max_requests
                .checked_mul(
                    num_spec_tokens
                        .checked_add(1)
                        .expect("Qwen3 target distributions per request must fit usize"),
                )
                .expect("Qwen3 target distribution capacity must fit usize");
            let rejector = Rc::new(RejectionSampler::new(
                &device,
                num_spec_tokens,
                config.max_requests,
                sampler_bounds.top_k,
            ));
            (
                Qwen3Speculator::DSpark(Box::new(Qwen3DSparkSpeculator {
                    execution,
                    rejection_sampling: Replay::new(
                        "qwen3 rejection sampling",
                        RejectionSampling::new(Rc::clone(&sampler), rejector),
                    ),
                    spec_probs: SpecProbsStore::new(&device, num_spec_tokens, config.max_requests, MAX_TOP_K),
                    target_distribution_indices: Buffer::from_slice(
                        &device,
                        &compact_target_distribution_indices(max_target_distributions),
                    ),
                })),
                num_gqa_page_ids_per_block,
            )
        },
    };
    let num_main_gqa_page_ids_per_block = usize::try_from(gqa_page_table_layout.num_gqa_layers)
        .expect("qwen3 Main GQA layer count must fit usize")
        .checked_mul(
            usize::try_from(gqa_page_table_layout.num_page_ids_per_block)
                .expect("qwen3 Main GQA page count must fit usize"),
        )
        .expect("qwen3 Main page IDs per block must fit usize");
    let num_gqa_page_ids_per_main_lane_block = num_main_gqa_page_ids_per_block
        .checked_add(num_dspark_gqa_page_ids_per_block)
        .expect("Qwen3 Main cache-lane page IDs per block must fit usize");
    let weight_source = match &init_mode {
        Qwen3InitMode::Vanilla => Qwen3WeightSource::Vanilla,
        Qwen3InitMode::DSpark { model_dir, config, .. } => {
            Qwen3WeightSource::DSpark {
                model_dir: model_dir.to_path_buf(),
                config: config.clone(),
            }
        },
    };
    Ok(Qwen3Executor {
        model_name: "qwen3".to_string(),
        model_dir: model_dir.to_path_buf(),
        weight_source,
        model_config,
        default_stop_sequences,
        config,
        runtime,
        layout,
        token_ids: Buffer::new_zeroed(&device, layout.token_id_bytes()),
        token_hidden_input,
        hidden_output,
        gather_flat_indices: Buffer::new_zeroed_elements(&device, config.max_tokens, Dtype::Uint32),
        unembed_hidden: Buffer::new_zeroed(&device, layout.hidden_bytes()),
        unembed_logits: Buffer::new_zeroed(&device, unembed_config.logits_bytes()),
        main_embed: Replay::new("qwen3 MainEmbed", Qwen3MainEmbed::new(embed)),
        main: Replay::new("qwen3 Main", main),
        gather_unembed: Replay::new("qwen3 GatherUnembed", gather_unembed),
        sampling: Replay::new(
            "qwen3 sampling",
            Sampling {
                sampler: Rc::clone(&sampler),
            },
        ),
        sampler,
        sampler_bounds,
        sampler_output: TopKSamplingOutputBuffers::new(&device, sampler_bounds),
        request_sampling: RequestSamplingState::new(config.max_requests),
        main_gqa_state,
        speculator,
        pages: PageArena::new(&device, config.num_cache_pages, QWEN3_PAGE_SIZE_BYTES),
        pending_transactions: Qwen3PendingTransactions::new(),
        gqa_page_table_layout,
        num_gqa_page_ids_per_main_lane_block,
        state_fingerprint: crate::model::state_snapshot::ModelFingerprint::for_process_instance("qwen3"),
        unloaded_embed: None,
        unloaded_unembed: None,
    })
}
