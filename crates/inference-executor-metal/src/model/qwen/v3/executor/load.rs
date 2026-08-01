use std::mem::size_of;
use std::path::Path;
use std::rc::Rc;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::DSparkBlockCapacity;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3::QWEN3_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;
use inference_executor_core::model::qwen::v3::init_qwen3_model_config;
use inference_executor_core::model::qwen::v3::weight_layout::Qwen3ModelWeightBindings;
use inference_executor_core::model::qwen::v3::weight_layout::resolve_qwen3_model_weight_bindings;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkMainConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkWeightBindings;
use inference_executor_core::model::qwen::v3_x::dspark::init_qwen3x_dspark_config;
use inference_executor_core::model::qwen::v3_x::dspark::resolve_qwen3x_dspark_weight_bindings;
use inference_executor_core::sampling::HFGenerationConfig;
use inference_executor_core::sampling::MAX_TOP_K;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_runtime_core::runtime::Token;

use crate::attn::dspark::state::UngatedDSparkGQAState;
use crate::checkpoint::SafeTensorStore;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedConfig;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::page_arena::PageArena;
use crate::model::qwen::v3::executor::Qwen3Executor;
use crate::model::qwen::v3::executor::Qwen3PendingTransactions;
use crate::model::qwen::v3::executor::compact_target_distribution_indices;
use crate::model::qwen::v3::executor::num_page_ids_per_block;
use crate::model::qwen::v3::main::Qwen3Main;
use crate::model::qwen::v3::main::embed::Qwen3MainEmbed;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQAState;
use crate::model::qwen::v3::main::layer::Qwen3MainLayerScratch;
use crate::model::qwen::v3::main::output::Qwen3GatherUnembed;
use crate::model::qwen::v3::main::plan::qwen3_dense_mlp_core_and_metal;
use crate::model::qwen::v3::main::plan::qwen3_gqa_core_and_metal;
use crate::model::qwen::v3_x::dspark::attention::qwen3x_dspark_gqa_compute_config;
use crate::model::qwen::v3_x::dspark::attention::qwen3x_dspark_gqa_core;
use crate::model::qwen::v3_x::dspark::embed::Qwen3xDSparkEmbed;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkBody;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkContext;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkModel;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkGatherUnembed;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkSampling;
use crate::model::qwen::v3_x::dspark::sampling::Qwen3xDSparkMarkov;
use crate::model::qwen::v3_x::weight::to_u32;
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

struct Qwen3DSparkLoaded {
    page_table_layout: GQAPageTableLayout,
    gqa_state: UngatedDSparkGQAState,
    model: Rc<Qwen3xDSparkModel>,
    embed: Rc<Embed>,
    unembed: Rc<Unembed>,
    markov: Rc<Qwen3xDSparkMarkov>,
    block_size: usize,
    mask_token_id: i32,
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

pub fn init_qwen_3_model(
    model_dir: impl AsRef<Path>,
    config: Qwen3ExecutorConfig,
) -> Result<Qwen3Executor, ModelExecutorError> {
    init_qwen_3_model_inner(model_dir.as_ref(), None, config)
}

pub fn init_qwen_3_model_with_dspark(
    model_dir: impl AsRef<Path>,
    dspark_model_dir: impl AsRef<Path>,
    config: Qwen3ExecutorConfig,
) -> Result<Qwen3Executor, ModelExecutorError> {
    init_qwen_3_model_inner(model_dir.as_ref(), Some(dspark_model_dir.as_ref()), config)
}

fn init_qwen_3_model_inner(
    model_dir: &Path,
    dspark_model_dir: Option<&Path>,
    config: Qwen3ExecutorConfig,
) -> Result<Qwen3Executor, ModelExecutorError> {
    config.validate();
    let model_config = init_qwen3_model_config(model_dir)?;
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
    let main_gqa_state = Qwen3MainGQAState::load(
        &device,
        main_gqa_core,
        main_gqa_metal,
        gqa_page_table_layout,
        config.max_tokens,
        config.num_cache_pages,
        0,
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
    let embed = Rc::new(Embed::load(
        &device,
        &mut store,
        layout.embedding_config(),
        embed_bindings,
    )?);
    let token_hidden_input = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    let hidden_output = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    let unembed_config = layout.unembed_config();
    let gather_unembed = Qwen3GatherUnembed::load(&device, &mut store, unembed_config, unembed_bindings)?;
    let sampler_bounds = TopKSamplingBounds {
        max_sampling_inputs: layout.max_tokens,
        vocab_size: layout.vocab_size,
        top_k: MAX_TOP_K.try_into().expect("qwen3 sampler top_k must fit u32"),
    };
    sampler_bounds.validate();
    let dspark_loaded = dspark_model_dir
        .map(|dspark_model_dir| {
            load_qwen3_dspark(
                &device,
                dspark_model_dir,
                &model_config,
                config,
                Rc::clone(&embed),
                gather_unembed.unembed(),
                sampler_bounds,
            )
        })
        .transpose()?;
    let residual_capture = dspark_loaded.as_ref().map(|loaded| {
        let capture: Rc<dyn MainResidualCapture> = loaded.model.main_feature_projector();
        capture
    });
    let main = Qwen3Main::load(
        &device,
        &mut store,
        &model_config,
        main_bindings,
        &main_gqa_state,
        residual_capture,
        layer_scratch,
        &dense_scratch,
    )?;
    drop(store);

    let sampler = Rc::new(TopKSampling::new(&device, sampler_bounds));
    let dspark_block_size = dspark_loaded.as_ref().map_or(0, |loaded| loaded.block_size);
    let num_main_page_ids_per_block = usize::try_from(gqa_page_table_layout.num_gqa_layers)
        .expect("qwen3 Main GQA layer count must fit usize")
        .checked_mul(
            usize::try_from(gqa_page_table_layout.num_page_ids_per_block)
                .expect("qwen3 Main GQA page count must fit usize"),
        )
        .expect("qwen3 Main page IDs per block must fit usize");
    let num_dspark_page_ids_per_block = dspark_loaded.as_ref().map_or(0, |loaded| {
        usize::try_from(loaded.page_table_layout.num_gqa_layers)
            .expect("Qwen3 DSpark GQA layer count must fit usize")
            .checked_mul(
                usize::try_from(loaded.page_table_layout.num_page_ids_per_block)
                    .expect("Qwen3 DSpark GQA page count must fit usize"),
            )
            .expect("Qwen3 DSpark page IDs per block must fit usize")
    });
    let num_runtime_page_ids_per_block = num_main_page_ids_per_block
        .checked_add(num_dspark_page_ids_per_block)
        .expect("Qwen3 runtime page IDs per block must fit usize");
    let max_target_distributions = config
        .max_requests
        .checked_mul(
            dspark_block_size
                .checked_add(1)
                .expect("Qwen3 target distributions per request must fit usize"),
        )
        .expect("Qwen3 target distribution capacity must fit usize");
    let (
        dspark_context,
        dspark_gqa_state,
        dspark_embed,
        dspark,
        dspark_gather_unembed,
        dspark_sampling,
        dspark_markov,
        dspark_hidden_input,
        dspark_hidden_output,
        dspark_unembed_hidden,
        dspark_logits,
        dspark_page_table_layout,
        dspark_mask_token_id,
    ) = if let Some(loaded) = dspark_loaded {
        let max_block_tokens = config
            .max_requests
            .checked_mul(loaded.block_size)
            .expect("Qwen3 DSpark block token capacity must fit usize");
        let hidden_bytes = max_block_tokens
            .checked_mul(layout.hidden_dim as usize)
            .and_then(|elements| elements.checked_mul(Dtype::Bfloat16.item_size()))
            .expect("Qwen3 DSpark hidden byte capacity must fit usize");
        let dspark_unembed_config = UnembedConfig {
            max_tokens: max_block_tokens
                .try_into()
                .expect("Qwen3 DSpark unembed rows must fit u32"),
            ..unembed_config
        };
        (
            Some(Replay::new(
                "qwen3 DSparkContext",
                Qwen3xDSparkContext::new(Rc::clone(&loaded.model)),
            )),
            Some(loaded.gqa_state),
            Some(Replay::new("qwen3 DSparkEmbed", Qwen3xDSparkEmbed::new(loaded.embed))),
            Some(Replay::new(
                "qwen3 DSpark",
                Qwen3xDSparkBody::new(Rc::clone(&loaded.model)),
            )),
            Some(Replay::new(
                "qwen3 DSpark GatherUnembed",
                Qwen3xDSparkGatherUnembed::new(
                    &device,
                    loaded.block_size,
                    config.max_requests,
                    layout.hidden_dim,
                    loaded.unembed,
                ),
            )),
            Some(Replay::new(
                "qwen3 DSpark Sampling",
                Qwen3xDSparkSampling::new(Rc::clone(&loaded.markov)),
            )),
            Some(loaded.markov),
            Some(Rc::new(Buffer::new_zeroed(&device, hidden_bytes))),
            Some(Rc::new(Buffer::new_zeroed(&device, hidden_bytes))),
            Some(Buffer::new_zeroed(&device, hidden_bytes)),
            Some(Buffer::new_zeroed(&device, dspark_unembed_config.logits_bytes())),
            Some(loaded.page_table_layout),
            Some(loaded.mask_token_id),
        )
    } else {
        (
            None, None, None, None, None, None, None, None, None, None, None, None, None,
        )
    };
    let rejection_sampler = (dspark_block_size > 0).then(|| {
        Rc::new(RejectionSampler::new(
            &device,
            dspark_block_size,
            config.max_requests,
            sampler_bounds.top_k,
        ))
    });
    let rejection_sampling = rejection_sampler.as_ref().map(|rejector| {
        Replay::new(
            "qwen3 rejection sampling",
            RejectionSampling::new(Rc::clone(&sampler), Rc::clone(rejector)),
        )
    });
    Ok(Qwen3Executor {
        model_name: "qwen3".to_string(),
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
        dspark_context,
        gather_unembed: Replay::new("qwen3 GatherUnembed", gather_unembed),
        sampling: Replay::new(
            "qwen3 sampling",
            Sampling {
                sampler: Rc::clone(&sampler),
            },
        ),
        rejection_sampling,
        sampler,
        sampler_bounds,
        sampler_output: TopKSamplingOutputBuffers::new(&device, sampler_bounds),
        request_sampling: RequestSamplingState::new(config.max_requests),
        main_gqa_state,
        dspark_gqa_state,
        dspark_embed,
        dspark,
        dspark_gather_unembed,
        dspark_sampling,
        dspark_markov,
        dspark_hidden_input,
        dspark_hidden_output,
        dspark_unembed_hidden,
        dspark_logits,
        spec_probs: SpecProbsStore::new(&device, dspark_block_size, config.max_requests, MAX_TOP_K),
        target_distribution_indices: Buffer::from_slice(
            &device,
            &compact_target_distribution_indices(max_target_distributions.max(1)),
        ),
        pages: PageArena::new(&device, config.num_cache_pages, QWEN3_PAGE_SIZE_BYTES),
        pending_transactions: Qwen3PendingTransactions::new(),
        gqa_page_table_layout,
        dspark_page_table_layout,
        num_runtime_page_ids_per_block,
        dspark_block_size,
        dspark_mask_token_id,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_qwen3_dspark(
    device: &Device,
    model_dir: &Path,
    main_config: &Qwen3ModelConfig,
    executor_config: Qwen3ExecutorConfig,
    main_embed: Rc<Embed>,
    main_unembed: Rc<Unembed>,
    sampler_bounds: TopKSamplingBounds,
) -> Result<Qwen3DSparkLoaded, ModelExecutorError> {
    let config = init_qwen3x_dspark_config(model_dir)?;
    let text = &main_config.text_config;
    config.validate_main(Qwen3xDSparkMainConfig {
        hidden_size: text.hidden_size,
        num_hidden_layers: text.num_hidden_layers,
        vocab_size: text.vocab_size,
        max_position_embeddings: text.max_position_embeddings,
        rope_theta: text.rope_theta,
    })?;
    let mut store = SafeTensorStore::from_model_dir(model_dir)?;
    let Qwen3xDSparkWeightBindings {
        embed: embed_bindings,
        main_feature: main_feature_bindings,
        layers: layer_bindings,
        final_norm_weight,
        unembed: unembed_bindings,
        markov: markov_bindings,
        confidence: confidence_bindings,
    } = resolve_qwen3x_dspark_weight_bindings(&config, store.index().tensor_names())?;
    let attention_core = qwen3x_dspark_gqa_core(&config, 0);
    let attention_compute_config = qwen3x_dspark_gqa_compute_config(&config, QWEN3_PAGE_SIZE_BYTES)?;
    let tokens_per_page = attention_compute_config.num_tokens_per_page() as usize;
    let page_ids_per_block = num_page_ids_per_block(executor_config.num_tokens_per_block, tokens_per_page);
    let page_table_layout = GQAPageTableLayout {
        num_req_slots: executor_config
            .max_requests
            .try_into()
            .expect("Qwen3 DSpark max_requests must fit u32"),
        num_blocks: text
            .max_position_embeddings
            .div_ceil(executor_config.num_tokens_per_block)
            .max(1)
            .try_into()
            .expect("Qwen3 DSpark block capacity must fit u32"),
        num_gqa_layers: config
            .num_hidden_layers
            .try_into()
            .expect("Qwen3 DSpark layer count must fit u32"),
        num_page_ids_per_block: page_ids_per_block
            .try_into()
            .expect("Qwen3 DSpark pages per block must fit u32"),
    };
    let capacity = DSparkBlockCapacity::new(executor_config.max_requests, config.block_size);
    if capacity.max_tokens > executor_config.max_tokens {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark proposal capacity={} exceeds executor max_tokens={}; increase --max-tokens or reduce the \
             executor request capacity",
            capacity.max_tokens, executor_config.max_tokens
        )));
    }
    let gqa_state = UngatedDSparkGQAState::new(
        device,
        attention_core,
        attention_compute_config,
        page_table_layout,
        capacity,
        executor_config.max_tokens,
        executor_config.num_cache_pages,
        0,
    );
    let quantization = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DSpark Metal executor requires quantization config"))?;
    let max_block_tokens = capacity.max_tokens;
    let embed = if let Some(embed_bindings) = embed_bindings {
        let resolved = quantization.resolve_for_tensor(&embed_bindings.weight);
        Rc::new(Embed::load(
            device,
            &mut store,
            EmbedConfig {
                max_tokens: max_block_tokens
                    .try_into()
                    .expect("Qwen3 DSpark embed rows must fit u32"),
                vocab_size: config
                    .vocab_size
                    .try_into()
                    .expect("Qwen3 DSpark embedding vocabulary must fit u32"),
                hidden_dim: config
                    .hidden_size
                    .try_into()
                    .expect("Qwen3 DSpark embedding width must fit u32"),
                group_size: to_u32("Qwen3x DSpark embedding group_size", resolved.group_size)?,
                bits: to_u32("Qwen3x DSpark embedding bits", resolved.bits)?,
                scale_bias_dtype: Dtype::Bfloat16,
                output_dtype: Dtype::Bfloat16,
            },
            embed_bindings,
        )?)
    } else {
        main_embed
    };
    let unembed = if let Some(unembed_bindings) = unembed_bindings {
        let resolved = quantization.resolve_for_tensor(&unembed_bindings.weight);
        Rc::new(Unembed::load(
            device,
            &mut store,
            UnembedConfig {
                max_tokens: max_block_tokens
                    .try_into()
                    .expect("Qwen3 DSpark unembed rows must fit u32"),
                vocab_size: config
                    .vocab_size
                    .try_into()
                    .expect("Qwen3 DSpark unembed vocabulary must fit u32"),
                hidden_dim: config
                    .hidden_size
                    .try_into()
                    .expect("Qwen3 DSpark unembed hidden width must fit u32"),
                group_size: to_u32("Qwen3x DSpark unembed group_size", resolved.group_size)?,
                bits: to_u32("Qwen3x DSpark unembed bits", resolved.bits)?,
                input_dtype: Dtype::Bfloat16,
                output_dtype: Dtype::Bfloat16,
                scale_bias_dtype: Dtype::Bfloat16,
            },
            unembed_bindings,
        )?)
    } else {
        main_unembed
    };
    let markov = Rc::new(Qwen3xDSparkMarkov::load(
        device,
        &mut store,
        &config,
        &markov_bindings,
        confidence_bindings.as_ref(),
        executor_config.max_requests,
        sampler_bounds,
    )?);
    let model = Qwen3xDSparkModel::load(
        device,
        &mut store,
        &config,
        QWEN3_PAGE_SIZE_BYTES,
        main_feature_bindings,
        layer_bindings,
        final_norm_weight,
        &gqa_state,
        executor_config.max_tokens,
        max_block_tokens,
    )?;
    Ok(Qwen3DSparkLoaded {
        page_table_layout,
        gqa_state,
        model,
        embed,
        unembed,
        markov,
        block_size: config.block_size,
        mask_token_id: config
            .mask_token_id
            .try_into()
            .map_err(|_| ModelExecutorError::custom("Qwen3 DSpark MASK token ID must fit i32"))?,
    })
}
