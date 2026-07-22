use std::mem::size_of;
use std::path::Path;
use std::rc::Rc;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::Qwen35PendingTransactions;
use inference_executor_core::model::qwen::v3_5::init_model_config;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MTPWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_model_weight_bindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_mtp_weight_bindings;
use inference_executor_core::sampling::HFGenerationConfig;
use inference_executor_core::sampling::MAX_TOP_K;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_runtime_core::runtime::Token;

use super::Qwen35Executor;
use super::num_page_ids_per_block;
use crate::checkpoint::SafeTensorStore;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::embed_unembed::Embed;
use crate::model::embed_unembed::EmbedConfig;
use crate::model::embed_unembed::UnembedConfig;
use crate::model::page_arena::PageArena;
use crate::model::qwen::v3_5::layer::scratch::Qwen35LayerScratch;
use crate::model::qwen::v3_5::model::Qwen35GatherUnembed;
use crate::model::qwen::v3_5::model::Qwen35Main;
use crate::model::qwen::v3_5::model::Qwen35MainEmbed;
use crate::model::qwen::v3_5::mtp::Qwen35MTP;
use crate::model::qwen::v3_5::mtp::Qwen35MTPEmbed;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::plan::qwen35_dense_mlp_core_and_metal;
use crate::model::qwen::v3_5::plan::qwen35_gdn_core_and_metal;
use crate::model::qwen::v3_5::plan::qwen35_gqa_core_and_metal;
use crate::model::qwen::v3_5::plan::qwen35_layer_counts;
use crate::model::qwen::v3_5::plan::qwen35_moe_core_and_metal;
use crate::model::qwen::v3_5::plan::validate_qwen35_mtp_config;
use crate::model::qwen::v3_5::rejection_sampling::Qwen35RejectionSampler;
use crate::model::qwen::v3_5::rejection_sampling::RejectionSampling;
use crate::model::qwen::v3_5::state::Qwen35GDNState;
use crate::model::qwen::v3_5::state::Qwen35GQAState;
use crate::replay::Replay;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_replay::DraftSampling;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::trace;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35ExecutorConfig {
    pub max_requests: usize,
    pub max_tokens: usize,
    pub max_tokens_per_request: usize,
    pub num_cache_pages: usize,
    pub num_tokens_per_block: usize,
    pub num_mtp_modules: usize,
}

impl Qwen35ExecutorConfig {
    pub fn validate(self) {
        assert!(
            self.max_requests > 0,
            "qwen3.5 replay executor requires max_requests > 0"
        );
        assert!(self.max_tokens > 0, "qwen3.5 replay executor requires max_tokens > 0");
        assert!(
            self.max_tokens_per_request > 0,
            "qwen3.5 replay executor requires max_tokens_per_request > 0"
        );
        let min_mtp_forward_tokens = self
            .num_mtp_modules
            .checked_add(1)
            .expect("qwen3.5 MTP forward token count overflow");
        assert!(
            min_mtp_forward_tokens <= self.max_tokens_per_request,
            "qwen3.5 MTP forward tokens={} exceed scheduler max_tokens_per_request={}",
            min_mtp_forward_tokens,
            self.max_tokens_per_request
        );
        assert!(
            self.num_cache_pages > 0,
            "qwen3.5 replay executor requires num_cache_pages > 0"
        );
        assert!(
            self.num_tokens_per_block > 0,
            "qwen3.5 replay executor requires num_tokens_per_block > 0"
        );
        assert!(
            u32::try_from(self.max_requests).is_ok(),
            "qwen3.5 max_requests must fit the u32 request-slot domain"
        );
        assert!(
            i32::try_from(self.max_tokens).is_ok(),
            "qwen3.5 max_tokens must fit the i32 flattened-token domain"
        );
        assert!(
            u32::try_from(self.max_tokens_per_request).is_ok(),
            "qwen3.5 max_tokens_per_request must fit the u32 position/state-version domain"
        );
        assert!(
            u32::try_from(self.num_tokens_per_block).is_ok(),
            "qwen3.5 num_tokens_per_block must fit the u32 cache-block domain"
        );
        assert!(self.num_mtp_modules <= 1, "qwen3.5 supports zero or one MTP module");
        assert!(
            u32::try_from(self.num_cache_pages - 1).is_ok(),
            "qwen3.5 cache page IDs must fit u32"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35ModelLayout {
    pub max_tokens: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub affine_dtype: Dtype,
    pub hidden_dtype: Dtype,
    pub rms_norm_eps: f32,
}

impl Qwen35ModelLayout {
    fn from_model_config(model_config: &Qwen35ModelConfig, max_tokens: usize) -> Result<Self, ModelExecutorError> {
        let text = &model_config.text_config;
        let quant = model_config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("qwen3.5 replay model requires quantization config"))?;
        Ok(Self {
            max_tokens: max_tokens
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 max_tokens must fit u32"))?,
            vocab_size: text
                .vocab_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 vocab_size must fit u32"))?,
            hidden_dim: text
                .hidden_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 hidden_size must fit u32"))?,
            group_size: quant
                .group_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 quantization group_size must fit u32"))?,
            bits: quant
                .bits
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 quantization bits must fit u32"))?,
            affine_dtype: Dtype::Bfloat16,
            hidden_dtype: Dtype::Bfloat16,
            rms_norm_eps: text.rms_norm_eps,
        })
    }

    fn validate(self) {
        assert!(self.max_tokens > 0, "qwen3.5 replay model requires positive max_tokens");
        assert!(self.vocab_size > 0, "qwen3.5 replay model requires positive vocab_size");
        assert!(self.hidden_dim > 0, "qwen3.5 replay model requires positive hidden_dim");
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.hidden_dim % self.group_size, 0);
        assert!(matches!(self.affine_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert_eq!(self.hidden_dtype, Dtype::Bfloat16);
        assert!(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0);
        i32::try_from(self.vocab_size).expect("qwen3.5 vocab index must fit i32");
        i32::try_from(self.hidden_dim).expect("qwen3.5 hidden dimension must fit i32");
        i32::try_from(self.group_size).expect("qwen3.5 quantization group size must fit i32");
        i32::try_from(self.bits).expect("qwen3.5 quantization bits must fit i32");
        self.max_tokens
            .checked_mul(self.hidden_dim)
            .expect("qwen3.5 flattened hidden tensor index must fit u32");
    }

    fn embedding_config(self) -> EmbedConfig {
        EmbedConfig {
            max_tokens: self.max_tokens,
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            affine_dtype: self.affine_dtype,
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
            affine_dtype: self.affine_dtype,
        }
    }

    fn hidden_bytes(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .and_then(|elements| elements.checked_mul(self.hidden_dtype.item_size()))
            .expect("qwen3.5 hidden buffer byte length must fit usize")
    }

    fn token_id_bytes(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(size_of::<i32>())
            .expect("qwen3.5 token ID buffer byte length must fit usize")
    }
}

pub fn init_qwen_3_5_model(
    model_dir: impl AsRef<Path>,
    config: Qwen35ExecutorConfig,
) -> Result<Qwen35Executor, ModelExecutorError> {
    init_qwen_3_5_model_inner(model_dir.as_ref(), None, config)
}

pub fn init_qwen_3_5_model_with_hf_mtp(
    model_dir: impl AsRef<Path>,
    mtp_model_dir: impl AsRef<Path>,
    config: Qwen35ExecutorConfig,
) -> Result<Qwen35Executor, ModelExecutorError> {
    init_qwen_3_5_model_inner(model_dir.as_ref(), Some(mtp_model_dir.as_ref()), config)
}

fn init_qwen_3_5_model_inner(
    model_dir: &Path,
    mtp_model_dir: Option<&Path>,
    config: Qwen35ExecutorConfig,
) -> Result<Qwen35Executor, ModelExecutorError> {
    config.validate();
    let model_config = init_model_config(model_dir)?;
    if config.num_mtp_modules > 0 {
        assert!(
            mtp_model_dir.is_some(),
            "qwen3.5 replay model requires hf_mtp_model_dir when num_mtp_modules > 0"
        );
    }
    let generation_config = HFGenerationConfig::load(model_dir)?;
    let sampler_config = generation_config.sampler();
    let default_stop_sequences = generation_config
        .eos_token_ids()
        .iter()
        .map(|&token_id| vec![Token::new(token_id)])
        .collect();
    let device = Device::system_default();
    let runtime = MetalRuntime::new(device.clone());
    let mut store = SafeTensorStore::from_model_dir(model_dir)?;
    let weight_bindings = resolve_qwen35_model_weight_bindings(&model_config, store.index().tensor_names())?;
    let layer_counts = qwen35_layer_counts(&model_config)?;
    assert!(layer_counts.gqa > 0, "qwen3.5 Main requires at least one GQA layer");
    assert!(layer_counts.gdn > 0, "qwen3.5 Main requires at least one GDN layer");
    let metal_defaults = Qwen35MetalDefaults::from_quantization(model_config.quantization.as_ref())?;
    let layout = Qwen35ModelLayout::from_model_config(&model_config, config.max_tokens)?;
    layout.validate();
    let sampler_bounds = TopKSamplingBounds {
        max_sampling_inputs: layout.max_tokens,
        vocab_size: layout.vocab_size,
        top_k: MAX_TOP_K.try_into().expect("qwen3.5 sampler top_k must fit u32"),
    };
    sampler_bounds.validate();
    trace::qwen35_state(|| {
        format!(
            "event=sampler_config temperature={} top_k={} top_p={} bounds_top_k={} max_sampling_inputs={} \
             vocab_size={}",
            sampler_config.temperature,
            sampler_config.top_k,
            sampler_config.top_p,
            sampler_bounds.top_k,
            sampler_bounds.max_sampling_inputs,
            sampler_bounds.vocab_size
        )
    });
    let unembed_config = layout.unembed_config();
    let first_gqa_layer = (0..model_config.text_config.num_hidden_layers)
        .find(|&index| {
            model_config
                .layer_type_at(index)
                .is_ok_and(|kind| kind == inference_executor_core::model::qwen::v3_5::LayerType::FullAttention)
        })
        .expect("qwen3.5 Main requires a GQA layer");
    let (main_gqa_core, main_gqa_metal) =
        qwen35_gqa_core_and_metal(first_gqa_layer, &model_config.text_config, metal_defaults)?;
    let gqa_tokens_per_page = main_gqa_metal.num_tokens_per_page(&main_gqa_core) as usize;
    let main_page_ids_per_block = num_page_ids_per_block(config.num_tokens_per_block, gqa_tokens_per_page);
    let target_gqa_page_table_layout = GQAPageTableLayout {
        num_req_slots: config
            .max_requests
            .try_into()
            .expect("qwen3.5 max_requests must fit u32"),
        num_blocks: model_config
            .text_config
            .max_position_embeddings
            .div_ceil(config.num_tokens_per_block)
            .max(1)
            .try_into()
            .expect("qwen3.5 GQA block capacity must fit u32"),
        num_gqa_layers: layer_counts
            .gqa
            .try_into()
            .expect("qwen3.5 GQA layer count must fit u32"),
        num_page_ids_per_block: main_page_ids_per_block
            .try_into()
            .expect("qwen3.5 GQA pages per block must fit u32"),
    };
    let gqa_page_table_layout = target_gqa_page_table_layout;
    let mut mtp_load = if config.num_mtp_modules == 1 {
        let mtp_model_dir = mtp_model_dir.expect("qwen3.5 replay model checked MTP model dir");
        let mtp_model_config = init_model_config(mtp_model_dir)?;
        validate_qwen35_mtp_config(&model_config, &mtp_model_config)?;
        let mtp_store = SafeTensorStore::from_model_dir(mtp_model_dir)?;
        let mtp_weight_bindings =
            resolve_qwen35_mtp_weight_bindings(&mtp_model_config, 1, mtp_store.index().tensor_names())?;
        Some((mtp_model_config, mtp_store, mtp_weight_bindings))
    } else {
        None
    };
    let mtp_gqa_geometry = mtp_load
        .as_ref()
        .map(|(mtp_config, ..)| {
            qwen35_gqa_core_and_metal(
                model_config.text_config.num_hidden_layers,
                &mtp_config.text_config,
                Qwen35MetalDefaults::from_quantization(mtp_config.quantization.as_ref())?,
            )
        })
        .transpose()?;
    let mtp_gqa_page_table_layout = mtp_gqa_geometry.as_ref().map(|(core, metal)| {
        GQAPageTableLayout {
            num_req_slots: config
                .max_requests
                .try_into()
                .expect("qwen3.5 max requests must fit u32"),
            num_blocks: model_config
                .text_config
                .max_position_embeddings
                .div_ceil(config.num_tokens_per_block)
                .max(1)
                .try_into()
                .expect("qwen3.5 MTP GQA block capacity must fit u32"),
            num_gqa_layers: 1,
            num_page_ids_per_block: num_page_ids_per_block(
                config.num_tokens_per_block,
                metal.num_tokens_per_page(core) as usize,
            )
            .try_into()
            .expect("qwen3.5 MTP GQA pages per block must fit u32"),
        }
    });
    let main_gqa_state = Qwen35GQAState::load(
        &device,
        main_gqa_core,
        main_gqa_metal,
        gqa_page_table_layout,
        config.max_tokens,
        config.num_cache_pages,
        0,
    );
    let mtp_gqa_state = mtp_gqa_geometry
        .zip(mtp_gqa_page_table_layout)
        .map(|((core, metal), page_table_layout)| {
            Qwen35GQAState::load(
                &device,
                core,
                metal,
                page_table_layout,
                config.max_tokens,
                config.num_cache_pages,
                1,
            )
        });
    let gdn_layers = (0..model_config.text_config.num_hidden_layers)
        .filter(|&index| {
            model_config
                .layer_type_at(index)
                .is_ok_and(|kind| kind == inference_executor_core::model::qwen::v3_5::LayerType::GDN)
        })
        .collect::<Vec<_>>();
    let gdn_cores = gdn_layers
        .iter()
        .map(|&index| qwen35_gdn_core_and_metal(index, &model_config.text_config, metal_defaults).map(|pair| pair.0))
        .collect::<Result<Vec<_>, _>>()?;
    let (_, gdn_metal) = qwen35_gdn_core_and_metal(gdn_layers[0], &model_config.text_config, metal_defaults)?;
    let max_spec_tokens = config.num_mtp_modules;
    let main_gdn_state = Qwen35GDNState::load(
        &device,
        &gdn_cores,
        gdn_metal,
        config.max_requests,
        max_spec_tokens,
        config.max_tokens,
        config.max_tokens_per_request,
        config.num_tokens_per_block,
        QWEN35_PAGE_SIZE_BYTES,
    );
    let layer_scratch = std::rc::Rc::new(Qwen35LayerScratch::new(
        &device,
        config.max_tokens,
        layout.hidden_dim as usize,
    ));
    let dense_mlp_scratch = if layer_counts.has_dense_mlp
        || mtp_load
            .as_ref()
            .is_some_and(|(mtp_config, ..)| !mtp_config.layer_uses_moe(0))
    {
        let source = (0..model_config.text_config.num_hidden_layers)
            .find(|&index| !model_config.layer_uses_moe(index))
            .map(|index| (&model_config, index))
            .or_else(|| mtp_load.as_ref().map(|(mtp_config, ..)| (mtp_config, 0)))
            .expect("qwen3.5 dense scratch requires a dense layer");
        let (core, metal) = qwen35_dense_mlp_core_and_metal(source.1, &source.0.text_config, metal_defaults)?;
        Some(std::rc::Rc::new(DenseMLPScratch::new(
            &device,
            &core,
            metal,
            config.max_tokens,
        )))
    } else {
        None
    };
    let moe_scratch = if layer_counts.has_moe
        || mtp_load
            .as_ref()
            .is_some_and(|(mtp_config, ..)| mtp_config.layer_uses_moe(0))
    {
        let source = (0..model_config.text_config.num_hidden_layers)
            .find(|&index| model_config.layer_uses_moe(index))
            .map(|index| (&model_config, index))
            .or_else(|| mtp_load.as_ref().map(|(mtp_config, ..)| (mtp_config, 0)))
            .expect("qwen3.5 MoE scratch requires an MoE layer");
        let defaults = Qwen35MetalDefaults::from_quantization(source.0.quantization.as_ref())?;
        let (core, metal) = qwen35_moe_core_and_metal(&format!("layers.{}", source.1), source.1, source.0, defaults)?;
        Some(std::rc::Rc::new(MoEScratch::new(
            &device,
            &core,
            metal,
            config.max_tokens,
        )))
    } else {
        None
    };
    let inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35ModelWeightBindings {
        embed: embed_bindings,
        main: main_bindings,
        unembed: unembed_bindings,
    } = weight_bindings;
    let embed = std::rc::Rc::new(Embed::load(
        &device,
        &mut store,
        layout.embedding_config(),
        embed_bindings,
    )?);
    let token_hidden_input = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    let hidden_output = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    let mtp_hidden_input = mtp_load
        .as_ref()
        .map(|_| Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes())));
    let gather_unembed = Qwen35GatherUnembed::load(&device, &mut store, unembed_config, unembed_bindings)?;
    let main = Qwen35Main::load(
        &device,
        &mut store,
        &model_config,
        metal_defaults,
        main_bindings,
        &main_gqa_state,
        &main_gdn_state,
        std::rc::Rc::clone(&layer_scratch),
        dense_mlp_scratch.as_ref(),
        moe_scratch.as_ref(),
    )?;
    let mtp = if let Some((mtp_model_config, mut mtp_store, mtp_bindings)) = mtp_load.take() {
        let Qwen35MTPWeightBindings {
            embed: mtp_embed_bindings,
            body,
            final_norm_weight,
        } = mtp_bindings;
        let mtp_embed = Qwen35MTPEmbed::load(
            &device,
            &mut mtp_store,
            &mtp_model_config,
            mtp_embed_bindings,
            Rc::clone(&embed),
            config.max_tokens,
        )?;
        let mtp = Qwen35MTP::load(
            &device,
            &mut mtp_store,
            &model_config,
            &mtp_model_config,
            Qwen35MetalDefaults::from_quantization(mtp_model_config.quantization.as_ref())?,
            body,
            final_norm_weight,
            mtp_gqa_state
                .as_ref()
                .expect("qwen3.5 enabled MTP requires a distinct GQA state"),
            Rc::clone(&layer_scratch),
            dense_mlp_scratch.as_ref(),
            moe_scratch.as_ref(),
        )?;
        Some((mtp_embed, mtp))
    } else {
        None
    };
    drop(store);
    let pages = PageArena::new(&device, config.num_cache_pages, QWEN35_PAGE_SIZE_BYTES);
    let target_distribution_indices = Buffer::from_slice(&device, &(0..layout.max_tokens).collect::<Vec<_>>());
    let sampler = Rc::new(TopKSampling::new(&device, sampler_bounds));
    let rejection_sampler = Rc::new(Qwen35RejectionSampler::new(
        &device,
        max_spec_tokens,
        config.max_requests,
        sampler_bounds.top_k,
    ));
    let (mtp_embed, mtp) = match mtp {
        Some((mtp_embed, mtp)) => (Some(mtp_embed), Some(mtp)),
        None => (None, None),
    };
    let model = Qwen35Executor {
        model_name: model_config.model_type,
        default_stop_sequences,
        config,
        runtime,
        layout,
        token_ids: Buffer::new_zeroed(&device, layout.token_id_bytes()),
        token_hidden_input,
        hidden_output,
        mtp_hidden_input,
        mtp_input_gather_flat_indices: Buffer::new_zeroed_elements(&device, config.max_tokens, Dtype::Uint32),
        draft_distribution_indices: Buffer::new_zeroed_elements(&device, config.max_requests, Dtype::Uint32),
        target_distribution_indices,
        mtp_previous_hidden: Buffer::new_zeroed(&device, layout.hidden_bytes()),
        gather_flat_indices: Buffer::new_zeroed_elements(&device, config.max_tokens, Dtype::Uint32),
        unembed_hidden: Buffer::new_zeroed(&device, layout.hidden_bytes()),
        unembed_logits: Buffer::new_zeroed(&device, unembed_config.logits_bytes()),
        main_embed: Replay::new("qwen3.5 MainEmbed", Qwen35MainEmbed::new(Rc::clone(&embed))),
        main: Replay::new("qwen3.5 Main", main),
        gather_unembed: Replay::new("qwen3.5 GatherUnembed", gather_unembed),
        sampling: Replay::new(
            "qwen3.5 sampling",
            Sampling {
                sampler: Rc::clone(&sampler),
            },
        ),
        mtp_embed: mtp_embed.map(|mtp_embed| Replay::new("qwen3.5 MTPEmbed", mtp_embed)),
        mtp: mtp.map(|mtp| Replay::new("qwen3.5 MTP", mtp)),
        draft_sampling: Replay::new(
            "qwen3.5 draft sampling",
            DraftSampling {
                sampler: Rc::clone(&sampler),
            },
        ),
        rejection_sampling: Replay::new(
            "qwen3.5 rejection sampling",
            RejectionSampling::new(Rc::clone(&sampler), rejection_sampler),
        ),
        sampler: Rc::clone(&sampler),
        sampler_bounds,
        sampler_output: TopKSamplingOutputBuffers::new(&device, sampler_bounds),
        request_sampling: RequestSamplingState::new(config.max_requests),
        main_gqa_state,
        main_gdn_state,
        mtp_gqa_state,
        spec_probs: SpecProbsStore::new(
            &device,
            max_spec_tokens,
            config.max_requests,
            sampler_bounds.top_k as usize,
        ),
        pages,
        pending_transactions: Qwen35PendingTransactions::new(),
        gqa_page_table_layout,
    };
    Ok(model)
}
