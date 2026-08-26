use std::path::Path;
use std::rc::Rc;

use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::BiDiBlockCapacity;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkWeightBindings;
use inference_executor_core::model::qwen::v3_x::dspark::resolve_qwen3x_dspark_weight_bindings;
use inference_executor_core::sampling::TopKSamplingBounds;

use crate::attn::bidi_block_gqa::state::BiDiBlockGQAState;
use crate::checkpoint::SafeTensorStore;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedConfig;
use crate::model::qwen::v3_x::dspark::attention::qwen3x_dspark_gqa_core;
use crate::model::qwen::v3_x::dspark::attention::qwen3x_dspark_gqa_sdpa_config;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkModel;
use crate::model::qwen::v3_x::dspark::sampling::Qwen3xDSparkMarkov;
use crate::model::qwen::v3_x::weight::to_u32;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3xDSparkLoadConfig {
    pub page_size_bytes: usize,
    pub max_position_embeddings: usize,
    pub max_requests: usize,
    pub max_tokens: usize,
    pub num_cache_pages: usize,
    pub num_tokens_per_block: usize,
}

pub struct Qwen3xDSparkLoaded {
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_state: BiDiBlockGQAState,
    pub model: Rc<Qwen3xDSparkModel>,
    pub embed: Rc<Embed>,
    pub unembed: Rc<Unembed>,
    pub markov: Qwen3xDSparkMarkov,
    pub num_spec_tokens: usize,
    pub mask_token_id: i32,
    pub page_bytes: usize,
    pub max_main_tokens: usize,
    pub embed_uses_main: bool,
    pub unembed_uses_main: bool,
}

pub fn load_qwen3x_dspark(
    device: &Device,
    model_dir: &Path,
    config: &Qwen3xDSparkConfig,
    load_config: Qwen3xDSparkLoadConfig,
    main_embed: Rc<Embed>,
    main_unembed: Rc<Unembed>,
    sampler_bounds: TopKSamplingBounds,
) -> Result<Qwen3xDSparkLoaded, ModelExecutorError> {
    let num_spec_tokens = config.num_spec_tokens().get();
    let mut store = SafeTensorStore::from_model_dir(model_dir)?;
    let Qwen3xDSparkWeightBindings {
        embed: embed_bindings,
        main_feature: main_feature_bindings,
        layers: layer_bindings,
        final_norm_weight,
        unembed: unembed_bindings,
        markov: markov_bindings,
        confidence: confidence_bindings,
    } = resolve_qwen3x_dspark_weight_bindings(config, store.index().tensor_names())?;
    let attention_core = qwen3x_dspark_gqa_core(config, num_spec_tokens, 0);
    let attention_split_kv_config = qwen3x_dspark_gqa_sdpa_config(config, load_config.page_size_bytes)?;
    let tokens_per_page = attention_split_kv_config.tokens_per_page as usize;
    let page_ids_per_block = num_page_ids_per_block(load_config.num_tokens_per_block, tokens_per_page);
    let page_table_layout = GQAPageTableLayout {
        num_req_slots: load_config
            .max_requests
            .try_into()
            .expect("Qwen3x DSpark max_requests must fit u32"),
        num_blocks: load_config
            .max_position_embeddings
            .div_ceil(load_config.num_tokens_per_block)
            .max(1)
            .try_into()
            .expect("Qwen3x DSpark block capacity must fit u32"),
        num_gqa_layers: config
            .num_hidden_layers
            .try_into()
            .expect("Qwen3x DSpark layer count must fit u32"),
        num_page_ids_per_block: page_ids_per_block
            .try_into()
            .expect("Qwen3x DSpark pages per block must fit u32"),
    };
    let capacity = BiDiBlockCapacity::new(load_config.max_requests, num_spec_tokens);
    let gqa_state = BiDiBlockGQAState::new(
        device,
        attention_core,
        attention_split_kv_config,
        page_table_layout,
        capacity,
        load_config.max_tokens,
        load_config.num_cache_pages,
    );
    let quantization = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DSpark Metal executor requires quantization config"))?;
    let max_block_tokens = capacity.max_tokens;
    let embed_uses_main = embed_bindings.is_none();
    let embed = if let Some(embed_bindings) = embed_bindings {
        let resolved = quantization.resolve_for_tensor(&embed_bindings.weight);
        let embed_config = EmbedConfig {
            max_tokens: max_block_tokens
                .try_into()
                .expect("Qwen3x DSpark embed rows must fit u32"),
            vocab_size: config
                .vocab_size
                .try_into()
                .expect("Qwen3x DSpark embedding vocabulary must fit u32"),
            hidden_dim: config
                .hidden_size
                .try_into()
                .expect("Qwen3x DSpark embedding width must fit u32"),
            group_size: to_u32("Qwen3x DSpark embedding group_size", resolved.group_size)?,
            bits: to_u32("Qwen3x DSpark embedding bits", resolved.bits)?,
            scale_bias_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        };
        let mut embed = Embed::new(device, embed_config);
        embed.load_weights(device, &mut store, embed_bindings)?;
        Rc::new(embed)
    } else {
        Rc::new(
            main_embed.with_max_tokens(
                max_block_tokens
                    .try_into()
                    .expect("Qwen3x DSpark shared embed rows must fit u32"),
            ),
        )
    };
    let unembed_uses_main = unembed_bindings.is_none();
    let unembed = if let Some(unembed_bindings) = unembed_bindings {
        let resolved = quantization.resolve_for_tensor(&unembed_bindings.weight);
        let unembed_config = UnembedConfig {
            max_tokens: max_block_tokens
                .try_into()
                .expect("Qwen3x DSpark unembed rows must fit u32"),
            vocab_size: config
                .vocab_size
                .try_into()
                .expect("Qwen3x DSpark unembed vocabulary must fit u32"),
            hidden_dim: config
                .hidden_size
                .try_into()
                .expect("Qwen3x DSpark unembed hidden width must fit u32"),
            group_size: to_u32("Qwen3x DSpark unembed group_size", resolved.group_size)?,
            bits: to_u32("Qwen3x DSpark unembed bits", resolved.bits)?,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let mut unembed = Unembed::new(device, unembed_config);
        unembed.load_weights(device, &mut store, unembed_bindings)?;
        Rc::new(unembed)
    } else {
        Rc::new(
            main_unembed.with_max_tokens(
                max_block_tokens
                    .try_into()
                    .expect("Qwen3x DSpark shared unembed rows must fit u32"),
            ),
        )
    };
    let mut markov = Qwen3xDSparkMarkov::new(
        device,
        config,
        num_spec_tokens,
        &markov_bindings,
        load_config.max_requests,
        sampler_bounds,
    )?;
    markov.load_weights(device, &mut store, &markov_bindings, &confidence_bindings)?;
    let mut model = Qwen3xDSparkModel::new(
        device,
        config,
        num_spec_tokens,
        load_config.page_size_bytes,
        &main_feature_bindings,
        &layer_bindings,
        &gqa_state,
        load_config.max_tokens,
        max_block_tokens,
    )?;
    model.load_weights(
        device,
        &mut store,
        config,
        &main_feature_bindings,
        layer_bindings,
        final_norm_weight,
    )?;
    let model = Rc::new(model);
    Ok(Qwen3xDSparkLoaded {
        page_table_layout,
        gqa_state,
        model,
        embed,
        unembed,
        markov,
        num_spec_tokens,
        mask_token_id: config
            .mask_token_id
            .try_into()
            .map_err(|_| ModelExecutorError::custom("Qwen3x DSpark MASK token ID must fit i32"))?,
        page_bytes: load_config.page_size_bytes,
        max_main_tokens: load_config.max_tokens,
        embed_uses_main,
        unembed_uses_main,
    })
}

fn num_page_ids_per_block(num_tokens_per_block: usize, num_tokens_per_page: usize) -> usize {
    assert!(
        num_tokens_per_block > 0,
        "Qwen3x DSpark GQA requires positive tokens per block"
    );
    assert!(
        num_tokens_per_page > 0,
        "Qwen3x DSpark GQA requires positive tokens per page"
    );
    assert!(
        num_tokens_per_block.is_multiple_of(num_tokens_per_page),
        "Qwen3x DSpark GQA tokens per block must be divisible by tokens per page"
    );
    num_tokens_per_block / num_tokens_per_page
}
