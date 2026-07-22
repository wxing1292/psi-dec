mod batch;
pub use batch::Qwen35DecodeDecision;
pub use batch::Qwen35Microbatch;
pub use batch::Qwen35ModelBatchRequest;
pub use batch::Qwen35SampledTokens;
pub use batch::gather_flat_indices;
pub use batch::has_synced_pages;
pub use batch::num_target_hidden_states;
pub use batch::sample_decisions_from_sampled_tokens;
pub use batch::sample_sampler_configs;
pub use batch::sample_token_positions;
pub use batch::to_core_batch_resp;
pub use batch::verified_state_versions;
pub use batch::verified_state_versions_for_decisions;

mod config;
pub use config::LayerType;
pub use config::QuantizationConfig;
pub use config::Qwen35ModelConfig;
pub use config::ResolvedQuantizationConfig;
pub use config::RopeParameters;
pub use config::TensorPathLayout;
pub use config::TensorQuantizationOverride;
pub use config::TextConfig;
pub use config::default_tensor_path_layout;
pub use config::init_model_config;
pub use config::layer_type_at;
pub use config::layer_uses_moe;
pub use config::normalize_qwen_name;
pub use config::normalize_text_config;
pub use config::parse_nested_quantization_config;
pub use config::resolve_tensor_path_layout_from_names;
pub use config::tensor_path_layout_candidates;

pub mod weight_layout;

mod dspark_config;
pub use dspark_config::DSparkConfig;
pub use dspark_config::DSparkDFlashConfig;
pub use dspark_config::init_dspark_config;

pub mod dspark_weight_layout;

mod pending_transactions;
pub use pending_transactions::Qwen35PendingTransactions;

pub const QWEN35_PAGE_SIZE_BYTES: usize = 32 * 1024;
