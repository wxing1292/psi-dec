use inference_backend_metal::components::GQASplitKVConfig;

use self::backend::GQAMetalConfig;

pub mod batch_metadata;
pub mod backend;
pub mod request_page_table;
pub mod scratch;
pub mod ungated_backend;
pub mod ungated_scratch;

fn gqa_split_kv_config(
    config: GQAMetalConfig,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> GQASplitKVConfig {
    GQASplitKVConfig {
        io_dtype: config.io_dtype,
        page_bytes: config.page_bytes,
        num_q_heads: num_q_heads.try_into().expect("GQA Q-head count must fit u32"),
        num_kv_heads: num_kv_heads.try_into().expect("GQA KV-head count must fit u32"),
        head_dim: head_dim.try_into().expect("GQA head_dim must fit u32"),
    }
}
