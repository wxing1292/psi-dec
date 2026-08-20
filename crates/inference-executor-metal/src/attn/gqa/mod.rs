use inference_backend_metal::components::gqa::sdpa as backend_sdpa;

use self::backend::GQAMetalConfig;

pub mod batch_metadata;
pub mod backend;
pub mod request_page_table;
pub mod scratch;
pub mod sdpa;
pub mod ungated_backend;
pub mod ungated_scratch;

fn gqa_sdpa_config(
    config: GQAMetalConfig,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> backend_sdpa::Config {
    let num_q_heads = num_q_heads.try_into().expect("GQA Q-head count must fit u32");
    let num_kv_heads = num_kv_heads.try_into().expect("GQA KV-head count must fit u32");
    let head_dim: u32 = head_dim.try_into().expect("GQA head_dim must fit u32");
    let io_bytes_per_token = u64::from(num_kv_heads)
        .checked_mul(u64::from(head_dim))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(config.io_dtype.item_size() as u64))
        .expect("GQA KV-cache bytes per token must fit u64");
    assert_eq!(
        u64::from(config.page_bytes) % io_bytes_per_token,
        0,
        "GQA page bytes must contain whole KV tokens"
    );
    let tokens_per_page = (u64::from(config.page_bytes) / io_bytes_per_token)
        .try_into()
        .expect("GQA tokens per page must fit u32");
    backend_sdpa::Config {
        io_dtype: config.io_dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        tokens_per_page,
    }
}
