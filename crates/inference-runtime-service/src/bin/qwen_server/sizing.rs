pub const QWEN35_DEFAULT_NUM_CACHE_PAGES: usize = 384 * 1024;

pub fn kv_dtype_bytes(dtype: Option<&str>) -> usize {
    match dtype {
        Some("float32") => 4,
        Some("float16") | Some("half") | Some("bfloat16") | Some("bf16") | None => 2,
        _ => unimplemented!(),
    }
}

pub fn block_cache_capacity(
    num_pages: usize,
    num_kv_pages_per_block: usize,
    num_state_pages_per_block: usize,
) -> usize {
    let num_pages = u64::try_from(num_pages).expect("cache physical page count must fit u64");
    let num_pages_per_block = u64::try_from(num_kv_pages_per_block)
        .expect("cache KV pages per block must fit u64")
        .checked_add(u64::try_from(num_state_pages_per_block).expect("cache state pages per block must fit u64"))
        .expect("cache block physical page count overflow");
    assert!(
        num_pages_per_block > 0,
        "cache block must consume at least one physical page"
    );
    assert!(
        num_pages >= num_pages_per_block,
        "num_pages={num_pages} is too small for one cache block requiring {num_pages_per_block} pages"
    );
    usize::try_from(num_pages / num_pages_per_block).expect("cache block capacity must fit usize")
}

#[cfg(test)]
mod tests {
    use super::block_cache_capacity;

    #[test]
    #[should_panic(expected = "cache block physical page count overflow")]
    fn test_block_capacity_rejects_page_count_overflow() {
        block_cache_capacity(usize::MAX, usize::MAX, 1);
    }

    #[test]
    fn test_block_capacity_counts_complete_blocks() {
        assert_eq!(block_cache_capacity(25, 7, 5), 2);
    }
}
