use inference_runtime_core::Result;
use inference_runtime_core::log_err_internal;
use inference_runtime_core::log_info_invalid_argument;

pub fn context_window(model_context_window: usize, block_spec_num_spec_tokens: usize) -> Result<usize> {
    if model_context_window <= block_spec_num_spec_tokens {
        return Err(log_info_invalid_argument!(
            "model context window={model_context_window} must exceed block-spec proposal token \
             count={block_spec_num_spec_tokens}"
        ));
    }
    Ok(model_context_window - block_spec_num_spec_tokens)
}

pub fn block_cache_capacity(
    num_pages: usize,
    num_kv_pages_per_block: usize,
    num_state_pages_per_block: usize,
) -> Result<usize> {
    let num_pages =
        u64::try_from(num_pages).map_err(|_| log_err_internal!("cache physical page count must fit u64"))?;
    let num_pages_per_block = u64::try_from(num_kv_pages_per_block)
        .map_err(|_| log_err_internal!("cache KV pages per block must fit u64"))?
        .checked_add(
            u64::try_from(num_state_pages_per_block)
                .map_err(|_| log_err_internal!("cache state pages per block must fit u64"))?,
        )
        .ok_or_else(|| log_err_internal!("cache block physical page count overflow"))?;
    if num_pages_per_block == 0 {
        return Err(log_err_internal!("cache block must consume at least one physical page"));
    }
    if num_pages < num_pages_per_block {
        return Err(log_info_invalid_argument!(
            "--num-cache-pages={num_pages} is too small for one cache block requiring {num_pages_per_block} pages"
        ));
    }
    usize::try_from(num_pages / num_pages_per_block)
        .map_err(|_| log_err_internal!("cache block capacity must fit usize"))
}

#[cfg(test)]
mod tests {
    use inference_runtime_core::Error;

    use super::block_cache_capacity;
    #[test]
    fn test_block_capacity_rejects_incomplete_block() {
        assert!(matches!(
            block_cache_capacity(11, 7, 5),
            Err(Error::InvalidArgument(message)) if message.contains("requiring 12 pages")
        ));
    }

    #[test]
    fn test_block_capacity_counts_complete_blocks() {
        assert_eq!(block_cache_capacity(25, 7, 5).unwrap(), 2);
    }
}
