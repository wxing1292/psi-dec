use crate::def::DenseLinearShape;

#[derive(Clone, Debug, PartialEq)]
pub struct GQACore {
    pub model_layer_index: usize,
    pub hidden_dim: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub scale: f32,
}

impl GQACore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_layer_index: usize,
        hidden_dim: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        scale: f32,
    ) -> Self {
        Self {
            model_layer_index,
            hidden_dim,
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
        }
    }

    pub fn validate(&self) {
        assert!(self.hidden_dim > 0);
        assert!(self.head_dim > 0);
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.scale > 0.0);
        let _ = self.qgkv_dim();
    }

    pub fn q_dim(&self) -> usize {
        self.num_q_heads
            .checked_mul(self.head_dim)
            .expect("GQA query dimension must fit usize")
    }

    pub fn g_dim(&self) -> usize {
        self.q_dim()
    }

    pub fn k_dim(&self) -> usize {
        self.num_kv_heads
            .checked_mul(self.head_dim)
            .expect("GQA key dimension must fit usize")
    }

    pub fn v_dim(&self) -> usize {
        self.k_dim()
    }

    pub fn qg_dim(&self) -> usize {
        self.q_dim()
            .checked_add(self.g_dim())
            .expect("GQA query/gate dimension must fit usize")
    }

    pub fn qgkv_dim(&self) -> usize {
        self.qg_dim()
            .checked_add(self.k_dim())
            .and_then(|dim| dim.checked_add(self.v_dim()))
            .expect("GQA fused projection dimension must fit usize")
    }

    pub fn qgkv_shape(&self) -> DenseLinearShape {
        self.validate();
        DenseLinearShape {
            out_dim: self.qgkv_dim(),
            in_dim: self.hidden_dim,
        }
    }

    pub fn output_shape(&self) -> DenseLinearShape {
        self.validate();
        DenseLinearShape {
            out_dim: self.hidden_dim,
            in_dim: self.q_dim(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQAReplayShape {
    /// Current flattened Q-token count (`Tq`).
    pub num_tokens: u32,
    /// Current request-local Q-token-tile count along `Tq`.
    pub num_q_token_tiles: u32,
    /// Replay-padded total `SDPAMapTaskTemplate` extent. The valid
    /// TaskTemplate count may be smaller; sentinel tail TaskTemplates do no work.
    pub total_sdpa_map_task_templates: u32,
    /// Whether the unpadded batch plan semantically has partial outputs to merge.
    pub reduce_sdpa_partial_outputs: bool,
}

impl GQAReplayShape {
    pub fn validate(self) {
        assert!(self.num_tokens > 0);
        assert!(self.num_q_token_tiles > 0 && self.num_q_token_tiles <= self.num_tokens);
        assert!(self.total_sdpa_map_task_templates > 0);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQAPageTableLayout {
    /// Init-time page-table storage layout. Treat the memory as a tensor with
    /// shape `[num_req_slots, num_gqa_layers, num_blocks, num_page_ids_per_block]`.
    ///
    /// One physical page stores both K and V:
    /// `[K/V, kv_head, page_token_index, head_dim]`.
    pub num_req_slots: u32,
    /// Number of GQA page-table layer slots.
    pub num_gqa_layers: u32,
    /// Logical cache-block capacity per request slot and GQA layer.
    pub num_blocks: u32,
    /// Physical page IDs assigned to one request/GQA-layer/cache-block tuple.
    pub num_page_ids_per_block: u32,
}

impl GQAPageTableLayout {
    pub fn validate(self) {
        assert!(self.num_req_slots > 0);
        assert!(self.num_blocks > 0);
        assert!(self.num_gqa_layers > 0);
        assert!(self.num_page_ids_per_block > 0);
    }

    pub fn num_page_ids(self) -> usize {
        (self.num_req_slots as usize)
            .checked_mul(self.num_gqa_layers as usize)
            .and_then(|count| count.checked_mul(self.num_blocks as usize))
            .and_then(|count| count.checked_mul(self.num_page_ids_per_block as usize))
            .expect("GQA page-table element count must fit usize")
    }

    /// Number of physical GQA pages addressable by one request/GQA-layer table.
    pub fn num_physical_pages_per_request(self) -> usize {
        (self.num_blocks as usize)
            .checked_mul(self.num_page_ids_per_block as usize)
            .expect("GQA per-request physical page count must fit usize")
    }
}

#[cfg(test)]
mod tests {
    use super::GQACore;

    #[test]
    #[should_panic(expected = "GQA query dimension must fit usize")]
    fn test_dimension_overflow_panics() {
        let core = GQACore::new(0, 1, 2, usize::MAX, 1, 1.0);
        core.validate();
    }
}
