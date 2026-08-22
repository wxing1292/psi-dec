//! Persistent and replay state for block-spec GQA.

use std::rc::Rc;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::BlockSpecCapacity;
use inference_executor_core::attn::BlockSpecGQACore;
use inference_executor_core::attn::BlockSpecMetadata;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::block_spec::backend::BlockSpecGQA;
use crate::attn::block_spec::backend::BlockSpecGQAMetalConfig;
use crate::attn::block_spec::capacity::BlockSpecGQACapacity;
use crate::attn::block_spec::context::BlockSpecGQAContextScratch;
use crate::attn::block_spec::metadata::BlockSpecGQAMetadataBuffers;
use crate::attn::block_spec::scratch::BlockSpecScratch;
use crate::attn::block_spec::sdpa::Selection as SDPASelection;
use crate::attn::block_spec::sdpa::Selector as SDPASelector;
use crate::attn::gqa::request_page_table::GQARequestPageTable;

mod file_io;

pub struct BlockSpecGQAState {
    sdpa_selection: SDPASelection,
    block_scratch: Option<Rc<BlockSpecScratch>>,
    context_scratch: Option<Rc<BlockSpecGQAContextScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
    metadata: Option<BlockSpecGQAMetadataBuffers>,
    core: BlockSpecGQACore,
    sdpa_config: backend_sdpa::Config,
    capacity: BlockSpecGQACapacity,
    max_context_tokens: usize,
    page_table_layout: GQAPageTableLayout,
    num_tokens_per_page: usize,
    num_cache_pages: usize,
}

impl BlockSpecGQAState {
    pub fn new(
        device: &Device,
        core: BlockSpecGQACore,
        sdpa_config: backend_sdpa::Config,
        page_table_layout: GQAPageTableLayout,
        capacity: BlockSpecCapacity,
        max_context_tokens: usize,
        num_cache_pages: usize,
    ) -> Self {
        assert!(max_context_tokens > 0, "block-spec context scratch requires tokens");
        assert!(num_cache_pages > 0, "block-spec GQA state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "block-spec cache page IDs must fit u32"
        );
        core.validate();
        sdpa_config.validate();
        page_table_layout.validate();
        assert_eq!(
            core.block_size, capacity.block_size,
            "block-spec GQA state core and capacity block sizes must match"
        );
        let attention = &core.attention;
        assert_eq!(sdpa_config.num_q_heads as usize, attention.num_q_heads);
        assert_eq!(sdpa_config.num_kv_heads as usize, attention.num_kv_heads);
        assert_eq!(sdpa_config.head_dim as usize, attention.head_dim);
        let sdpa_selection = SDPASelector::new(backend_sdpa::Registry::new(sdpa_config), capacity).select();
        let sdpa_execution = sdpa_selection.execution();
        let gqa_capacity = sdpa_selection.capacity();
        let num_tokens_per_page = sdpa_config.tokens_per_page as usize;
        Self {
            sdpa_selection,
            block_scratch: Some(Rc::new(BlockSpecScratch::new(
                device,
                &core,
                sdpa_config.io_dtype,
                gqa_capacity,
            ))),
            context_scratch: Some(Rc::new(BlockSpecGQAContextScratch::new(
                device,
                &core,
                sdpa_config.io_dtype,
                max_context_tokens,
            ))),
            request_page_table: Some(Rc::new(GQARequestPageTable::new(device, page_table_layout))),
            metadata: Some(BlockSpecGQAMetadataBuffers::new(device, gqa_capacity, sdpa_execution)),
            core,
            sdpa_config,
            capacity: gqa_capacity,
            max_context_tokens,
            page_table_layout,
            num_tokens_per_page,
            num_cache_pages,
        }
    }

    pub fn num_tokens_per_page(&self) -> usize {
        self.num_tokens_per_page
    }

    pub fn prepare_block(&self, block: &BlockSpecMetadata) -> GQAReplayShape {
        self.metadata().update(block)
    }

    pub fn new_gqa(&self, device: &Device, core: BlockSpecGQACore, metal: BlockSpecGQAMetalConfig) -> BlockSpecGQA {
        let shared = &self.core.attention;
        let attention = &core.attention;
        assert_eq!(core.block_size, self.core.block_size);
        assert_eq!(attention.hidden_dim, shared.hidden_dim);
        assert_eq!(attention.head_dim, shared.head_dim);
        assert_eq!(attention.num_q_heads, shared.num_q_heads);
        assert_eq!(attention.num_kv_heads, shared.num_kv_heads);
        assert_eq!(attention.scale, shared.scale);
        assert_eq!(metal.io_dtype, self.sdpa_config.io_dtype);
        BlockSpecGQA::new(device, core, metal, self.sdpa_selection.execution())
    }

    pub fn write_page_ids(&self, req_slot: u32, block_index: usize, page_ids: &[u32]) {
        let request_page_table = self.request_page_table_ref();
        let num_page_ids_per_layer = request_page_table.num_page_ids_per_block();
        let expected_page_ids = request_page_table
            .num_layers()
            .checked_mul(num_page_ids_per_layer)
            .expect("block-spec GQA page-ID count must fit usize");
        assert_eq!(
            page_ids.len(),
            expected_page_ids,
            "block-spec GQA cache block must contain all layer page IDs"
        );
        assert!(
            page_ids
                .iter()
                .all(|&page_id| (page_id as usize) < self.num_cache_pages),
            "runtime supplied a block-spec GQA page ID outside the cache-page buffer"
        );
        for (layer_index, layer_page_ids) in page_ids.chunks_exact(num_page_ids_per_layer).enumerate() {
            request_page_table.write_page_ids(req_slot, layer_index, block_index, layer_page_ids);
        }
    }

    pub fn read_page_ids(&self, req_slot: u32, block_index: usize) -> Vec<u32> {
        let request_page_table = self.request_page_table_ref();
        let mut page_ids = Vec::with_capacity(
            request_page_table
                .num_layers()
                .checked_mul(request_page_table.num_page_ids_per_block())
                .expect("block-spec GQA page-ID count must fit usize"),
        );
        for layer_index in 0..request_page_table.num_layers() {
            page_ids.extend(request_page_table.read_page_ids(req_slot, layer_index, block_index));
        }
        page_ids
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_page_table_ref().reset_req_slots(req_slots);
    }

    pub fn block_scratch(&self) -> Rc<BlockSpecScratch> {
        Rc::clone(
            self.block_scratch
                .as_ref()
                .expect("block-spec GQA block scratch state must be loaded"),
        )
    }

    pub fn context_scratch(&self) -> Rc<BlockSpecGQAContextScratch> {
        Rc::clone(
            self.context_scratch
                .as_ref()
                .expect("block-spec GQA context scratch state must be loaded"),
        )
    }

    pub fn request_page_table(&self) -> Rc<GQARequestPageTable> {
        Rc::clone(self.request_page_table_ref())
    }

    pub fn metadata(&self) -> &BlockSpecGQAMetadataBuffers {
        self.metadata
            .as_ref()
            .expect("block-spec GQA metadata state must be loaded")
    }

    pub fn release_resources(&mut self) {
        assert!(
            self.block_scratch.is_some()
                && self.context_scratch.is_some()
                && self.request_page_table.is_some()
                && self.metadata.is_some(),
            "block-spec GQA state resources are not loaded"
        );
        self.request_page_table
            .take()
            .expect("block-spec GQA request page-table state must be loaded");
        self.metadata.take();
        self.context_scratch.take();
        self.block_scratch.take();
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        assert!(
            self.block_scratch.is_none()
                && self.context_scratch.is_none()
                && self.request_page_table.is_none()
                && self.metadata.is_none(),
            "block-spec GQA state resources are already loaded"
        );
        self.block_scratch = Some(Rc::new(BlockSpecScratch::new(
            device,
            &self.core,
            self.sdpa_config.io_dtype,
            self.capacity,
        )));
        self.context_scratch = Some(Rc::new(BlockSpecGQAContextScratch::new(
            device,
            &self.core,
            self.sdpa_config.io_dtype,
            self.max_context_tokens,
        )));
        self.request_page_table = Some(Rc::new(GQARequestPageTable::new(device, self.page_table_layout)));
        self.metadata = Some(BlockSpecGQAMetadataBuffers::new(
            device,
            self.capacity,
            self.sdpa_selection.execution(),
        ));
    }

    fn request_page_table_ref(&self) -> &Rc<GQARequestPageTable> {
        self.request_page_table
            .as_ref()
            .expect("block-spec GQA request page-table state must be loaded")
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;
    use inference_executor_core::attn::BlockSpecCapacity;
    use inference_executor_core::attn::BlockSpecGQACore;
    use inference_executor_core::attn::GQAPageTableLayout;
    use inference_executor_core::attn::UngatedGQACore;

    use super::BlockSpecGQAState;

    #[test]
    fn test_write_read_page_ids_uses_complete_block_spec_block() {
        let device = Device::system_default();
        let state = new_state(&device);

        state.write_page_ids(1, 1, &[30, 31, 40, 41]);

        assert_eq!(state.read_page_ids(1, 1), vec![30, 31, 40, 41]);
    }

    #[test]
    #[should_panic(expected = "runtime supplied a block-spec GQA page ID outside the cache-page buffer")]
    fn test_write_page_ids_rejects_page_id_outside_cache() {
        let device = Device::system_default();
        let state = new_state(&device);

        state.write_page_ids(1, 1, &[30, 31, 40, 64]);
    }

    fn new_state(device: &Device) -> BlockSpecGQAState {
        BlockSpecGQAState::new(
            device,
            BlockSpecGQACore::new(UngatedGQACore::new(0, 128, 128, 1, 1, 1.0), 1),
            backend_sdpa::Config {
                io_dtype: Dtype::Bfloat16,
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 128,
                tokens_per_page: 8,
            },
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 2,
                num_blocks: 2,
                num_page_ids_per_block: 2,
            },
            BlockSpecCapacity::new(2, 1),
            2,
            64,
        )
    }
}
