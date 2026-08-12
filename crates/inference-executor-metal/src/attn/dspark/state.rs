use std::rc::Rc;

use inference_backend_metal::components::GQACompute;
use inference_backend_metal::components::GQAComputeConfig;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::DSparkBlockCapacity;
use inference_executor_core::attn::DSparkBlockMetadata;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::attn::UngatedDSparkGQACore;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::dspark::capacity::DSparkGQACapacity;
use crate::attn::dspark::context::DSparkGQAContextScratch;
use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::scratch::DSparkBlockScratch;
use crate::attn::gqa::request_page_table::GQARequestPageTable;

mod file_io;

pub struct UngatedDSparkGQAState {
    compute: GQACompute,
    block_scratch: Option<Rc<DSparkBlockScratch>>,
    context_scratch: Option<Rc<DSparkGQAContextScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
    metadata: Option<DSparkGQAMetadataBuffers>,
    core: UngatedDSparkGQACore,
    compute_config: GQAComputeConfig,
    capacity: DSparkGQACapacity,
    max_context_tokens: usize,
    page_table_layout: GQAPageTableLayout,
    num_tokens_per_page: usize,
    num_cache_pages: usize,
}

impl UngatedDSparkGQAState {
    pub fn new(
        device: &Device,
        core: UngatedDSparkGQACore,
        compute_config: GQAComputeConfig,
        page_table_layout: GQAPageTableLayout,
        capacity: DSparkBlockCapacity,
        max_context_tokens: usize,
        num_cache_pages: usize,
    ) -> Self {
        assert!(max_context_tokens > 0, "DSpark context scratch requires tokens");
        assert!(num_cache_pages > 0, "DSpark GQA state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "DSpark cache page IDs must fit u32"
        );
        core.validate();
        compute_config.validate();
        page_table_layout.validate();
        assert_eq!(
            core.block_size, capacity.block_size,
            "DSpark GQA state core and capacity block sizes must match"
        );
        let gqa_capacity = DSparkGQACapacity::new(capacity);
        let attention = &core.attention;
        assert_eq!(compute_config.num_q_heads as usize, attention.num_q_heads);
        assert_eq!(compute_config.num_kv_heads as usize, attention.num_kv_heads);
        assert_eq!(compute_config.head_dim as usize, attention.head_dim);
        let compute = GQACompute::new_dspark_history(compute_config);
        let num_tokens_per_page = compute_config.num_tokens_per_page() as usize;
        Self {
            compute,
            block_scratch: Some(Rc::new(DSparkBlockScratch::new(
                device,
                &core,
                compute_config.io_dtype,
                gqa_capacity,
            ))),
            context_scratch: Some(Rc::new(DSparkGQAContextScratch::new(
                device,
                &core,
                compute_config.io_dtype,
                max_context_tokens,
            ))),
            request_page_table: Some(Rc::new(GQARequestPageTable::new(device, page_table_layout))),
            metadata: Some(DSparkGQAMetadataBuffers::new(device, gqa_capacity)),
            core,
            compute_config,
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

    pub fn prepare_block(&self, block: &DSparkBlockMetadata) -> GQAReplayShape {
        let num_tokens = block
            .num_tokens()
            .try_into()
            .expect("DSpark GQA token count must fit u32");
        let compute_path = self.compute.select(num_tokens, num_tokens);
        self.metadata().update(block, compute_path)
    }

    pub fn write_page_ids(&self, req_slot: u32, block_index: usize, page_ids: &[u32]) {
        let request_page_table = self.request_page_table_ref();
        let num_page_ids_per_layer = request_page_table.num_page_ids_per_block();
        let expected_page_ids = request_page_table
            .num_layers()
            .checked_mul(num_page_ids_per_layer)
            .expect("DSpark GQA page-ID count must fit usize");
        assert_eq!(
            page_ids.len(),
            expected_page_ids,
            "DSpark GQA cache block must contain all layer page IDs"
        );
        assert!(
            page_ids
                .iter()
                .all(|&page_id| (page_id as usize) < self.num_cache_pages),
            "runtime supplied a DSpark GQA page ID outside the cache-page buffer"
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
                .expect("DSpark GQA page-ID count must fit usize"),
        );
        for layer_index in 0..request_page_table.num_layers() {
            page_ids.extend(request_page_table.read_page_ids(req_slot, layer_index, block_index));
        }
        page_ids
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_page_table_ref().reset_req_slots(req_slots);
    }

    pub fn block_scratch(&self) -> Rc<DSparkBlockScratch> {
        Rc::clone(
            self.block_scratch
                .as_ref()
                .expect("DSpark GQA block scratch state must be loaded"),
        )
    }

    pub fn context_scratch(&self) -> Rc<DSparkGQAContextScratch> {
        Rc::clone(
            self.context_scratch
                .as_ref()
                .expect("DSpark GQA context scratch state must be loaded"),
        )
    }

    pub fn request_page_table(&self) -> Rc<GQARequestPageTable> {
        Rc::clone(self.request_page_table_ref())
    }

    pub fn metadata(&self) -> &DSparkGQAMetadataBuffers {
        self.metadata
            .as_ref()
            .expect("DSpark GQA metadata state must be loaded")
    }

    pub fn release_resources(&mut self) {
        assert!(
            self.block_scratch.is_some()
                && self.context_scratch.is_some()
                && self.request_page_table.is_some()
                && self.metadata.is_some(),
            "DSpark GQA state resources are not loaded"
        );
        self.request_page_table
            .take()
            .expect("DSpark GQA request page-table state must be loaded");
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
            "DSpark GQA state resources are already loaded"
        );
        self.block_scratch = Some(Rc::new(DSparkBlockScratch::new(
            device,
            &self.core,
            self.compute_config.io_dtype,
            self.capacity,
        )));
        self.context_scratch = Some(Rc::new(DSparkGQAContextScratch::new(
            device,
            &self.core,
            self.compute_config.io_dtype,
            self.max_context_tokens,
        )));
        self.request_page_table = Some(Rc::new(GQARequestPageTable::new(device, self.page_table_layout)));
        self.metadata = Some(DSparkGQAMetadataBuffers::new(device, self.capacity));
    }

    fn request_page_table_ref(&self) -> &Rc<GQARequestPageTable> {
        self.request_page_table
            .as_ref()
            .expect("DSpark GQA request page-table state must be loaded")
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::GQAComputeConfig;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;
    use inference_executor_core::attn::DSparkBlockCapacity;
    use inference_executor_core::attn::GQAPageTableLayout;
    use inference_executor_core::attn::UngatedDSparkGQACore;
    use inference_executor_core::attn::UngatedGQACore;

    use super::UngatedDSparkGQAState;

    #[test]
    fn test_write_read_page_ids_uses_complete_dspark_block() {
        let device = Device::system_default();
        let state = new_state(&device);

        state.write_page_ids(1, 1, &[30, 31, 40, 41]);

        assert_eq!(state.read_page_ids(1, 1), vec![30, 31, 40, 41]);
    }

    #[test]
    #[should_panic(expected = "runtime supplied a DSpark GQA page ID outside the cache-page buffer")]
    fn test_write_page_ids_rejects_page_id_outside_cache() {
        let device = Device::system_default();
        let state = new_state(&device);

        state.write_page_ids(1, 1, &[30, 31, 40, 64]);
    }

    fn new_state(device: &Device) -> UngatedDSparkGQAState {
        UngatedDSparkGQAState::new(
            device,
            UngatedDSparkGQACore::new(UngatedGQACore::new(0, 128, 128, 1, 1, 1.0), 1),
            GQAComputeConfig {
                io_dtype: Dtype::Bfloat16,
                page_bytes: 4096,
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 128,
            },
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 2,
                num_blocks: 2,
                num_page_ids_per_block: 2,
            },
            DSparkBlockCapacity::new(2, 1),
            2,
            64,
        )
    }
}
