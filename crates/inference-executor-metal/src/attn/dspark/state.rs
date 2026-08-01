use std::rc::Rc;

use inference_backend_metal::components::GQACompute;
use inference_backend_metal::components::GQAComputeConfig;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::DSparkBlockCapacity;
use inference_executor_core::attn::DSparkBlockMetadata;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::attn::UngatedDSparkGQACore;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::dspark::capacity::DSparkGQACapacity;
use crate::attn::dspark::context::DSparkGQAContextScratch;
use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::scratch::DSparkBlockScratch;
use crate::attn::gqa::request_page_table::GQARequestPageTable;

pub struct UngatedDSparkGQAState {
    compute: GQACompute,
    block_scratch: Rc<DSparkBlockScratch>,
    context_scratch: Rc<DSparkGQAContextScratch>,
    request_page_table: Rc<GQARequestPageTable>,
    metadata: DSparkGQAMetadataBuffers,
    num_tokens_per_page: usize,
    num_cache_pages: usize,
    cache_lane: usize,
}

impl UngatedDSparkGQAState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        core: UngatedDSparkGQACore,
        compute_config: GQAComputeConfig,
        page_table_layout: GQAPageTableLayout,
        capacity: DSparkBlockCapacity,
        max_context_tokens: usize,
        num_cache_pages: usize,
        cache_lane: usize,
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
            block_scratch: Rc::new(DSparkBlockScratch::new(
                device,
                &core,
                compute_config.io_dtype,
                gqa_capacity,
            )),
            context_scratch: Rc::new(DSparkGQAContextScratch::new(
                device,
                &core,
                compute_config.io_dtype,
                max_context_tokens,
            )),
            request_page_table: Rc::new(GQARequestPageTable::new(device, page_table_layout)),
            metadata: DSparkGQAMetadataBuffers::new(device, gqa_capacity),
            num_tokens_per_page,
            num_cache_pages,
            cache_lane,
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
        self.metadata.update(block, compute_path)
    }

    pub fn prepare_page_span(
        &self,
        batch: &BatchDeviceRequest,
        num_runtime_page_ids_per_block: usize,
        page_id_offset: usize,
    ) {
        self.request_page_table.prepare_span(
            batch,
            self.cache_lane,
            self.num_cache_pages,
            num_runtime_page_ids_per_block,
            page_id_offset,
        );
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_page_table.reset_req_slots(req_slots);
    }

    pub fn block_scratch(&self) -> Rc<DSparkBlockScratch> {
        Rc::clone(&self.block_scratch)
    }

    pub fn context_scratch(&self) -> Rc<DSparkGQAContextScratch> {
        Rc::clone(&self.context_scratch)
    }

    pub fn request_page_table(&self) -> Rc<GQARequestPageTable> {
        Rc::clone(&self.request_page_table)
    }

    pub fn metadata(&self) -> &DSparkGQAMetadataBuffers {
        &self.metadata
    }
}
