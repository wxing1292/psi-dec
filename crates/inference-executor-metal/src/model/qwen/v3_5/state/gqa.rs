use std::rc::Rc;

use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gqa::backend::GQA;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::gqa::scratch::GQAScratch;

pub struct Qwen35GQAState {
    backend: Rc<GQA>,
    scratch: Rc<GQAScratch>,
    request_page_table: Rc<GQARequestPageTable>,
    metadata: GQAMetadataBuffers,
    num_cache_pages: usize,
    cache_lane: usize,
}

impl Qwen35GQAState {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        core: GQACore,
        metal: GQAMetalConfig,
        page_table_layout: GQAPageTableLayout,
        max_tokens: usize,
        num_cache_pages: usize,
        cache_lane: usize,
    ) -> Self {
        assert!(num_cache_pages > 0, "qwen3.5 GQA state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "qwen3.5 cache page IDs must fit u32"
        );
        page_table_layout.validate();
        Self {
            backend: Rc::new(GQA::new(device, core.clone(), metal)),
            scratch: Rc::new(GQAScratch::new(device, &core, metal, max_tokens)),
            request_page_table: Rc::new(GQARequestPageTable::new(device, page_table_layout)),
            metadata: GQAMetadataBuffers::new(device, max_tokens),
            num_cache_pages,
            cache_lane,
        }
    }

    pub fn backend(&self) -> &Rc<GQA> {
        &self.backend
    }

    pub fn scratch(&self) -> &Rc<GQAScratch> {
        &self.scratch
    }

    pub fn request_page_table(&self) -> &Rc<GQARequestPageTable> {
        &self.request_page_table
    }

    pub fn metadata(&self) -> &GQAMetadataBuffers {
        &self.metadata
    }

    pub fn prepare_pages(&self, core_batch: &BatchDeviceRequest) {
        self.request_page_table
            .prepare(core_batch, self.cache_lane, self.num_cache_pages);
    }

    pub fn prepare_metadata(&self, microbatch: &Qwen35Microbatch) -> GQAReplayShape {
        self.backend.prepare(
            &self.metadata,
            microbatch.req_slots(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
        )
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_page_table.reset_req_slots(req_slots);
    }
}
