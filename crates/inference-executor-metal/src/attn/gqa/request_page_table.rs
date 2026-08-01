use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_runtime_core::compute::BatchDeviceRequest;

#[derive(Debug)]
pub struct GQARequestPageTable {
    layout: GQAPageTableLayout,
    page_ids: Buffer,
}

impl GQARequestPageTable {
    pub fn new(device: &Device, layout: GQAPageTableLayout) -> Self {
        layout.validate();
        Self {
            layout,
            page_ids: Buffer::new_zeroed_elements(
                device,
                layout.num_page_ids(),
                inference_backend_metal::metal::Dtype::Uint32,
            ),
        }
    }

    pub fn num_req_slots(&self) -> usize {
        self.layout.num_req_slots as usize
    }

    pub fn num_layers(&self) -> usize {
        self.layout.num_gqa_layers as usize
    }

    pub fn num_blocks(&self) -> usize {
        self.layout.num_blocks as usize
    }

    pub fn num_page_ids_per_block(&self) -> usize {
        self.layout.num_page_ids_per_block as usize
    }

    pub fn layout(&self) -> GQAPageTableLayout {
        self.layout
    }

    pub fn read_page_ids(&self, req_slot: u32, layer_index: usize, block_index: usize) -> Vec<u32> {
        self.assert_req_slot(req_slot);
        self.assert_layer_index(layer_index);
        self.assert_block_index(block_index);
        let start = self.page_ids_start_index(req_slot, layer_index, block_index);
        self.page_ids.read_typed(start, self.num_page_ids_per_block())
    }

    pub fn page_ids_buffer(&self) -> &Buffer {
        &self.page_ids
    }

    pub fn write_page_ids(&self, req_slot: u32, layer_index: usize, block_index: usize, page_ids: &[u32]) {
        self.assert_req_slot(req_slot);
        self.assert_layer_index(layer_index);
        self.assert_block_index(block_index);
        assert_eq!(
            page_ids.len(),
            self.num_page_ids_per_block(),
            "GQA page-id count must match one request/GQA-layer/block entry"
        );
        let start = self.page_ids_start_index(req_slot, layer_index, block_index);
        self.page_ids.write_typed(start, page_ids);
    }

    pub fn prepare(&self, batch: &BatchDeviceRequest, cache_lane: usize, num_cache_pages: usize) {
        self.prepare_span(
            batch,
            cache_lane,
            num_cache_pages,
            self.num_layers()
                .checked_mul(self.num_page_ids_per_block())
                .expect("GQA runtime page-id count must fit usize"),
            0,
        );
    }

    pub fn prepare_span(
        &self,
        batch: &BatchDeviceRequest,
        cache_lane: usize,
        num_cache_pages: usize,
        num_runtime_page_ids_per_block: usize,
        page_id_offset: usize,
    ) {
        let num_table_page_ids = self
            .num_layers()
            .checked_mul(self.num_page_ids_per_block())
            .expect("GQA page-table span length must fit usize");
        assert!(
            page_id_offset
                .checked_add(num_table_page_ids)
                .is_some_and(|end| end <= num_runtime_page_ids_per_block),
            "GQA page-table span exceeds one runtime cache block"
        );
        for request in &batch.dev_reqs {
            let page_ids_by_lane_and_block = request.decoder_sync_blocks.kv_page_ids();
            let page_ids_by_block = page_ids_by_lane_and_block
                .get(cache_lane)
                .unwrap_or_else(|| panic!("GQA request page table missing cache lane {cache_lane} for kv page ids"));
            for (block_offset, page_ids) in page_ids_by_block.iter().enumerate() {
                assert_eq!(
                    page_ids.len(),
                    num_runtime_page_ids_per_block,
                    "GQA request page table expects {num_runtime_page_ids_per_block} page IDs for each runtime cache \
                     block in lane {cache_lane}, got {}",
                    page_ids.len()
                );
                self.write_runtime_block_span(
                    request.req_slot,
                    request
                        .decoder_sync_blocks
                        .block_index()
                        .checked_add(block_offset)
                        .expect("GQA cache-block index must fit usize"),
                    page_ids,
                    page_id_offset,
                    num_cache_pages,
                );
            }
        }
    }

    pub fn reset_req_slot(&self, req_slot: u32) {
        self.assert_req_slot(req_slot);
        let start = self
            .page_ids_start_index(req_slot, 0, 0)
            .checked_mul(size_of::<u32>())
            .expect("GQA request page-table reset byte offset must fit usize");
        let len = self
            .num_layers()
            .checked_mul(self.num_blocks())
            .and_then(|count| count.checked_mul(self.num_page_ids_per_block()))
            .and_then(|count| count.checked_mul(size_of::<u32>()))
            .expect("GQA request page-table reset byte length must fit usize");
        self.page_ids.zero_bytes(start, len);
    }

    pub fn reset_req_slots(&self, req_slots: &[u32]) {
        for &req_slot in req_slots {
            self.reset_req_slot(req_slot);
        }
    }

    fn assert_req_slot(&self, req_slot: u32) {
        if req_slot as usize >= self.num_req_slots() {
            panic!(
                "GQA request page table req_slot out of range: req_slot={req_slot} num_req_slots={}",
                self.num_req_slots()
            );
        }
    }

    fn assert_layer_index(&self, layer_index: usize) {
        if layer_index >= self.num_layers() {
            panic!(
                "GQA request page table layer out of range: layer_index={layer_index} num_layers={}",
                self.num_layers()
            );
        }
    }

    fn assert_block_index(&self, block_index: usize) {
        if block_index >= self.num_blocks() {
            panic!(
                "GQA request page table block out of range: block_index={block_index} num_blocks={}",
                self.num_blocks()
            );
        }
    }

    fn page_ids_start_index(&self, req_slot: u32, layer_index: usize, block_index: usize) -> usize {
        usize::try_from(req_slot)
            .expect("GQA request slot must fit host usize")
            .checked_mul(self.num_layers())
            .and_then(|index| index.checked_add(layer_index))
            .and_then(|index| index.checked_mul(self.num_blocks()))
            .and_then(|index| index.checked_add(block_index))
            .and_then(|index| index.checked_mul(self.num_page_ids_per_block()))
            .expect("GQA request page-table flat index must fit usize")
    }

    fn write_runtime_block_span(
        &self,
        req_slot: u32,
        block_index: usize,
        runtime_page_ids: &[u32],
        page_id_offset: usize,
        num_cache_pages: usize,
    ) {
        let num_table_page_ids = self
            .num_layers()
            .checked_mul(self.num_page_ids_per_block())
            .expect("GQA page-table span length must fit usize");
        let span_end = page_id_offset
            .checked_add(num_table_page_ids)
            .expect("GQA page-table span end must fit usize");
        let page_ids = runtime_page_ids
            .get(page_id_offset..span_end)
            .expect("GQA page-table span must fit one runtime cache block");
        assert!(
            page_ids.iter().all(|&page_id| (page_id as usize) < num_cache_pages),
            "runtime supplied a GQA page ID outside the cache-page buffer"
        );
        for (layer_index, layer_page_ids) in page_ids.chunks_exact(self.num_page_ids_per_block()).enumerate() {
            self.write_page_ids(req_slot, layer_index, block_index, layer_page_ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;
    use inference_executor_core::attn::GQAPageTableLayout;

    use super::GQARequestPageTable;

    #[test]
    fn test_read_write() {
        let device = Device::system_default();
        let page_table = GQARequestPageTable::new(
            &device,
            GQAPageTableLayout {
                num_req_slots: 4,
                num_gqa_layers: 3,
                num_blocks: 6,
                num_page_ids_per_block: 2,
            },
        );

        page_table.write_page_ids(1, 0, 0, &[10, 11]);
        page_table.write_page_ids(1, 0, 1, &[12, 13]);
        page_table.write_page_ids(1, 2, 0, &[20, 21]);
        page_table.write_page_ids(2, 1, 0, &[30, 31]);
        page_table.write_page_ids(3, 0, 0, &[40, 41]);

        assert_eq!(page_table.num_req_slots(), 4);
        assert_eq!(page_table.num_layers(), 3);
        assert_eq!(page_table.num_blocks(), 6);
        assert_eq!(page_table.num_page_ids_per_block(), 2);
        assert_eq!(page_table.read_page_ids(1, 0, 0), vec![10, 11]);
        assert_eq!(page_table.read_page_ids(1, 0, 1), vec![12, 13]);
        assert_eq!(page_table.read_page_ids(1, 2, 0), vec![20, 21]);
    }

    #[test]
    fn test_reset() {
        let device = Device::system_default();
        let page_table = GQARequestPageTable::new(
            &device,
            GQAPageTableLayout {
                num_req_slots: 4,
                num_gqa_layers: 3,
                num_blocks: 6,
                num_page_ids_per_block: 2,
            },
        );

        page_table.write_page_ids(0, 1, 0, &[100, 101]);
        page_table.write_page_ids(2, 0, 0, &[200, 201]);
        page_table.write_page_ids(2, 2, 0, &[220, 221]);
        page_table.write_page_ids(3, 1, 0, &[300, 301]);

        page_table.reset_req_slot(2);

        assert_eq!(page_table.read_page_ids(0, 1, 0), vec![100, 101]);
        assert_eq!(page_table.read_page_ids(2, 0, 0), vec![0, 0]);
        assert_eq!(page_table.read_page_ids(2, 2, 0), vec![0, 0]);
        assert_eq!(page_table.read_page_ids(3, 1, 0), vec![300, 301]);
    }

    #[test]
    fn test_write_runtime_block_span() {
        let device = Device::system_default();
        let main = GQARequestPageTable::new(
            &device,
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 2,
                num_blocks: 3,
                num_page_ids_per_block: 2,
            },
        );
        let dspark = GQARequestPageTable::new(
            &device,
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 1,
                num_blocks: 3,
                num_page_ids_per_block: 3,
            },
        );
        let runtime_page_ids = [10, 11, 20, 21, 30, 31, 32];

        main.write_runtime_block_span(1, 2, &runtime_page_ids, 0, 64);
        dspark.write_runtime_block_span(1, 2, &runtime_page_ids, 4, 64);

        assert_eq!(main.read_page_ids(1, 0, 2), vec![10, 11]);
        assert_eq!(main.read_page_ids(1, 1, 2), vec![20, 21]);
        assert_eq!(dspark.read_page_ids(1, 0, 2), vec![30, 31, 32]);

        main.reset_req_slots(&[1]);
        dspark.reset_req_slots(&[1]);

        assert_eq!(main.read_page_ids(1, 0, 2), vec![0, 0]);
        assert_eq!(main.read_page_ids(1, 1, 2), vec![0, 0]);
        assert_eq!(dspark.read_page_ids(1, 0, 2), vec![0, 0, 0]);
    }
}
