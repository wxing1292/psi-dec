use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::model::qwen::v3_5::has_synced_pages;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::page_ids_by_layer_for_lane;

use crate::trace;

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
        for request in &batch.dev_reqs {
            let page_ids_by_layer = page_ids_by_layer_for_lane(
                request,
                cache_lane,
                self.num_layers(),
                self.num_page_ids_per_block(),
                "qwen3.5 GQA request page table",
            );
            if !has_synced_pages(&page_ids_by_layer) {
                continue;
            }
            trace::qwen35_state(|| {
                format!(
                    "event=gqa_prepare req_slot={} cache_lane={} block_index={} num_layers={} page_ids_per_block={:?}",
                    request.req_slot,
                    cache_lane,
                    request.decoder_sync_blocks.block_index(),
                    self.num_layers(),
                    page_ids_by_layer.iter().map(Vec::len).collect::<Vec<_>>()
                )
            });
            for (layer_index, page_ids_by_block) in page_ids_by_layer.iter().enumerate() {
                for (block_offset, page_ids) in page_ids_by_block.iter().enumerate() {
                    assert!(
                        page_ids.iter().all(|&page_id| (page_id as usize) < num_cache_pages),
                        "qwen3.5 runtime supplied a GQA page ID outside the cache-page buffer"
                    );
                    self.write_page_ids(
                        request.req_slot,
                        layer_index,
                        request
                            .decoder_sync_blocks
                            .block_index()
                            .checked_add(block_offset)
                            .expect("qwen3.5 GQA cache-block index must fit usize"),
                        page_ids,
                    );
                }
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
}
