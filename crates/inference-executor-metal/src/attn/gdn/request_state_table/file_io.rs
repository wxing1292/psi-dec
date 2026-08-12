use crate::attn::gdn::request_state_table::GDNRequestSlots;

impl GDNRequestSlots {
    pub fn assert_snapshot_ready(
        &self,
        max_pending_publishes_per_req: usize,
        num_pages_per_state_slot: usize,
        num_cache_pages: usize,
    ) {
        assert!(
            self.txn_state_slots.iter().all(Vec::is_empty),
            "GDN state snapshots require all candidate state transactions to complete"
        );
        assert!(
            self.pending_publish_pages
                .iter()
                .all(|pages| pages.len() <= max_pending_publishes_per_req),
            "GDN pending publish metadata exceeds its configured capacity"
        );
        assert!(
            self.pending_publish_pages
                .iter()
                .flatten()
                .all(|(_, page_ids)| page_ids.len() == num_pages_per_state_slot),
            "GDN pending publish metadata must contain one complete state-slot page mapping"
        );
        assert!(
            self.pending_publish_pages
                .iter()
                .flatten()
                .flat_map(|(_, page_ids)| page_ids)
                .all(|&page_id| (page_id as usize) < num_cache_pages),
            "GDN pending publish metadata contains a page ID outside the cache-page buffer"
        );
        #[cfg(debug_assertions)]
        self.sanity_check();
    }
}
