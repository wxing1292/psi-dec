use std::collections::VecDeque;

use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_macro::sanity_check;

use crate::trace;

#[derive(Clone, Debug)]
pub struct GDNRequestSlots {
    current_state_slots: Vec<u32>,
    current_state_versions: Vec<u32>,
    free_state_slots: VecDeque<u32>,
    txn_state_slots: Vec<Vec<(u32, u32)>>,
    txn_publish_pages: Vec<Vec<(u32, Vec<u32>)>>,
    num_state_slots_per_req: usize,
}

impl GDNRequestSlots {
    pub fn new(num_req_slots: usize, num_state_slots_per_req: usize) -> Self {
        assert!(
            num_req_slots > 0,
            "GDN request state table requires positive num_req_slots"
        );
        assert!(
            num_state_slots_per_req >= 2,
            "GDN request state table requires at least current and candidate states"
        );
        let num_state_slots_usize = num_req_slots
            .checked_mul(num_state_slots_per_req)
            .expect("GDN request state table state count must fit usize");
        let num_state_slots_u32: u32 = num_state_slots_usize
            .try_into()
            .expect("GDN request state slot count must fit u32");
        let mut free_state_slots = (0..num_state_slots_u32).collect::<VecDeque<_>>();
        let current_state_slots = (0..num_req_slots)
            .map(|_| {
                free_state_slots
                    .pop_front()
                    .expect("GDN request state table initial state slots exhausted")
            })
            .collect();
        let table = Self {
            current_state_slots,
            current_state_versions: vec![0; num_req_slots],
            free_state_slots,
            txn_state_slots: vec![Vec::new(); num_req_slots],
            txn_publish_pages: vec![Vec::new(); num_req_slots],
            num_state_slots_per_req,
        };
        #[cfg(debug_assertions)]
        table.sanity_check();
        table
    }

    pub fn num_req_slots(&self) -> usize {
        self.current_state_slots.len()
    }

    pub fn current_state_version(&self, req_slot: u32) -> u32 {
        self.current_state_versions[self.req_slot_index(req_slot)]
    }

    pub fn current_state_slot(&self, req_slot: u32) -> u32 {
        self.current_state_slots[self.req_slot_index(req_slot)]
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn begin_txn(&mut self, req_slot: u32, candidate_state_versions: &[u32], publish_pages: Vec<GDNStatePages>) {
        let raw_req_slot = req_slot;
        let req_slot_index = self.req_slot_index(req_slot);
        trace::gdn_state(|| {
            let publish_versions = publish_pages
                .iter()
                .map(|pages| pages.state_version)
                .collect::<Vec<_>>();
            format!(
                "event=gdn_table_begin_txn req_slot={} current_slot={} current_version={} candidates={:?} \
                 publish_versions={:?} free_slots={}",
                raw_req_slot,
                self.current_state_slots[req_slot_index],
                self.current_state_versions[req_slot_index],
                candidate_state_versions,
                publish_versions,
                self.free_state_slots.len()
            )
        });
        assert!(
            self.txn_state_slots[req_slot_index].is_empty(),
            "GDN request state table cannot begin a txn with live candidate state slots"
        );
        let current_state_version = self.current_state_versions[req_slot_index];
        for &candidate_state_version in candidate_state_versions {
            assert!(
                candidate_state_version >= current_state_version,
                "GDN candidate state_version must not precede current state_version"
            );
            if candidate_state_version == current_state_version
                || self.txn_state_slots[req_slot_index]
                    .iter()
                    .any(|&(state_version, _)| state_version == candidate_state_version)
            {
                continue;
            }
            assert!(
                self.txn_state_slots[req_slot_index].len() + 1 < self.num_state_slots_per_req,
                "GDN request state table txn exceeds per-request state capacity"
            );
            let state_slot = self
                .free_state_slots
                .pop_front()
                .expect("GDN request state table free state slots exhausted");
            self.txn_state_slots[req_slot_index].push((candidate_state_version, state_slot));
        }
        self.txn_state_slots[req_slot_index].sort_by_key(|&(state_version, _)| state_version);
        self.txn_publish_pages[req_slot_index].retain(|(state_version, _)| *state_version > current_state_version);
        for pages in publish_pages {
            self.set_txn_publish_pages(req_slot_index, pages.state_version, pages.page_ids);
        }
        trace::gdn_state(|| {
            format!(
                "event=gdn_table_begin_txn_done req_slot={} txn_slots={:?} queued_publish_versions={:?} free_slots={}",
                raw_req_slot,
                self.txn_state_slots[req_slot_index],
                self.txn_publish_pages[req_slot_index]
                    .iter()
                    .map(|(state_version, _)| *state_version)
                    .collect::<Vec<_>>(),
                self.free_state_slots.len()
            )
        });
    }

    pub fn candidate_state_slot(&self, req_slot: u32, candidate_state_version: u32) -> u32 {
        let req_slot_index = self.req_slot_index(req_slot);
        assert!(
            candidate_state_version >= self.current_state_versions[req_slot_index],
            "GDN candidate state_version must not precede current state_version"
        );
        if candidate_state_version == self.current_state_versions[req_slot_index] {
            return self.current_state_slots[req_slot_index];
        }
        if let Some((_, state_slot)) = self.txn_state_slots[req_slot_index]
            .iter()
            .find(|&&(state_version, _)| state_version == candidate_state_version)
        {
            return *state_slot;
        }
        panic!("GDN candidate state_version must be registered when beginning txn");
    }

    pub fn txn_publish_state_versions(&self, req_slot: u32) -> impl Iterator<Item = u32> + '_ {
        self.txn_publish_pages[self.req_slot_index(req_slot)]
            .iter()
            .map(|(state_version, _)| *state_version)
    }

    pub fn reset_req_slots(&mut self, req_slots: &[RawRequestSlot]) {
        for &req_slot in req_slots {
            self.reset_req_slot(req_slot);
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn reset_req_slot(&mut self, req_slot: u32) {
        let raw_req_slot = req_slot;
        let req_slot_index = self.req_slot_index(req_slot);
        trace::gdn_state(|| {
            format!(
                "event=gdn_table_reset_req_slot req_slot={} old_current_slot={} old_current_version={} txn_slots={:?}",
                raw_req_slot,
                self.current_state_slots[req_slot_index],
                self.current_state_versions[req_slot_index],
                self.txn_state_slots[req_slot_index]
            )
        });
        self.free_state_slots
            .push_back(self.current_state_slots[req_slot_index]);
        for (_, state_slot) in self.txn_state_slots[req_slot_index].drain(..) {
            self.free_state_slots.push_back(state_slot);
        }
        self.current_state_slots[req_slot_index] = self
            .free_state_slots
            .pop_front()
            .expect("GDN request state table reset requires a free state slot");
        self.current_state_versions[req_slot_index] = 0;
        self.txn_publish_pages[req_slot_index].clear();
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn commit_txn(&mut self, req_slot: u32, state_version: u32) -> Vec<GDNStatePublish> {
        let raw_req_slot = req_slot;
        let req_slot_index = self.req_slot_index(req_slot);
        trace::gdn_state(|| {
            format!(
                "event=gdn_table_commit_txn req_slot={} requested_version={} current_slot={} current_version={} \
                 txn_slots={:?}",
                raw_req_slot,
                state_version,
                self.current_state_slots[req_slot_index],
                self.current_state_versions[req_slot_index],
                self.txn_state_slots[req_slot_index]
            )
        });
        if state_version == self.current_state_versions[req_slot_index] {
            for (_, state_slot) in self.txn_state_slots[req_slot_index].drain(..) {
                self.free_state_slots.push_back(state_slot);
            }
            trace::gdn_state(|| {
                format!(
                    "event=gdn_table_commit_txn_done req_slot={} new_current_slot={} new_current_version={} \
                     publishes=0 free_slots={}",
                    raw_req_slot,
                    self.current_state_slots[req_slot_index],
                    self.current_state_versions[req_slot_index],
                    self.free_state_slots.len()
                )
            });
            Vec::new()
        } else {
            let txn_index = self.txn_state_slots[req_slot_index]
                .iter()
                .position(|&(candidate_state_version, _)| candidate_state_version == state_version)
                .expect("GDN commit state_version must select a txn candidate state slot");
            let new_current_state_slot = self.txn_state_slots[req_slot_index][txn_index].1;
            let mut publishes = Vec::new();
            let mut remaining_publish_pages = Vec::new();
            for (publish_state_version, page_ids) in self.txn_publish_pages[req_slot_index].drain(..) {
                if publish_state_version <= state_version {
                    let src_state_slot = if publish_state_version == self.current_state_versions[req_slot_index] {
                        self.current_state_slots[req_slot_index]
                    } else {
                        self.txn_state_slots[req_slot_index]
                            .iter()
                            .find(|&&(candidate_state_version, _)| candidate_state_version == publish_state_version)
                            .map(|&(_, state_slot)| state_slot)
                            .expect("GDN publish state_version must select a materialized txn state slot")
                    };
                    publishes.push(GDNStatePublish {
                        req_slot: req_slot_index.try_into().expect("GDN request slot must fit u32"),
                        src_state_slot,
                        state_version: publish_state_version,
                        page_ids,
                    });
                } else {
                    remaining_publish_pages.push((publish_state_version, page_ids));
                }
            }
            self.txn_publish_pages[req_slot_index] = remaining_publish_pages;
            self.free_state_slots
                .push_back(self.current_state_slots[req_slot_index]);
            for (candidate_state_version, state_slot) in self.txn_state_slots[req_slot_index].drain(..) {
                if candidate_state_version != state_version {
                    self.free_state_slots.push_back(state_slot);
                }
            }
            self.current_state_slots[req_slot_index] = new_current_state_slot;
            self.current_state_versions[req_slot_index] = state_version;
            trace::gdn_state(|| {
                format!(
                    "event=gdn_table_commit_txn_done req_slot={} new_current_slot={} new_current_version={} \
                     publishes={} publish_versions={:?} free_slots={}",
                    raw_req_slot,
                    self.current_state_slots[req_slot_index],
                    self.current_state_versions[req_slot_index],
                    publishes.len(),
                    publishes
                        .iter()
                        .map(|publish| publish.state_version)
                        .collect::<Vec<_>>(),
                    self.free_state_slots.len()
                )
            });
            publishes
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn restore(&mut self, req_slot: u32, state_version: u32, page_ids: Vec<u32>) -> GDNStateRestore {
        let req_slot_index = self.req_slot_index(req_slot);
        let dst_state_slot = self.current_state_slots[req_slot_index];
        assert!(
            self.txn_state_slots[req_slot_index].is_empty(),
            "GDN restore cannot replace state during a live transaction"
        );
        assert!(
            state_version > self.current_state_versions[req_slot_index],
            "GDN restore must advance the current state version"
        );
        trace::gdn_state(|| {
            format!(
                "event=gdn_table_restore req_slot={} old_current_slot={} old_current_version={} dst_state_slot={} \
                 restored_version={} pages={}",
                req_slot,
                self.current_state_slots[req_slot_index],
                self.current_state_versions[req_slot_index],
                dst_state_slot,
                state_version,
                page_ids.len()
            )
        });
        self.current_state_slots[req_slot_index] = dst_state_slot;
        self.current_state_versions[req_slot_index] = state_version;
        self.txn_publish_pages[req_slot_index]
            .retain(|(publish_state_version, _)| *publish_state_version > state_version);
        GDNStateRestore {
            req_slot,
            dst_state_slot,
            state_version,
            page_ids,
        }
    }

    fn req_slot_index(&self, req_slot: u32) -> usize {
        let req_slot_index = req_slot as usize;
        assert!(
            req_slot_index < self.num_req_slots(),
            "GDN request state table req_slot out of range"
        );
        req_slot_index
    }

    fn set_txn_publish_pages(&mut self, req_slot: usize, state_version: u32, page_ids: Vec<u32>) {
        assert!(
            state_version > self.current_state_versions[req_slot],
            "GDN txn publish pages must target a future state_version"
        );
        let publish_pages = &mut self.txn_publish_pages[req_slot];
        if let Some((_, existing_page_ids)) = publish_pages
            .iter_mut()
            .find(|(publish_state_version, _)| *publish_state_version == state_version)
        {
            *existing_page_ids = page_ids;
            return;
        }
        publish_pages.push((state_version, page_ids));
        publish_pages.sort_by_key(|(publish_state_version, _)| *publish_state_version);
    }

    fn sanity_check(&self) {
        debug_assert_eq!(self.current_state_slots.len(), self.current_state_versions.len());
        debug_assert_eq!(self.current_state_slots.len(), self.txn_state_slots.len());
        debug_assert_eq!(self.current_state_slots.len(), self.txn_publish_pages.len());
        let num_state_slots = self
            .num_req_slots()
            .checked_mul(self.num_state_slots_per_req)
            .expect("GDN sanity-check state-slot count must fit usize");
        let mut owned = vec![false; num_state_slots];
        let mut claim = |state_slot: u32, owner: &str| {
            let state_slot_index = usize::try_from(state_slot).expect("GDN state slot must fit host usize");
            debug_assert!(
                state_slot_index < num_state_slots,
                "GDN {owner} state slot out of range"
            );
            debug_assert!(
                !owned[state_slot_index],
                "GDN state slot has multiple owners: slot={state_slot_index} owner={owner}"
            );
            owned[state_slot_index] = true;
        };
        for &state_slot in &self.current_state_slots {
            claim(state_slot, "current");
        }
        for &state_slot in &self.free_state_slots {
            claim(state_slot, "free");
        }
        for (req_slot, txn_slots) in self.txn_state_slots.iter().enumerate() {
            debug_assert!(
                txn_slots.len() < self.num_state_slots_per_req,
                "GDN request txn exceeds its state-slot capacity"
            );
            let current_version = self.current_state_versions[req_slot];
            let mut previous_version = current_version;
            for &(state_version, state_slot) in txn_slots {
                debug_assert!(
                    state_version > previous_version,
                    "GDN txn state versions must be unique and increasing"
                );
                previous_version = state_version;
                claim(state_slot, "candidate");
            }
            let mut previous_publish_version = current_version;
            for (state_version, _) in &self.txn_publish_pages[req_slot] {
                debug_assert!(
                    *state_version > previous_publish_version,
                    "GDN publish state versions must be unique, future, and increasing"
                );
                previous_publish_version = *state_version;
            }
        }
        debug_assert!(owned.into_iter().all(|is_owned| is_owned), "GDN state slot is unowned");
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GDNStatePages {
    pub state_version: u32,
    pub page_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GDNStateRestore {
    pub req_slot: u32,
    pub dst_state_slot: u32,
    pub state_version: u32,
    pub page_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GDNStatePublish {
    pub req_slot: u32,
    pub src_state_slot: u32,
    pub state_version: u32,
    pub page_ids: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::GDNRequestSlots;
    use super::GDNStatePages;
    use super::GDNStatePublish;
    use super::GDNStateRestore;

    #[test]
    #[should_panic(expected = "GDN request state slot count must fit u32")]
    fn test_state_slot_domain_panics() {
        GDNRequestSlots::new(u32::MAX as usize, 2);
    }

    #[test]
    fn test_state_slots() {
        let mut table = GDNRequestSlots::new(4, 4);

        assert_eq!(table.current_state_slot(2), 2);
        assert_eq!(table.current_state_version(2), 0);

        table.begin_txn(2, &[17], Vec::new());
        let candidate = table.candidate_state_slot(2, 17);

        assert_ne!(candidate, table.current_state_slot(2));
        assert_eq!(table.candidate_state_slot(2, 17), candidate);
        assert_eq!(table.commit_txn(2, 17), Vec::<GDNStatePublish>::new());
        assert_eq!(table.current_state_slot(2), candidate);
        assert_eq!(table.current_state_version(2), 17);
    }

    #[test]
    fn test_commit_publish() {
        let mut table = GDNRequestSlots::new(2, 3);

        table.begin_txn(
            1,
            &[8, 16],
            vec![
                state_pages(8, vec![80]),
                state_pages(16, vec![160]),
                state_pages(16, vec![161]),
            ],
        );
        let boundary = table.candidate_state_slot(1, 8);
        let committed = table.candidate_state_slot(1, 16);

        assert_eq!(
            table.commit_txn(1, 16),
            vec![
                GDNStatePublish {
                    req_slot: 1,
                    src_state_slot: boundary,
                    state_version: 8,
                    page_ids: vec![80],
                },
                GDNStatePublish {
                    req_slot: 1,
                    src_state_slot: committed,
                    state_version: 16,
                    page_ids: vec![161],
                },
            ]
        );
        assert_eq!(table.current_state_slot(1), committed);
        assert_eq!(table.current_state_version(1), 16);
    }

    #[test]
    fn test_commit_future() {
        let mut table = GDNRequestSlots::new(1, 3);

        table.begin_txn(0, &[8], vec![state_pages(8, vec![80]), state_pages(16, vec![160])]);
        let state_slot_at_version_8 = table.candidate_state_slot(0, 8);

        assert_eq!(
            table.commit_txn(0, 8),
            vec![GDNStatePublish {
                req_slot: 0,
                src_state_slot: state_slot_at_version_8,
                state_version: 8,
                page_ids: vec![80],
            }]
        );

        table.begin_txn(0, &[16], Vec::new());
        let state_slot_at_version_16 = table.candidate_state_slot(0, 16);
        assert_eq!(
            table.commit_txn(0, 16),
            vec![GDNStatePublish {
                req_slot: 0,
                src_state_slot: state_slot_at_version_16,
                state_version: 16,
                page_ids: vec![160],
            }]
        );
    }

    #[test]
    fn test_commit_current() {
        let mut table = GDNRequestSlots::new(1, 3);

        table.begin_txn(0, &[8], vec![state_pages(8, vec![80])]);
        let current = table.current_state_slot(0);
        let _candidate = table.candidate_state_slot(0, 8);

        assert_eq!(table.commit_txn(0, 0), Vec::<GDNStatePublish>::new());
        assert_eq!(table.current_state_slot(0), current);
        assert_eq!(table.current_state_version(0), 0);

        table.begin_txn(0, &[8], Vec::new());
        let candidate = table.candidate_state_slot(0, 8);
        assert_ne!(candidate, current);
    }

    #[test]
    fn test_restore() {
        let mut table = GDNRequestSlots::new(2, 3);

        let restore = table.restore(1, 9, vec![1, 2]);

        assert_eq!(
            restore,
            GDNStateRestore {
                req_slot: 1,
                dst_state_slot: 1,
                state_version: 9,
                page_ids: vec![1, 2],
            }
        );
        assert_eq!(table.current_state_slot(1), 1);
        assert_eq!(table.current_state_version(1), 9);
    }

    #[test]
    fn test_reset() {
        let mut table = GDNRequestSlots::new(2, 3);

        table.begin_txn(1, &[9], vec![state_pages(16, vec![1, 2])]);
        let _candidate = table.candidate_state_slot(1, 9);
        table.reset_req_slot(1);

        assert_eq!(table.current_state_version(1), 0);
        table.begin_txn(1, &[4], Vec::new());
        let candidate = table.candidate_state_slot(1, 4);
        assert_ne!(candidate, table.current_state_slot(1));
    }

    fn state_pages(state_version: u32, page_ids: Vec<u32>) -> GDNStatePages {
        GDNStatePages {
            state_version,
            page_ids,
        }
    }
}
