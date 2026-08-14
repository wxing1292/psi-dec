use std::collections::VecDeque;

use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_macro::sanity_check;
use wincode::SchemaRead;
use wincode::SchemaWrite;

use crate::trace;

mod file_io;

#[derive(Clone, Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
pub struct GDNRequestSlots {
    current_recurrent_state_slots: Vec<u32>,
    current_conv_state_slots: Vec<u32>,
    current_state_versions: Vec<u32>,
    free_recurrent_state_slots: VecDeque<u32>,
    free_conv_state_slots: VecDeque<u32>,
    txn_recurrent_state_slots: Vec<Vec<(u32, u32)>>,
    txn_conv_state_slots: Vec<Vec<(u32, u32)>>,
    pending_publish_pages: Vec<Vec<(u32, Vec<u32>)>>,
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
        let mut free_recurrent_state_slots = (0..num_state_slots_u32).collect::<VecDeque<_>>();
        let mut free_conv_state_slots = (0..num_state_slots_u32).collect::<VecDeque<_>>();
        let current_recurrent_state_slots = (0..num_req_slots)
            .map(|_| {
                free_recurrent_state_slots
                    .pop_front()
                    .expect("GDN request state table initial recurrent state slots exhausted")
            })
            .collect::<Vec<_>>();
        let current_conv_state_slots = (0..num_req_slots)
            .map(|_| {
                free_conv_state_slots
                    .pop_front()
                    .expect("GDN request state table initial convolution state slots exhausted")
            })
            .collect::<Vec<_>>();
        let table = Self {
            current_recurrent_state_slots,
            current_conv_state_slots,
            current_state_versions: vec![0; num_req_slots],
            free_recurrent_state_slots,
            free_conv_state_slots,
            txn_recurrent_state_slots: vec![Vec::new(); num_req_slots],
            txn_conv_state_slots: vec![Vec::new(); num_req_slots],
            pending_publish_pages: vec![Vec::new(); num_req_slots],
            num_state_slots_per_req,
        };
        #[cfg(debug_assertions)]
        table.sanity_check();
        table
    }

    pub fn num_req_slots(&self) -> usize {
        self.current_recurrent_state_slots.len()
    }

    pub fn current_state_version(&self, req_slot: u32) -> u32 {
        self.current_state_versions[self.req_slot_index(req_slot)]
    }

    pub fn current_recurrent_state_slot(&self, req_slot: u32) -> u32 {
        self.current_recurrent_state_slots[self.req_slot_index(req_slot)]
    }

    pub fn current_conv_state_slot(&self, req_slot: u32) -> u32 {
        self.current_conv_state_slots[self.req_slot_index(req_slot)]
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn begin_txn(
        &mut self,
        req_slot: u32,
        recurrent_materialized_state_versions: &[u32],
        conv_materialized_state_versions: &[u32],
        publish_pages: Vec<GDNStatePages>,
    ) {
        let raw_req_slot = req_slot;
        let req_slot_index = self.req_slot_index(req_slot);
        trace::gdn_state(|| {
            let publish_versions = publish_pages
                .iter()
                .map(|pages| pages.state_version)
                .collect::<Vec<_>>();
            format!(
                "event=gdn_table_begin_txn req_slot={} current_recurrent_slot={} current_conv_slot={} \
                 current_version={} materialized_versions={:?} publish_versions={:?} free_recurrent_slots={} \
                 free_conv_slots={}",
                raw_req_slot,
                self.current_recurrent_state_slots[req_slot_index],
                self.current_conv_state_slots[req_slot_index],
                self.current_state_versions[req_slot_index],
                (recurrent_materialized_state_versions, conv_materialized_state_versions),
                publish_versions,
                self.free_recurrent_state_slots.len(),
                self.free_conv_state_slots.len()
            )
        });
        assert!(
            self.txn_recurrent_state_slots[req_slot_index].is_empty(),
            "GDN request state table cannot begin a txn with live recurrent candidate state slots"
        );
        assert!(
            self.txn_conv_state_slots[req_slot_index].is_empty(),
            "GDN request state table cannot begin a txn with live convolution candidate state slots"
        );
        let current_state_version = self.current_state_versions[req_slot_index];
        allocate_txn_state_slots(
            current_state_version,
            recurrent_materialized_state_versions,
            self.num_state_slots_per_req,
            &mut self.free_recurrent_state_slots,
            &mut self.txn_recurrent_state_slots[req_slot_index],
            "recurrent",
        );
        allocate_txn_state_slots(
            current_state_version,
            conv_materialized_state_versions,
            self.num_state_slots_per_req,
            &mut self.free_conv_state_slots,
            &mut self.txn_conv_state_slots[req_slot_index],
            "convolution",
        );
        self.pending_publish_pages[req_slot_index].retain(|(state_version, _)| *state_version > current_state_version);
        for pages in publish_pages {
            self.set_pending_publish_pages(req_slot_index, pages.state_version, pages.page_ids);
        }
        trace::gdn_state(|| {
            format!(
                "event=gdn_table_begin_txn_done req_slot={} txn_recurrent_slots={:?} txn_conv_slots={:?} \
                 queued_publish_versions={:?} free_recurrent_slots={} free_conv_slots={}",
                raw_req_slot,
                self.txn_recurrent_state_slots[req_slot_index],
                self.txn_conv_state_slots[req_slot_index],
                self.pending_publish_pages[req_slot_index]
                    .iter()
                    .map(|(state_version, _)| *state_version)
                    .collect::<Vec<_>>(),
                self.free_recurrent_state_slots.len(),
                self.free_conv_state_slots.len()
            )
        });
    }

    fn candidate_state_slot(
        &self,
        req_slot: u32,
        candidate_state_version: u32,
        current_state_slots: &[u32],
        txn_state_slots: &[Vec<(u32, u32)>],
    ) -> u32 {
        let req_slot_index = self.req_slot_index(req_slot);
        assert!(
            candidate_state_version >= self.current_state_versions[req_slot_index],
            "GDN candidate state_version must not precede current state_version"
        );
        if candidate_state_version == self.current_state_versions[req_slot_index] {
            return current_state_slots[req_slot_index];
        }
        if let Some((_, state_slot)) = txn_state_slots[req_slot_index]
            .iter()
            .find(|&&(state_version, _)| state_version == candidate_state_version)
        {
            return *state_slot;
        }
        panic!("GDN candidate state_version must be registered when beginning txn");
    }

    pub fn candidate_recurrent_state_slot(&self, req_slot: u32, candidate_state_version: u32) -> u32 {
        self.candidate_state_slot(
            req_slot,
            candidate_state_version,
            &self.current_recurrent_state_slots,
            &self.txn_recurrent_state_slots,
        )
    }

    pub fn candidate_conv_state_slot(&self, req_slot: u32, candidate_state_version: u32) -> u32 {
        self.candidate_state_slot(
            req_slot,
            candidate_state_version,
            &self.current_conv_state_slots,
            &self.txn_conv_state_slots,
        )
    }

    pub fn txn_publish_state_versions(&self, req_slot: u32) -> impl Iterator<Item = u32> + '_ {
        self.pending_publish_pages[self.req_slot_index(req_slot)]
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
                "event=gdn_table_reset_req_slot req_slot={} old_current_recurrent_slot={} old_current_conv_slot={} \
                 old_current_version={} txn_recurrent_slots={:?} txn_conv_slots={:?}",
                raw_req_slot,
                self.current_recurrent_state_slots[req_slot_index],
                self.current_conv_state_slots[req_slot_index],
                self.current_state_versions[req_slot_index],
                self.txn_recurrent_state_slots[req_slot_index],
                self.txn_conv_state_slots[req_slot_index]
            )
        });
        self.free_recurrent_state_slots
            .push_back(self.current_recurrent_state_slots[req_slot_index]);
        self.free_conv_state_slots
            .push_back(self.current_conv_state_slots[req_slot_index]);
        for (_, state_slot) in self.txn_recurrent_state_slots[req_slot_index].drain(..) {
            self.free_recurrent_state_slots.push_back(state_slot);
        }
        for (_, state_slot) in self.txn_conv_state_slots[req_slot_index].drain(..) {
            self.free_conv_state_slots.push_back(state_slot);
        }
        self.current_recurrent_state_slots[req_slot_index] = self
            .free_recurrent_state_slots
            .pop_front()
            .expect("GDN request state table reset requires a free recurrent state slot");
        self.current_conv_state_slots[req_slot_index] = self
            .free_conv_state_slots
            .pop_front()
            .expect("GDN request state table reset requires a free convolution state slot");
        self.current_state_versions[req_slot_index] = 0;
        self.pending_publish_pages[req_slot_index].clear();
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn commit_txn(&mut self, req_slot: u32, state_version: u32) -> Vec<GDNStatePublish> {
        let raw_req_slot = req_slot;
        let req_slot_index = self.req_slot_index(req_slot);
        trace::gdn_state(|| {
            format!(
                "event=gdn_table_commit_txn req_slot={} requested_version={} current_recurrent_slot={} \
                 current_conv_slot={} current_version={} txn_recurrent_slots={:?} txn_conv_slots={:?}",
                raw_req_slot,
                state_version,
                self.current_recurrent_state_slots[req_slot_index],
                self.current_conv_state_slots[req_slot_index],
                self.current_state_versions[req_slot_index],
                self.txn_recurrent_state_slots[req_slot_index],
                self.txn_conv_state_slots[req_slot_index]
            )
        });
        if state_version == self.current_state_versions[req_slot_index] {
            for (_, state_slot) in self.txn_recurrent_state_slots[req_slot_index].drain(..) {
                self.free_recurrent_state_slots.push_back(state_slot);
            }
            for (_, state_slot) in self.txn_conv_state_slots[req_slot_index].drain(..) {
                self.free_conv_state_slots.push_back(state_slot);
            }
            trace::gdn_state(|| {
                format!(
                    "event=gdn_table_commit_txn_done req_slot={} new_current_recurrent_slot={} \
                     new_current_conv_slot={} new_current_version={} publishes=0 free_recurrent_slots={} \
                     free_conv_slots={}",
                    raw_req_slot,
                    self.current_recurrent_state_slots[req_slot_index],
                    self.current_conv_state_slots[req_slot_index],
                    self.current_state_versions[req_slot_index],
                    self.free_recurrent_state_slots.len(),
                    self.free_conv_state_slots.len()
                )
            });
            Vec::new()
        } else {
            let new_current_recurrent_state_slot = txn_state_slot(
                &self.txn_recurrent_state_slots[req_slot_index],
                state_version,
                "GDN commit state_version must select a recurrent txn candidate state slot",
            );
            let new_current_conv_state_slot = txn_state_slot(
                &self.txn_conv_state_slots[req_slot_index],
                state_version,
                "GDN commit state_version must select a convolution txn candidate state slot",
            );
            let mut publishes = Vec::new();
            let mut remaining_publish_pages = Vec::new();
            for (publish_state_version, page_ids) in self.pending_publish_pages[req_slot_index].drain(..) {
                if publish_state_version <= state_version {
                    let (src_recurrent_state_slot, src_conv_state_slot) =
                        if publish_state_version == self.current_state_versions[req_slot_index] {
                            (
                                self.current_recurrent_state_slots[req_slot_index],
                                self.current_conv_state_slots[req_slot_index],
                            )
                        } else {
                            (
                                txn_state_slot(
                                    &self.txn_recurrent_state_slots[req_slot_index],
                                    publish_state_version,
                                    "GDN publish state_version must select a materialized recurrent txn state slot",
                                ),
                                txn_state_slot(
                                    &self.txn_conv_state_slots[req_slot_index],
                                    publish_state_version,
                                    "GDN publish state_version must select a materialized convolution txn state slot",
                                ),
                            )
                        };
                    publishes.push(GDNStatePublish {
                        req_slot: req_slot_index.try_into().expect("GDN request slot must fit u32"),
                        src_recurrent_state_slot,
                        src_conv_state_slot,
                        state_version: publish_state_version,
                        page_ids,
                    });
                } else {
                    remaining_publish_pages.push((publish_state_version, page_ids));
                }
            }
            self.pending_publish_pages[req_slot_index] = remaining_publish_pages;
            self.free_recurrent_state_slots
                .push_back(self.current_recurrent_state_slots[req_slot_index]);
            self.free_conv_state_slots
                .push_back(self.current_conv_state_slots[req_slot_index]);
            for (candidate_state_version, state_slot) in self.txn_recurrent_state_slots[req_slot_index].drain(..) {
                if candidate_state_version != state_version {
                    self.free_recurrent_state_slots.push_back(state_slot);
                }
            }
            for (candidate_state_version, state_slot) in self.txn_conv_state_slots[req_slot_index].drain(..) {
                if candidate_state_version != state_version {
                    self.free_conv_state_slots.push_back(state_slot);
                }
            }
            self.current_recurrent_state_slots[req_slot_index] = new_current_recurrent_state_slot;
            self.current_conv_state_slots[req_slot_index] = new_current_conv_state_slot;
            self.current_state_versions[req_slot_index] = state_version;
            trace::gdn_state(|| {
                format!(
                    "event=gdn_table_commit_txn_done req_slot={} new_current_recurrent_slot={} \
                     new_current_conv_slot={} new_current_version={} publishes={} publish_versions={:?} \
                     free_recurrent_slots={} free_conv_slots={}",
                    raw_req_slot,
                    self.current_recurrent_state_slots[req_slot_index],
                    self.current_conv_state_slots[req_slot_index],
                    self.current_state_versions[req_slot_index],
                    publishes.len(),
                    publishes
                        .iter()
                        .map(|publish| publish.state_version)
                        .collect::<Vec<_>>(),
                    self.free_recurrent_state_slots.len(),
                    self.free_conv_state_slots.len()
                )
            });
            publishes
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn restore(&mut self, req_slot: u32, state_version: u32, page_ids: Vec<u32>) -> GDNStateRestore {
        let req_slot_index = self.req_slot_index(req_slot);
        let dst_recurrent_state_slot = self.current_recurrent_state_slots[req_slot_index];
        let dst_conv_state_slot = self.current_conv_state_slots[req_slot_index];
        assert!(
            self.txn_recurrent_state_slots[req_slot_index].is_empty()
                && self.txn_conv_state_slots[req_slot_index].is_empty(),
            "GDN restore cannot replace recurrent or convolution state during a live transaction"
        );
        assert!(
            state_version > self.current_state_versions[req_slot_index],
            "GDN restore must advance the current state version"
        );
        trace::gdn_state(|| {
            format!(
                "event=gdn_table_restore req_slot={} current_recurrent_slot={} current_conv_slot={} \
                 old_current_version={} restored_version={} pages={}",
                req_slot,
                dst_recurrent_state_slot,
                dst_conv_state_slot,
                self.current_state_versions[req_slot_index],
                state_version,
                page_ids.len()
            )
        });
        self.current_state_versions[req_slot_index] = state_version;
        self.pending_publish_pages[req_slot_index]
            .retain(|(publish_state_version, _)| *publish_state_version > state_version);
        GDNStateRestore {
            req_slot,
            dst_recurrent_state_slot,
            dst_conv_state_slot,
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

    fn set_pending_publish_pages(&mut self, req_slot: usize, state_version: u32, page_ids: Vec<u32>) {
        assert!(
            state_version > self.current_state_versions[req_slot],
            "GDN txn publish pages must target a future state_version"
        );
        let publish_pages = &mut self.pending_publish_pages[req_slot];
        match publish_pages.binary_search_by_key(&state_version, |(publish_state_version, _)| *publish_state_version) {
            Ok(index) => publish_pages[index].1 = page_ids,
            Err(index) => publish_pages.insert(index, (state_version, page_ids)),
        }
    }

    fn sanity_check(&self) {
        debug_assert_eq!(
            self.current_recurrent_state_slots.len(),
            self.current_state_versions.len()
        );
        debug_assert_eq!(self.current_conv_state_slots.len(), self.current_state_versions.len());
        debug_assert_eq!(self.txn_recurrent_state_slots.len(), self.current_state_versions.len());
        debug_assert_eq!(self.txn_conv_state_slots.len(), self.current_state_versions.len());
        debug_assert_eq!(self.pending_publish_pages.len(), self.current_state_versions.len());
        let num_state_slots = self
            .num_req_slots()
            .checked_mul(self.num_state_slots_per_req)
            .expect("GDN sanity-check state-slot count must fit usize");
        sanity_check_state_slot_domain(
            &self.current_state_versions,
            &self.current_recurrent_state_slots,
            &self.free_recurrent_state_slots,
            &self.txn_recurrent_state_slots,
            self.num_state_slots_per_req,
            num_state_slots,
            "recurrent",
        );
        sanity_check_state_slot_domain(
            &self.current_state_versions,
            &self.current_conv_state_slots,
            &self.free_conv_state_slots,
            &self.txn_conv_state_slots,
            self.num_state_slots_per_req,
            num_state_slots,
            "convolution",
        );
        for (req_slot, publish_pages) in self.pending_publish_pages.iter().enumerate() {
            let current_version = self.current_state_versions[req_slot];
            let mut previous_publish_version = current_version;
            for (state_version, _) in publish_pages {
                debug_assert!(
                    *state_version > previous_publish_version,
                    "GDN publish state versions must be unique, future, and increasing"
                );
                previous_publish_version = *state_version;
            }
        }
    }
}

fn allocate_txn_state_slots(
    current_state_version: u32,
    materialized_state_versions: &[u32],
    num_state_slots_per_req: usize,
    free_state_slots: &mut VecDeque<u32>,
    txn_state_slots: &mut Vec<(u32, u32)>,
    state_domain: &str,
) {
    debug_assert!(
        materialized_state_versions
            .windows(2)
            .all(|versions| versions[0] < versions[1]),
        "GDN {state_domain} materialized state versions must be unique and increasing"
    );
    for &materialized_state_version in materialized_state_versions {
        assert!(
            materialized_state_version >= current_state_version,
            "GDN {state_domain} materialized state_version must not precede current state_version"
        );
        if materialized_state_version == current_state_version {
            continue;
        }
        assert!(
            txn_state_slots.len() + 1 < num_state_slots_per_req,
            "GDN request {state_domain} state txn exceeds per-request capacity"
        );
        let state_slot = free_state_slots
            .pop_front()
            .unwrap_or_else(|| panic!("GDN request state table free {state_domain} state slots exhausted"));
        txn_state_slots.push((materialized_state_version, state_slot));
    }
}

fn txn_state_slot(txn_state_slots: &[(u32, u32)], state_version: u32, error: &str) -> u32 {
    txn_state_slots
        .iter()
        .find(|&&(candidate_state_version, _)| candidate_state_version == state_version)
        .map(|&(_, state_slot)| state_slot)
        .unwrap_or_else(|| panic!("{error}"))
}

#[allow(clippy::too_many_arguments)]
fn sanity_check_state_slot_domain(
    current_state_versions: &[u32],
    current_state_slots: &[u32],
    free_state_slots: &VecDeque<u32>,
    txn_state_slots: &[Vec<(u32, u32)>],
    num_state_slots_per_req: usize,
    num_state_slots: usize,
    state_domain: &str,
) {
    let mut owned = vec![false; num_state_slots];
    let mut claim = |state_slot: u32, owner: &str| {
        let state_slot_index = state_slot as usize;
        debug_assert!(
            state_slot_index < num_state_slots,
            "GDN {state_domain} {owner} state slot out of range"
        );
        debug_assert!(
            !owned[state_slot_index],
            "GDN {state_domain} state slot has multiple owners: slot={state_slot_index} owner={owner}"
        );
        owned[state_slot_index] = true;
    };
    for &state_slot in current_state_slots {
        claim(state_slot, "current");
    }
    for &state_slot in free_state_slots {
        claim(state_slot, "free");
    }
    for (req_slot, txn_slots) in txn_state_slots.iter().enumerate() {
        debug_assert!(
            txn_slots.len() < num_state_slots_per_req,
            "GDN request {state_domain} txn exceeds its state-slot capacity"
        );
        let mut previous_version = current_state_versions[req_slot];
        for &(state_version, state_slot) in txn_slots {
            debug_assert!(
                state_version > previous_version,
                "GDN {state_domain} txn state versions must be unique and increasing"
            );
            previous_version = state_version;
            claim(state_slot, "candidate");
        }
    }
    debug_assert!(
        owned.into_iter().all(|is_owned| is_owned),
        "GDN {state_domain} state slot is unowned"
    );
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GDNStatePages {
    pub state_version: u32,
    pub page_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GDNStateRestore {
    pub req_slot: u32,
    pub dst_recurrent_state_slot: u32,
    pub dst_conv_state_slot: u32,
    pub state_version: u32,
    pub page_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GDNStatePublish {
    pub req_slot: u32,
    pub src_recurrent_state_slot: u32,
    pub src_conv_state_slot: u32,
    pub state_version: u32,
    pub page_ids: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::GDNRequestSlots;

    #[test]
    #[should_panic(expected = "GDN request state slot count must fit u32")]
    fn test_state_slot_domain_panics() {
        GDNRequestSlots::new(u32::MAX as usize, 2);
    }
}
