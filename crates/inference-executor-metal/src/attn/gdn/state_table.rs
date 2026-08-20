use std::cell::RefCell;
use std::cmp::Ordering;
use std::mem::size_of;
use std::mem::take;
use std::rc::Rc;

use inference_backend_metal::components::gdn::state_pages as backend_state_pages;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::gdn::state::GDNStateTxn;
use inference_executor_core::attn::gdn::state::to_candidate_state_version;
use inference_executor_core::attn::gdn::state::to_state_version;
use inference_executor_core::backend::recorder::Recorder;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gdn::request_slots::GDNRequestSlots;
use crate::attn::gdn::request_slots::GDNStatePages;
use crate::attn::gdn::request_slots::GDNStatePublish;
use crate::attn::gdn::request_slots::GDNStateRestore;
use crate::def::replay_op::ReplayOp;
use crate::trace;

mod file_io;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNStateLayout {
    num_gdn_layers: usize,
    num_state_slots: usize,
    max_state_io_requests: usize,
    recurrent_state_bytes: usize,
    conv_state_bytes: usize,
    page_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNStateCapacity {
    num_state_slots_per_req: usize,
    max_materialized_states_per_req: usize,
    max_publish_jobs_per_req: usize,
}

impl GDNStateCapacity {
    pub fn new(
        num_state_slots_per_req: usize,
        max_materialized_states_per_req: usize,
        max_publish_jobs_per_req: usize,
    ) -> Self {
        let capacity = Self {
            num_state_slots_per_req,
            max_materialized_states_per_req,
            max_publish_jobs_per_req,
        };
        capacity.validate();
        capacity
    }

    fn validate(self) {
        assert!(
            self.max_materialized_states_per_req > 0,
            "GDN state requires materialized-state capacity"
        );
        assert!(
            self.num_state_slots_per_req > self.max_materialized_states_per_req,
            "GDN state slots must include current state and all materialized states"
        );
        assert!(self.max_publish_jobs_per_req > 0, "GDN state requires publish capacity");
    }
}

pub struct GDNRequestStateTable {
    layout: GDNStateLayout,
    num_req_slots: usize,
    num_state_slots_per_req: usize,
    num_tokens_per_block: usize,
    num_cache_pages: usize,
    max_materialized_states_per_req: usize,
    max_publish_jobs_per_req: usize,
    resources: Option<Rc<GDNRequestStateResources>>,
    request_table: Option<RefCell<GDNRequestSlots>>,
    restores: RefCell<Vec<GDNStateRestore>>,
    publishes: RefCell<Vec<GDNStatePublish>>,
    pending_request_txns: RefCell<Vec<GDNStateRequestTxn>>,
}

pub struct GDNRequestStateResources {
    layout: GDNStateLayout,
    recurrent_states: Buffer,
    conv_states: Buffer,
    page_io: GDNStatePageIO,
}

pub struct GDNStatePageIO {
    page_ids: Buffer,
    recurrent_state_slots: Buffer,
    conv_state_slots: Buffer,
    read: backend_state_pages::Read,
    write: backend_state_pages::Write,
}

pub struct GDNPreparedRequestState {
    pub src_recurrent_state_slots: Vec<u32>,
    pub src_conv_state_slots: Vec<u32>,
    /// Persistent recurrent state slot for each forward row.
    ///
    /// `u32::MAX` means that the row produces its normal output, but its recurrent state
    /// must not be written to the persistent recurrent state arena.
    pub flat_materialized_recurrent_state_slots: Vec<u32>,
    /// Persistent convolution state slot for each forward row.
    ///
    /// `u32::MAX` means that the row produces its normal output, but its convolution state
    /// must not be written to the persistent convolution state arena.
    pub flat_materialized_conv_state_slots: Vec<u32>,
}

struct GDNPrepareOutput {
    prepared: GDNPreparedRequestState,
    request_table: GDNRequestSlots,
    restores: Vec<GDNStateRestore>,
    publishes: Vec<GDNStatePublish>,
    pending_request_txns: Vec<GDNStateRequestTxn>,
}

struct GDNPrepareInput {
    num_tokens_per_block: usize,
    max_materialized_states_per_req: usize,
    request_table: GDNRequestSlots,
    req_slots: Vec<u32>,
    block_indices: Vec<usize>,
    token_indices: Vec<u32>,
    cu_tokens: Vec<u32>,
    state_txns: Vec<GDNStateTxn>,
    state_page_ids_by_req: Vec<Vec<Vec<u32>>>,
    num_pages_per_state_slot: usize,
}

#[derive(Clone, Copy)]
struct GDNStateRequestTxn {
    req_slot: u32,
    txn: GDNStateTxn,
}

#[derive(Clone, Copy)]
pub struct GDNStateArenaBindings<'a> {
    pub recurrent_states: &'a Buffer,
    pub recurrent_layer_offset_bytes: u64,
    pub conv_states: &'a Buffer,
    pub conv_layer_offset_bytes: u64,
}

impl GDNRequestStateTable {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        cores: &[GDNCore],
        num_req_slots: usize,
        capacity: GDNStateCapacity,
        num_tokens_per_block: usize,
        num_cache_pages: usize,
        page_bytes: usize,
    ) -> Self {
        assert!(!cores.is_empty(), "GDN state requires layers");
        assert!(num_tokens_per_block > 0, "GDN state requires tokens per block");
        assert!(num_cache_pages > 0, "GDN state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "GDN cache page IDs must fit u32"
        );
        assert!(page_bytes.is_power_of_two(), "GDN page size must be a power of two");
        assert!(
            page_bytes.is_multiple_of(size_of::<f32>() * 4),
            "GDN page size must contain an integral number of float4 values"
        );
        let request_table = GDNRequestSlots::new(num_req_slots, capacity.num_state_slots_per_req);
        let num_state_slots = num_req_slots
            .checked_mul(capacity.num_state_slots_per_req)
            .expect("GDN state slot count overflow");

        for core in cores {
            core.validate();
        }
        let first_core = &cores[0];
        let recurrent_state_bytes = [
            first_core.num_v_heads,
            first_core.v_head_dim,
            first_core.qk_head_dim,
            size_of::<f32>(),
        ]
        .into_iter()
        .try_fold(1usize, |product, factor| product.checked_mul(factor))
        .expect("GDN recurrent state slot size must fit usize");
        let conv_state_bytes = [first_core.qkv_dim(), first_core.conv_state_len(), size_of::<f32>()]
            .into_iter()
            .try_fold(1usize, |product, factor| product.checked_mul(factor))
            .expect("GDN convolution state slot size must fit usize");
        u32::try_from(recurrent_state_bytes).expect("GDN recurrent state slot bytes must fit shader u32");
        u32::try_from(conv_state_bytes).expect("GDN convolution state slot bytes must fit shader u32");
        assert!(
            cores.iter().all(|core| {
                core.num_v_heads == first_core.num_v_heads
                    && core.v_head_dim == first_core.v_head_dim
                    && core.qk_head_dim == first_core.qk_head_dim
                    && core.qkv_dim() == first_core.qkv_dim()
                    && core.conv_state_len() == first_core.conv_state_len()
            }),
            "GDN all-layer state IO requires one shared layer layout"
        );
        let num_gdn_layers = cores.len();
        let max_state_io_requests = num_req_slots.max(
            num_req_slots
                .checked_mul(capacity.max_publish_jobs_per_req)
                .expect("GDN publish job count overflow"),
        );
        let layout = GDNStateLayout {
            num_gdn_layers,
            num_state_slots,
            max_state_io_requests,
            recurrent_state_bytes,
            conv_state_bytes,
            page_bytes,
        };
        Self {
            layout,
            num_req_slots,
            num_state_slots_per_req: capacity.num_state_slots_per_req,
            num_tokens_per_block,
            num_cache_pages,
            max_materialized_states_per_req: capacity.max_materialized_states_per_req,
            max_publish_jobs_per_req: capacity.max_publish_jobs_per_req,
            resources: Some(Rc::new(GDNRequestStateResources::new(device, layout))),
            request_table: Some(RefCell::new(request_table)),
            restores: RefCell::new(Vec::with_capacity(num_req_slots)),
            publishes: RefCell::new(Vec::with_capacity(layout.max_state_io_requests)),
            pending_request_txns: RefCell::new(Vec::with_capacity(num_req_slots)),
        }
    }

    pub fn num_pages_per_state_slot(&self) -> usize {
        (self.recurrent_state_bytes().div_ceil(self.layout.page_bytes)
            + self.conv_state_bytes().div_ceil(self.layout.page_bytes))
            * self.layout.num_gdn_layers
    }

    pub fn num_req_slots(&self) -> usize {
        self.request_table().borrow().num_req_slots()
    }

    pub fn num_layers(&self) -> usize {
        self.layout.num_gdn_layers
    }

    pub fn release_resources(&mut self) {
        self.assert_snapshot_ready();
        self.request_table
            .take()
            .expect("GDN request-state table must be loaded");
        self.resources
            .take()
            .expect("GDN request-state resources must be loaded");
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        assert!(
            self.request_table.is_none() && self.resources.is_none(),
            "GDN request-state resources are already loaded"
        );
        self.request_table = Some(RefCell::new(GDNRequestSlots::new(
            self.num_req_slots,
            self.num_state_slots_per_req,
        )));
        self.resources = Some(Rc::new(GDNRequestStateResources::new(device, self.layout)));
    }

    fn recurrent_state_bytes(&self) -> usize {
        self.layout.recurrent_state_bytes
    }

    fn conv_state_bytes(&self) -> usize {
        self.layout.conv_state_bytes
    }

    pub fn resources(&self) -> &Rc<GDNRequestStateResources> {
        self.resources
            .as_ref()
            .expect("GDN request-state resources must be loaded before execution")
    }

    fn request_table(&self) -> &RefCell<GDNRequestSlots> {
        self.request_table
            .as_ref()
            .expect("GDN request-state table must be loaded before execution")
    }

    fn assert_snapshot_ready(&self) {
        assert!(
            self.restores.borrow().is_empty(),
            "GDN state snapshot requires no restore jobs"
        );
        assert!(
            self.publishes.borrow().is_empty(),
            "GDN state snapshot requires no publish jobs"
        );
        assert!(
            self.pending_request_txns.borrow().is_empty(),
            "GDN state snapshot requires no pending batch transactions"
        );
    }

    pub fn layer_bindings(&self, gdn_layer_index: usize) -> GDNStateArenaBindings<'_> {
        self.resources().layer_bindings(gdn_layer_index)
    }

    pub fn restores(&self) -> Vec<GDNStateRestore> {
        self.restores.borrow().clone()
    }

    pub fn publishes(&self) -> Vec<GDNStatePublish> {
        self.publishes.borrow().clone()
    }

    pub fn prepare_restore(&self) -> bool {
        let restores = self.restores.borrow();
        if restores.is_empty() {
            return false;
        }
        assert!(
            restores.len() <= self.request_table().borrow().num_req_slots(),
            "GDN restore I/O requests exceed request-slot capacity"
        );
        self.resources()
            .page_io
            .prepare_restore(&restores, self.num_pages_per_state_slot());
        true
    }

    pub fn record_restore<'a, R>(&'a self, recorder: &mut R, pages: &'a Buffer)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let restores = self.restores.borrow();
        let resources = self.resources();
        resources.page_io.record_restore(
            recorder,
            pages,
            &resources.recurrent_states,
            &resources.conv_states,
            self.layout,
            &restores,
        );
    }

    pub fn finish_restore(&self) {
        self.restores.borrow_mut().clear();
    }

    pub fn record_publish<'a, R>(&'a self, recorder: &mut R, pages: &'a Buffer) -> bool
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let publishes = self.publishes.borrow();
        if publishes.is_empty() {
            return false;
        }
        assert!(
            publishes.len() <= self.layout.max_state_io_requests,
            "GDN publish I/O requests exceed state-I/O request capacity"
        );
        let resources = self.resources();
        resources
            .page_io
            .prepare_publish(&publishes, self.num_pages_per_state_slot());
        resources.page_io.record_publish(
            recorder,
            pages,
            &resources.recurrent_states,
            &resources.conv_states,
            self.layout,
            &publishes,
        );
        true
    }

    pub fn finish_publish(&self) {
        self.publishes.borrow_mut().clear();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        req_slots: &[u32],
        block_indices: &[usize],
        token_indices: &[u32],
        cu_tokens: &[u32],
        state_txns: &[GDNStateTxn],
        state_page_ids_by_req: &[Vec<Vec<u32>>],
    ) -> GDNPreparedRequestState {
        self.validate_batch(
            req_slots,
            block_indices,
            token_indices,
            cu_tokens,
            state_txns,
            state_page_ids_by_req,
        );
        let mut output = self
            .prepare_input(
                req_slots,
                block_indices,
                token_indices,
                cu_tokens,
                state_txns,
                state_page_ids_by_req,
            )
            .resolve();
        *self.request_table().borrow_mut() = output.request_table;
        *self.restores.borrow_mut() = take(&mut output.restores);
        *self.publishes.borrow_mut() = take(&mut output.publishes);
        *self.pending_request_txns.borrow_mut() = take(&mut output.pending_request_txns);
        output.prepared
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_input(
        &self,
        req_slots: &[u32],
        block_indices: &[usize],
        token_indices: &[u32],
        cu_tokens: &[u32],
        state_txns: &[GDNStateTxn],
        state_page_ids_by_req: &[Vec<Vec<u32>>],
    ) -> GDNPrepareInput {
        GDNPrepareInput {
            num_tokens_per_block: self.num_tokens_per_block,
            max_materialized_states_per_req: self.max_materialized_states_per_req,
            request_table: self.request_table().borrow().clone(),
            req_slots: req_slots.to_vec(),
            block_indices: block_indices.to_vec(),
            token_indices: token_indices.to_vec(),
            cu_tokens: cu_tokens.to_vec(),
            state_txns: state_txns.to_vec(),
            state_page_ids_by_req: state_page_ids_by_req.to_vec(),
            num_pages_per_state_slot: self.num_pages_per_state_slot(),
        }
    }

    pub fn commit(&self, dst_state_versions: &[u32]) {
        let pending_request_txns = take(&mut *self.pending_request_txns.borrow_mut());
        let mut publishes_out = self.publishes.borrow_mut();
        let mut request_table = self.request_table().borrow_mut();
        assert_eq!(pending_request_txns.len(), dst_state_versions.len());
        publishes_out.clear();
        for (request_txn, &dst_state_version) in pending_request_txns.iter().zip(dst_state_versions) {
            assert!(
                request_txn.txn.contains_dst_state_version(dst_state_version),
                "GDN commit state version must select the transaction's destination range"
            );
            let candidate_state_version =
                to_candidate_state_version(dst_state_version, request_txn.txn.candidate_state_version_shift());
            assert!(
                request_txn
                    .txn
                    .contains_candidate_state_version(candidate_state_version),
                "GDN shifted commit state version must select a recorded candidate"
            );
            let publishes = request_table.commit_txn(request_txn.req_slot, candidate_state_version);
            assert!(
                publishes.len() <= self.max_publish_jobs_per_req,
                "GDN publishes exceed per-request capacity"
            );
            publishes_out.extend(publishes);
        }
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        let mut request_table = self.request_table().borrow_mut();
        request_table.reset_req_slots(req_slots);
        for &req_slot in req_slots {
            self.zero_state_slots(
                request_table.current_recurrent_state_slot(req_slot),
                request_table.current_conv_state_slot(req_slot),
            );
        }
    }

    pub fn reset_req_slot(&self, req_slot: RawRequestSlot) {
        self.reset_req_slots(&[req_slot]);
    }

    fn zero_state_slots(&self, recurrent_state_slot: u32, conv_state_slot: u32) {
        let recurrent_state_slot_index = recurrent_state_slot as usize;
        let conv_state_slot_index = conv_state_slot as usize;
        assert!(recurrent_state_slot_index < self.layout.num_state_slots);
        assert!(conv_state_slot_index < self.layout.num_state_slots);
        let resources = self.resources();
        for gdn_layer_index in 0..self.layout.num_gdn_layers {
            let recurrent_state_slot_offset_bytes = (gdn_layer_index * self.layout.num_state_slots
                + recurrent_state_slot_index)
                * self.recurrent_state_bytes();
            debug_assert!(
                recurrent_state_slot_offset_bytes + self.recurrent_state_bytes()
                    <= resources.recurrent_states.len_bytes()
            );
            resources
                .recurrent_states
                .zero_bytes(recurrent_state_slot_offset_bytes, self.recurrent_state_bytes());
            let conv_state_slot_offset_bytes =
                (gdn_layer_index * self.layout.num_state_slots + conv_state_slot_index) * self.conv_state_bytes();
            debug_assert!(conv_state_slot_offset_bytes + self.conv_state_bytes() <= resources.conv_states.len_bytes());
            resources
                .conv_states
                .zero_bytes(conv_state_slot_offset_bytes, self.conv_state_bytes());
        }
        trace::gdn_state(|| {
            format!("event=gdn_state_zero recurrent_slot={recurrent_state_slot} conv_slot={conv_state_slot}")
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_batch(
        &self,
        req_slots: &[u32],
        block_indices: &[usize],
        token_indices: &[u32],
        cu_tokens: &[u32],
        state_txns: &[GDNStateTxn],
        state_page_ids_by_req: &[Vec<Vec<u32>>],
    ) {
        assert!(!req_slots.is_empty(), "GDN state batch requires requests");
        assert_eq!(block_indices.len(), req_slots.len());
        assert_eq!(token_indices.len(), req_slots.len());
        assert_eq!(state_txns.len(), req_slots.len());
        assert_eq!(state_page_ids_by_req.len(), req_slots.len());
        assert_eq!(cu_tokens.len(), req_slots.len() + 1);
        assert_eq!(cu_tokens[0], 0, "GDN state batch cu_tokens must start at zero");
        let num_req_slots = self.request_table().borrow().num_req_slots();
        assert!(
            req_slots.len() <= num_req_slots,
            "GDN request count exceeds state-table capacity"
        );
        assert!(
            req_slots.iter().all(|&req_slot| (req_slot as usize) < num_req_slots),
            "GDN request slot exceeds state-table capacity"
        );
        let num_pages_per_state_slot = self.num_pages_per_state_slot();
        for req_index in 0..req_slots.len() {
            let txn = state_txns[req_index];
            let num_tokens = cu_tokens[req_index + 1]
                .checked_sub(cu_tokens[req_index])
                .expect("GDN state batch cu_tokens must be nondecreasing");
            assert!(num_tokens > 0, "GDN state batch requires tokens for every request");
            assert_eq!(
                txn.dst_start_state_version(),
                to_state_version(token_indices[req_index])
            );
            assert_eq!(txn.num_forward_tokens(), num_tokens);
            assert!(
                txn.num_candidate_states() as usize <= self.max_materialized_states_per_req,
                "GDN candidate range exceeds per-request capacity"
            );
            let state_blocks = &state_page_ids_by_req[req_index];
            if !state_blocks.is_empty() {
                let block_count = block_indices[req_index]
                    .checked_add(state_blocks.len())
                    .expect("GDN cache block range must fit usize");
                let block_end = block_count
                    .checked_mul(self.num_tokens_per_block)
                    .expect("GDN cache block end must fit usize");
                u32::try_from(block_end).expect("GDN cache block state version must fit u32");
            }
            for page_ids in state_blocks {
                assert_eq!(
                    page_ids.len(),
                    num_pages_per_state_slot,
                    "GDN block state page count must cover every GDN layer"
                );
                assert!(
                    page_ids
                        .iter()
                        .all(|&page_id| (page_id as usize) < self.num_cache_pages),
                    "GDN runtime supplied a page ID outside the cache-page buffer"
                );
            }
        }
    }
}

impl GDNRequestStateResources {
    fn new(device: &Device, layout: GDNStateLayout) -> Self {
        let recurrent_layer_bytes = (layout.num_state_slots as u64)
            .checked_mul(layout.recurrent_state_bytes as u64)
            .expect("GDN recurrent layer byte length must fit u64");
        let conv_layer_bytes = (layout.num_state_slots as u64)
            .checked_mul(layout.conv_state_bytes as u64)
            .expect("GDN convolution layer byte length must fit u64");
        // Kernels bind the aggregate arenas at offset zero and add these layer
        // bases with Metal `ulong`. Their layer-local element indices remain u32.
        assert_u32_element_index_domain(recurrent_layer_bytes, size_of::<f32>(), "GDN recurrent layer state");
        assert_u32_element_index_domain(conv_layer_bytes, size_of::<f32>(), "GDN convolution layer state");
        let recurrent_states_bytes = (layout.num_gdn_layers as u64)
            .checked_mul(recurrent_layer_bytes)
            .expect("GDN recurrent state arena byte length must fit u64");
        let conv_states_bytes = (layout.num_gdn_layers as u64)
            .checked_mul(conv_layer_bytes)
            .expect("GDN convolution state arena byte length must fit u64");
        let pages_per_state_slot = layout
            .recurrent_state_bytes
            .div_ceil(layout.page_bytes)
            .checked_add(layout.conv_state_bytes.div_ceil(layout.page_bytes))
            .and_then(|pages| pages.checked_mul(layout.num_gdn_layers))
            .expect("GDN all-layer pages per state slot must fit usize");
        let num_page_ids = layout
            .max_state_io_requests
            .checked_mul(pages_per_state_slot)
            .expect("GDN page-ID size overflow");
        Self {
            layout,
            recurrent_states: Buffer::new_zeroed(device, recurrent_states_bytes),
            conv_states: Buffer::new_zeroed(device, conv_states_bytes),
            page_io: GDNStatePageIO::new(device, num_page_ids, layout),
        }
    }

    pub fn layer_bindings(&self, gdn_layer_index: usize) -> GDNStateArenaBindings<'_> {
        assert!(gdn_layer_index < self.layout.num_gdn_layers);
        let recurrent_layer_bytes = self.layout.num_state_slots * self.layout.recurrent_state_bytes;
        let conv_layer_bytes = self.layout.num_state_slots * self.layout.conv_state_bytes;
        let recurrent_layer_offset_bytes = gdn_layer_index * recurrent_layer_bytes;
        let conv_layer_offset_bytes = gdn_layer_index * conv_layer_bytes;
        debug_assert!(recurrent_layer_offset_bytes + recurrent_layer_bytes <= self.recurrent_states.len_bytes());
        debug_assert!(conv_layer_offset_bytes + conv_layer_bytes <= self.conv_states.len_bytes());
        GDNStateArenaBindings {
            recurrent_states: &self.recurrent_states,
            recurrent_layer_offset_bytes: recurrent_layer_offset_bytes as u64,
            conv_states: &self.conv_states,
            conv_layer_offset_bytes: conv_layer_offset_bytes as u64,
        }
    }
}

impl GDNStatePageIO {
    fn new(device: &Device, num_page_ids: usize, layout: GDNStateLayout) -> Self {
        let config = Self::config(layout);
        Self {
            page_ids: Buffer::new_zeroed_elements(device, num_page_ids, inference_backend_metal::metal::Dtype::Uint32),
            recurrent_state_slots: Buffer::new_zeroed_elements(
                device,
                layout.max_state_io_requests,
                inference_backend_metal::metal::Dtype::Uint32,
            ),
            conv_state_slots: Buffer::new_zeroed_elements(
                device,
                layout.max_state_io_requests,
                inference_backend_metal::metal::Dtype::Uint32,
            ),
            read: backend_state_pages::Read::new(device, config),
            write: backend_state_pages::Write::new(device, config),
        }
    }

    fn prepare_restore(&self, restores: &[GDNStateRestore], num_pages_per_state_slot: usize) {
        self.recurrent_state_slots.write_typed(
            0,
            &restores
                .iter()
                .map(|restore| restore.dst_recurrent_state_slot)
                .collect::<Vec<_>>(),
        );
        self.conv_state_slots.write_typed(
            0,
            &restores
                .iter()
                .map(|restore| restore.dst_conv_state_slot)
                .collect::<Vec<_>>(),
        );
        for (state_io_request_index, restore) in restores.iter().enumerate() {
            self.write_page_ids(state_io_request_index, &restore.page_ids, num_pages_per_state_slot);
        }
    }

    fn prepare_publish(&self, publishes: &[GDNStatePublish], num_pages_per_state_slot: usize) {
        self.recurrent_state_slots.write_typed(
            0,
            &publishes
                .iter()
                .map(|publish| publish.src_recurrent_state_slot)
                .collect::<Vec<_>>(),
        );
        self.conv_state_slots.write_typed(
            0,
            &publishes
                .iter()
                .map(|publish| publish.src_conv_state_slot)
                .collect::<Vec<_>>(),
        );
        for (state_io_request_index, publish) in publishes.iter().enumerate() {
            self.write_page_ids(state_io_request_index, &publish.page_ids, num_pages_per_state_slot);
        }
    }

    fn record_restore<'a, R>(
        &'a self,
        recorder: &mut R,
        pages: &'a Buffer,
        recurrent_states: &'a Buffer,
        conv_states: &'a Buffer,
        layout: GDNStateLayout,
        restores: &[GDNStateRestore],
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.assert_page_buffer_and_ids(
            pages,
            layout.page_bytes,
            restores.iter().flat_map(|restore| &restore.page_ids),
        );
        let num_state_io_requests = restores
            .len()
            .try_into()
            .expect("GDN restore I/O request count must fit u32");
        assert!(num_state_io_requests > 0, "GDN restore recording requires I/O requests");
        recorder.record(ReplayOp::opaque(self.read.invoke(
            Self::shape(num_state_io_requests),
            backend_state_pages::ReadBuffers {
                pages,
                recurrent_states,
                conv_states,
                page_ids: &self.page_ids,
                recurrent_state_slots: &self.recurrent_state_slots,
                conv_state_slots: &self.conv_state_slots,
            },
        )));
    }

    fn record_publish<'a, R>(
        &'a self,
        recorder: &mut R,
        pages: &'a Buffer,
        recurrent_states: &'a Buffer,
        conv_states: &'a Buffer,
        layout: GDNStateLayout,
        publishes: &[GDNStatePublish],
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.assert_page_buffer_and_ids(
            pages,
            layout.page_bytes,
            publishes.iter().flat_map(|publish| &publish.page_ids),
        );
        let num_state_io_requests = publishes
            .len()
            .try_into()
            .expect("GDN publish I/O request count must fit u32");
        assert!(num_state_io_requests > 0, "GDN publish recording requires I/O requests");
        recorder.record(ReplayOp::opaque(self.write.invoke(
            Self::shape(num_state_io_requests),
            backend_state_pages::WriteBuffers {
                pages,
                recurrent_states,
                conv_states,
                page_ids: &self.page_ids,
                recurrent_state_slots: &self.recurrent_state_slots,
                conv_state_slots: &self.conv_state_slots,
            },
        )));
    }

    fn shape(num_state_io_requests: u32) -> backend_state_pages::Shape {
        backend_state_pages::Shape { num_state_io_requests }
    }

    fn config(layout: GDNStateLayout) -> backend_state_pages::Config {
        backend_state_pages::Config {
            num_gdn_layers: layout.num_gdn_layers.try_into().expect("GDN layer count must fit u32"),
            num_state_slots: layout
                .num_state_slots
                .try_into()
                .expect("GDN state slot count must fit u32"),
            recurrent_state_bytes: layout
                .recurrent_state_bytes
                .try_into()
                .expect("GDN recurrent state bytes must fit u32"),
            conv_state_bytes: layout
                .conv_state_bytes
                .try_into()
                .expect("GDN convolution state bytes must fit u32"),
            page_bytes: layout.page_bytes.try_into().expect("GDN page bytes must fit u32"),
        }
    }

    fn write_page_ids(&self, state_io_request_index: usize, page_ids: &[u32], pages_per_state_slot: usize) {
        assert_eq!(page_ids.len(), pages_per_state_slot);
        let start = state_io_request_index * pages_per_state_slot;
        debug_assert!(start + page_ids.len() <= self.page_ids.len_bytes() / size_of::<u32>());
        self.page_ids.write_typed(start, page_ids);
    }

    fn assert_page_buffer_and_ids<'a>(
        &self,
        pages: &Buffer,
        page_bytes: usize,
        page_ids: impl Iterator<Item = &'a u32>,
    ) {
        assert_eq!(
            pages.len_bytes() % page_bytes,
            0,
            "GDN page buffer must contain whole pages"
        );
        let num_cache_pages = pages.len_bytes() / page_bytes;
        assert!(
            page_ids.copied().all(|page_id| (page_id as usize) < num_cache_pages),
            "GDN runtime supplied a page ID outside the cache-page buffer"
        );
    }
}

impl GDNPrepareInput {
    /// Resolves candidate states and cache-boundary snapshots into one ordered
    /// set of materialized state versions.
    ///
    /// The transaction keeps the complete destination range separate from the
    /// shifted candidate range. A cache boundary can precede the candidate range:
    ///
    /// ```text
    /// destination states   11   12   13   14
    /// candidate range                     [14, 15)
    /// cache boundaries          ^         ^
    /// materialized              12        14
    /// row state slots       MAX slot12 MAX slot14
    /// ```
    ///
    /// The recurrent and convolution materialization arrays map the union to the
    /// exact row that produces each version. `u32::MAX` means that the row produces
    /// its normal output but does not write that state domain to its persistent arena.
    fn resolve(mut self) -> GDNPrepareOutput {
        let mut restores = Vec::new();
        let publishes = Vec::new();
        let pending_request_txns = self
            .req_slots
            .iter()
            .copied()
            .zip(self.state_txns.iter().copied())
            .map(|(req_slot, txn)| GDNStateRequestTxn { req_slot, txn })
            .collect::<Vec<_>>();
        for (req_index, &req_slot) in self.req_slots.iter().enumerate() {
            assert!(
                self.request_table.current_state_version(req_slot) <= self.token_indices[req_index],
                "GDN current state version exceeds the runtime input token index"
            );
        }

        let mut restore_targets = vec![None; self.req_slots.len()];
        let mut pending_publish_pages = vec![Vec::new(); self.req_slots.len()];
        for req_index in 0..self.req_slots.len() {
            let token_index = self.token_indices[req_index] as usize;
            let base_block_index = self.block_indices[req_index];
            for (block_offset, block_page_ids) in self.state_page_ids_by_req[req_index].iter().enumerate() {
                debug_assert_eq!(block_page_ids.len(), self.num_pages_per_state_slot);
                let block_index = base_block_index + block_offset;
                let block_end = (block_index + 1) * self.num_tokens_per_block;
                let state_version = block_end as u32;
                if state_version <= self.request_table.current_state_version(self.req_slots[req_index]) {
                    continue;
                }
                if block_end <= token_index {
                    if self.request_table.current_state_version(self.req_slots[req_index])
                        < self.token_indices[req_index]
                    {
                        restore_targets[req_index] = Some((state_version, block_page_ids.clone()));
                    }
                } else {
                    pending_publish_pages[req_index].push(GDNStatePages {
                        state_version,
                        page_ids: block_page_ids.clone(),
                    });
                }
            }
        }
        for (req_index, target) in restore_targets.into_iter().enumerate() {
            let Some((state_version, page_ids)) = target else {
                continue;
            };
            restores.push(
                self.request_table
                    .restore(self.req_slots[req_index], state_version, page_ids),
            );
        }
        for (req_index, &req_slot) in self.req_slots.iter().enumerate() {
            assert_eq!(
                self.request_table.current_state_version(req_slot),
                self.token_indices[req_index],
                "GDN current state version must match the runtime input token index"
            );
        }

        let mut materialized_versions_by_req = Vec::with_capacity(self.req_slots.len());
        for (req_index, &req_slot) in self.req_slots.iter().enumerate() {
            let txn = self.state_txns[req_index];
            debug_assert!(
                pending_publish_pages[req_index]
                    .windows(2)
                    .all(|pages| pages[0].state_version < pages[1].state_version),
                "GDN runtime publish state versions must be unique and increasing"
            );
            let mut publish_versions = merge_ordered_unique_state_versions(
                self.request_table.txn_publish_state_versions(req_slot),
                pending_publish_pages[req_index].iter().map(|pages| pages.state_version),
            )
            .take_while(|&state_version| state_version < txn.candidate_end_state_version())
            .peekable();
            let mut materialized_versions = Vec::with_capacity(self.max_materialized_states_per_req);
            for candidate_state_version in txn.candidate_state_versions() {
                while publish_versions
                    .peek()
                    .is_some_and(|&publish_state_version| publish_state_version <= candidate_state_version)
                {
                    materialized_versions.push(
                        publish_versions
                            .next()
                            .expect("GDN publish state version must remain available"),
                    );
                }
                if materialized_versions.last().copied() != Some(candidate_state_version) {
                    materialized_versions.push(candidate_state_version);
                }
            }
            assert!(
                materialized_versions.len() <= self.max_materialized_states_per_req,
                "GDN materialized states exceed per-request capacity"
            );
            debug_assert!(
                materialized_versions
                    .windows(2)
                    .all(|versions| versions[0] < versions[1]),
                "GDN materialized state versions must be unique and increasing"
            );
            drop(publish_versions);
            self.request_table.begin_txn(
                req_slot,
                &materialized_versions,
                &materialized_versions,
                take(&mut pending_publish_pages[req_index]),
            );
            materialized_versions_by_req.push(materialized_versions);
        }

        let src_recurrent_state_slots = self
            .req_slots
            .iter()
            .map(|&req_slot| self.request_table.current_recurrent_state_slot(req_slot))
            .collect::<Vec<_>>();
        let src_conv_state_slots = self
            .req_slots
            .iter()
            .map(|&req_slot| self.request_table.current_conv_state_slot(req_slot))
            .collect::<Vec<_>>();
        let num_tokens = self.cu_tokens.last().copied().unwrap_or_default() as usize;
        let mut flat_materialized_recurrent_state_slots = Vec::with_capacity(num_tokens);
        let mut flat_materialized_conv_state_slots = Vec::with_capacity(num_tokens);
        for (req_index, materialized_versions) in materialized_versions_by_req.iter().enumerate() {
            let txn = self.state_txns[req_index];
            let req_slot = self.req_slots[req_index];
            let flat_start = self.cu_tokens[req_index];
            let flat_end = self.cu_tokens[req_index + 1];
            let mut materialized_versions = materialized_versions.iter().copied().peekable();
            for (flat_index, dst_state_version) in (flat_start..flat_end).zip(txn.dst_state_versions()) {
                debug_assert_eq!(
                    flat_index - flat_start,
                    dst_state_version - txn.dst_start_state_version(),
                    "GDN flat token and destination state offsets must match"
                );
                while materialized_versions
                    .peek()
                    .is_some_and(|&materialized_state_version| materialized_state_version < dst_state_version)
                {
                    materialized_versions.next();
                }
                if materialized_versions.peek().copied() == Some(dst_state_version) {
                    materialized_versions.next();
                    flat_materialized_recurrent_state_slots.push(
                        self.request_table
                            .candidate_recurrent_state_slot(req_slot, dst_state_version),
                    );
                    flat_materialized_conv_state_slots.push(
                        self.request_table
                            .candidate_conv_state_slot(req_slot, dst_state_version),
                    );
                } else {
                    flat_materialized_recurrent_state_slots.push(u32::MAX);
                    flat_materialized_conv_state_slots.push(u32::MAX);
                }
            }
        }
        GDNPrepareOutput {
            prepared: GDNPreparedRequestState {
                src_recurrent_state_slots,
                src_conv_state_slots,
                flat_materialized_recurrent_state_slots,
                flat_materialized_conv_state_slots,
            },
            request_table: self.request_table,
            restores,
            publishes,
            pending_request_txns,
        }
    }
}

fn merge_ordered_unique_state_versions(
    left: impl Iterator<Item = u32>,
    right: impl Iterator<Item = u32>,
) -> impl Iterator<Item = u32> {
    let mut left = left.peekable();
    let mut right = right.peekable();
    std::iter::from_fn(move || {
        match (left.peek(), right.peek()) {
            (Some(&left_version), Some(&right_version)) => {
                match left_version.cmp(&right_version) {
                    Ordering::Less => left.next(),
                    Ordering::Greater => right.next(),
                    Ordering::Equal => {
                        right.next();
                        left.next()
                    },
                }
            },
            (Some(_), None) => left.next(),
            (None, Some(_)) => right.next(),
            (None, None) => None,
        }
    })
}

fn assert_u32_element_index_domain(len_bytes: u64, item_size: usize, name: &str) {
    let item_size_u64 = item_size as u64;
    assert_eq!(
        len_bytes % item_size_u64,
        0,
        "{name} buffer must contain whole elements"
    );
    let num_elements = len_bytes / item_size_u64;
    assert!(num_elements > 0, "{name} buffer must not be empty");
    assert!(
        u32::try_from(num_elements - 1).is_ok(),
        "{name} buffer exceeds the shader u32 element-index domain: num_elements={num_elements}"
    );
}

#[cfg(test)]
#[path = "state_table_test.rs"]
mod tests;
