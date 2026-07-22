use std::cell::RefCell;
use std::mem::size_of;
use std::mem::take;

use inference_backend_metal::components::GDNStatePageBatchRead;
use inference_backend_metal::components::GDNStatePageBatchReadBuffers;
use inference_backend_metal::components::GDNStatePageBatchShape;
use inference_backend_metal::components::GDNStatePageBatchWrite;
use inference_backend_metal::components::GDNStatePageBatchWriteBuffers;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::gdn::state::GDNStateTxn;
use inference_executor_core::backend::recorder::Recorder;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gdn::request_state_table::GDNRequestSlots;
use crate::attn::gdn::request_state_table::GDNStatePages;
use crate::attn::gdn::request_state_table::GDNStatePublish;
use crate::attn::gdn::request_state_table::GDNStateRestore;
use crate::def::replay_op::ReplayOp;
use crate::trace;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNStateLayout {
    num_gdn_layers: usize,
    num_state_slots: usize,
    max_state_io_requests: usize,
    page_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNStateCapacity {
    num_state_slots_per_req: usize,
    max_candidate_states_per_req: usize,
    max_publish_jobs_per_req: usize,
}

impl GDNStateCapacity {
    fn derive(max_spec_tokens: usize, max_tokens_per_request: usize, num_tokens_per_block: usize) -> Self {
        assert!(
            max_tokens_per_request > 0,
            "GDN state requires positive max tokens per request"
        );
        assert!(num_tokens_per_block > 0, "GDN state requires positive tokens per block");

        let max_speculative_candidates = max_spec_tokens
            .checked_add(1)
            .expect("GDN speculative candidate count overflow");
        let max_boundary_candidates = max_tokens_per_request
            .checked_sub(1)
            .expect("GDN max tokens per request must be positive")
            .div_ceil(num_tokens_per_block)
            .checked_add(1)
            .expect("GDN boundary candidate count overflow");
        // An aligned request ends at a block boundary and shares its final
        // candidate with that boundary. An unaligned request can have one
        // additional final candidate after every crossed boundary.
        let max_candidate_states_per_req = max_speculative_candidates.max(max_boundary_candidates);
        let num_state_slots_per_req = max_candidate_states_per_req
            .checked_add(1)
            .expect("GDN per-request state slot count overflow");
        let max_publish_jobs_per_req = max_tokens_per_request.div_ceil(num_tokens_per_block);

        // Speculative prefix candidates include every state version reached by the
        // forward, including logical cache-block boundaries, so their capacities
        // overlap rather than add.
        Self {
            num_state_slots_per_req,
            max_candidate_states_per_req,
            max_publish_jobs_per_req,
        }
    }
}

pub struct GDNRequestStateTable {
    layout: GDNStateLayout,
    num_tokens_per_block: usize,
    max_spec_tokens: usize,
    max_tokens_per_request: usize,
    max_candidate_states_per_req: usize,
    max_publish_jobs_per_req: usize,
    recurrent_states: Buffer,
    conv_states: Buffer,
    request_table: RefCell<GDNRequestSlots>,
    restores: RefCell<Vec<GDNStateRestore>>,
    publishes: RefCell<Vec<GDNStatePublish>>,
    pending_request_txns: RefCell<Vec<GDNStateRequestTxn>>,
    page_io: GDNStatePageIO,
}

pub struct GDNStatePageIO {
    page_ids: Buffer,
    state_slots: Buffer,
    read: GDNStatePageBatchRead,
    write: GDNStatePageBatchWrite,
}

pub struct GDNPreparedRequestState {
    pub src_state_slots: Vec<u32>,
    pub dst_state_slots: Vec<u32>,
    pub flat_candidate_state_slots: Vec<u32>,
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
    max_spec_tokens: usize,
    max_candidate_states_per_req: usize,
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
        max_spec_tokens: usize,
        max_tokens_per_request: usize,
        num_tokens_per_block: usize,
        page_bytes: usize,
    ) -> Self {
        assert!(!cores.is_empty(), "GDN state requires layers");
        assert!(num_tokens_per_block > 0, "GDN state requires tokens per block");
        assert!(page_bytes.is_power_of_two(), "GDN page size must be a power of two");
        assert!(
            page_bytes.is_multiple_of(size_of::<f32>() * 4),
            "GDN page size must contain an integral number of float4 values"
        );
        let capacity = GDNStateCapacity::derive(max_spec_tokens, max_tokens_per_request, num_tokens_per_block);
        let request_table = GDNRequestSlots::new(num_req_slots, capacity.num_state_slots_per_req);
        let num_state_slots = num_req_slots
            .checked_mul(capacity.num_state_slots_per_req)
            .expect("GDN state slot count overflow");

        for core in cores {
            core.validate();
        }
        let first_core = &cores[0];
        let recurrent_state_bytes_u64 = [
            first_core.num_v_heads,
            first_core.v_head_dim,
            first_core.qk_head_dim,
            size_of::<f32>(),
        ]
        .into_iter()
        .try_fold(1u64, |product, factor| {
            product.checked_mul(factor.try_into().expect("GDN dimension must fit u64"))
        })
        .expect("GDN recurrent state slot size must fit u64");
        let conv_state_bytes_u64 = [first_core.qkv_dim(), first_core.conv_state_len(), size_of::<f32>()]
            .into_iter()
            .try_fold(1u64, |product, factor| {
                product.checked_mul(factor.try_into().expect("GDN dimension must fit u64"))
            })
            .expect("GDN convolution state slot size must fit u64");
        let recurrent_state_bytes: usize = recurrent_state_bytes_u64
            .try_into()
            .expect("GDN recurrent state slot size must fit host usize");
        let conv_state_bytes: usize = conv_state_bytes_u64
            .try_into()
            .expect("GDN convolution state slot size must fit host usize");
        u32::try_from(recurrent_state_bytes_u64).expect("GDN recurrent state slot bytes must fit shader u32");
        u32::try_from(conv_state_bytes_u64).expect("GDN convolution state slot bytes must fit shader u32");
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
            page_bytes,
        };
        let recurrent_layer_bytes = u64::try_from(layout.num_state_slots)
            .expect("GDN state-slot count must fit u64")
            .checked_mul(recurrent_state_bytes_u64)
            .expect("GDN recurrent layer byte length must fit u64");
        let conv_layer_bytes = u64::try_from(layout.num_state_slots)
            .expect("GDN state-slot count must fit u64")
            .checked_mul(conv_state_bytes_u64)
            .expect("GDN convolution layer byte length must fit u64");
        // Kernels bind the aggregate arenas at offset zero and add these layer
        // bases with Metal `ulong`. Their layer-local element indices remain u32.
        assert_u32_element_index_domain(recurrent_layer_bytes, size_of::<f32>(), "GDN recurrent layer state");
        assert_u32_element_index_domain(conv_layer_bytes, size_of::<f32>(), "GDN convolution layer state");
        let num_gdn_layers_u64 = u64::try_from(layout.num_gdn_layers).expect("GDN layer count must fit u64");
        let recurrent_states_bytes = num_gdn_layers_u64
            .checked_mul(recurrent_layer_bytes)
            .expect("GDN recurrent state arena byte length must fit u64");
        let conv_states_bytes = num_gdn_layers_u64
            .checked_mul(conv_layer_bytes)
            .expect("GDN convolution state arena byte length must fit u64");
        let num_page_ids = max_state_io_requests
            .checked_mul(
                layout
                    .num_gdn_layers
                    .checked_mul(
                        recurrent_state_bytes
                            .div_ceil(layout.page_bytes)
                            .checked_add(conv_state_bytes.div_ceil(layout.page_bytes))
                            .expect("GDN per-layer state page count overflow"),
                    )
                    .expect("GDN all-layer state page count overflow"),
            )
            .expect("GDN page-ID size overflow");

        Self {
            layout,
            num_tokens_per_block,
            max_spec_tokens,
            max_tokens_per_request,
            max_candidate_states_per_req: capacity.max_candidate_states_per_req,
            max_publish_jobs_per_req: capacity.max_publish_jobs_per_req,
            recurrent_states: Buffer::new_zeroed(device, recurrent_states_bytes),
            conv_states: Buffer::new_zeroed(device, conv_states_bytes),
            request_table: RefCell::new(request_table),
            restores: RefCell::new(Vec::with_capacity(num_req_slots)),
            publishes: RefCell::new(Vec::with_capacity(layout.max_state_io_requests)),
            pending_request_txns: RefCell::new(Vec::with_capacity(num_req_slots)),
            page_io: GDNStatePageIO::new(device, num_page_ids, layout.max_state_io_requests),
        }
    }

    pub fn num_pages_per_state_slot(&self) -> usize {
        self.recurrent_state_bytes()
            .div_ceil(self.layout.page_bytes)
            .checked_add(self.conv_state_bytes().div_ceil(self.layout.page_bytes))
            .and_then(|pages| pages.checked_mul(self.layout.num_gdn_layers))
            .expect("GDN all-layer pages per state slot must fit usize")
    }

    pub fn num_req_slots(&self) -> usize {
        self.request_table.borrow().num_req_slots()
    }

    pub fn num_layers(&self) -> usize {
        self.layout.num_gdn_layers
    }

    fn recurrent_state_bytes(&self) -> usize {
        self.recurrent_states.len_bytes()
            / self
                .layout
                .num_gdn_layers
                .checked_mul(self.layout.num_state_slots)
                .expect("GDN recurrent state leading dimensions must fit usize")
    }

    fn conv_state_bytes(&self) -> usize {
        self.conv_states.len_bytes()
            / self
                .layout
                .num_gdn_layers
                .checked_mul(self.layout.num_state_slots)
                .expect("GDN convolution state leading dimensions must fit usize")
    }

    pub fn layer_bindings(&self, gdn_layer_index: usize) -> GDNStateArenaBindings<'_> {
        assert!(gdn_layer_index < self.layout.num_gdn_layers);
        let recurrent_layer_offset_bytes = u64::try_from(gdn_layer_index)
            .expect("GDN layer index must fit u64")
            .checked_mul(
                u64::try_from(self.layout.num_state_slots)
                    .expect("GDN state-slot count must fit u64")
                    .checked_mul(
                        u64::try_from(self.recurrent_state_bytes())
                            .expect("GDN recurrent state slot bytes must fit u64"),
                    )
                    .expect("GDN recurrent layer byte length must fit u64"),
            )
            .expect("GDN recurrent layer byte offset must fit u64");
        let conv_layer_offset_bytes = u64::try_from(gdn_layer_index)
            .expect("GDN layer index must fit u64")
            .checked_mul(
                u64::try_from(self.layout.num_state_slots)
                    .expect("GDN state-slot count must fit u64")
                    .checked_mul(
                        u64::try_from(self.conv_state_bytes()).expect("GDN convolution state slot bytes must fit u64"),
                    )
                    .expect("GDN convolution layer byte length must fit u64"),
            )
            .expect("GDN convolution layer byte offset must fit u64");
        GDNStateArenaBindings {
            recurrent_states: &self.recurrent_states,
            recurrent_layer_offset_bytes,
            conv_states: &self.conv_states,
            conv_layer_offset_bytes,
        }
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
            restores.len() <= self.request_table.borrow().num_req_slots(),
            "GDN restore I/O requests exceed request-slot capacity"
        );
        self.page_io.prepare_restore(&restores, self.num_pages_per_state_slot());
        true
    }

    pub fn record_restore<'a, R>(&'a self, recorder: &mut R, pages: &'a Buffer)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let restores = self.restores.borrow();
        self.page_io.record_restore(
            recorder,
            pages,
            &self.recurrent_states,
            &self.conv_states,
            self.layout,
            &restores,
        );
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
        self.page_io
            .prepare_publish(&publishes, self.num_pages_per_state_slot());
        self.page_io.record_publish(
            recorder,
            pages,
            &self.recurrent_states,
            &self.conv_states,
            self.layout,
            &publishes,
        );
        true
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
        *self.request_table.borrow_mut() = output.request_table;
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
            max_spec_tokens: self.max_spec_tokens,
            max_candidate_states_per_req: self.max_candidate_states_per_req,
            request_table: self.request_table.borrow().clone(),
            req_slots: req_slots.to_vec(),
            block_indices: block_indices.to_vec(),
            token_indices: token_indices.to_vec(),
            cu_tokens: cu_tokens.to_vec(),
            state_txns: state_txns.to_vec(),
            state_page_ids_by_req: state_page_ids_by_req.to_vec(),
            num_pages_per_state_slot: self.num_pages_per_state_slot(),
        }
    }

    pub fn commit(&self, verified_state_versions: &[u32]) {
        let pending_request_txns = self.pending_request_txns.borrow();
        let mut publishes_out = self.publishes.borrow_mut();
        let mut request_table = self.request_table.borrow_mut();
        assert_eq!(pending_request_txns.len(), verified_state_versions.len());
        publishes_out.clear();
        for (request_txn, &state_version) in pending_request_txns.iter().zip(verified_state_versions) {
            if state_version != request_table.current_state_version(request_txn.req_slot) {
                assert!(
                    request_txn.txn.contains_candidate_state_version(state_version),
                    "GDN state version must select a recorded candidate"
                );
            }
            let publishes = request_table.commit_txn(request_txn.req_slot, state_version);
            assert!(
                publishes.len() <= self.max_publish_jobs_per_req,
                "GDN publishes exceed scheduler-derived per-request capacity"
            );
            publishes_out.extend(publishes);
        }
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        let mut request_table = self.request_table.borrow_mut();
        request_table.reset_req_slots(req_slots);
        for &req_slot in req_slots {
            self.zero_state_slot(request_table.current_state_slot(req_slot));
        }
    }

    pub fn reset_req_slot(&self, req_slot: RawRequestSlot) {
        self.reset_req_slots(&[req_slot]);
    }

    fn zero_state_slot(&self, state_slot: u32) {
        let state_slot_index = usize::try_from(state_slot).expect("GDN state slot must fit host usize");
        assert!(state_slot_index < self.layout.num_state_slots);
        for gdn_layer_index in 0..self.layout.num_gdn_layers {
            let layer = self.layer_bindings(gdn_layer_index);
            let recurrent_state_slot_offset_bytes = layer
                .recurrent_layer_offset_bytes
                .checked_add(
                    u64::try_from(state_slot_index)
                        .expect("GDN state slot must fit u64")
                        .checked_mul(
                            u64::try_from(self.recurrent_state_bytes())
                                .expect("GDN recurrent state slot bytes must fit u64"),
                        )
                        .expect("GDN recurrent state slot byte offset must fit u64"),
                )
                .expect("GDN recurrent arena byte offset must fit u64")
                .try_into()
                .expect("GDN recurrent arena byte offset must fit host usize");
            self.recurrent_states
                .zero_bytes(recurrent_state_slot_offset_bytes, self.recurrent_state_bytes());
            let conv_state_slot_offset_bytes = layer
                .conv_layer_offset_bytes
                .checked_add(
                    u64::try_from(state_slot_index)
                        .expect("GDN state slot must fit u64")
                        .checked_mul(
                            u64::try_from(self.conv_state_bytes())
                                .expect("GDN convolution state slot bytes must fit u64"),
                        )
                        .expect("GDN convolution state slot byte offset must fit u64"),
                )
                .expect("GDN convolution arena byte offset must fit u64")
                .try_into()
                .expect("GDN convolution arena byte offset must fit host usize");
            self.conv_states
                .zero_bytes(conv_state_slot_offset_bytes, self.conv_state_bytes());
        }
        trace::gdn_state(|| format!("event=gdn_state_zero slot={state_slot}"));
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
        assert!(
            req_slots.len() <= self.request_table.borrow().num_req_slots(),
            "GDN request count exceeds state-table capacity"
        );
        for req_index in 0..req_slots.len() {
            let txn = state_txns[req_index];
            let num_tokens = cu_tokens[req_index + 1]
                .checked_sub(cu_tokens[req_index])
                .expect("GDN state batch cu_tokens must be nondecreasing");
            assert!(num_tokens > 0, "GDN state batch requires tokens for every request");
            assert_eq!(txn.token_index, token_indices[req_index]);
            assert_eq!(txn.num_total_tokens, num_tokens);
            assert!(
                txn.num_total_tokens as usize <= self.max_tokens_per_request,
                "GDN request tokens exceed scheduler-derived per-request capacity"
            );
            assert!(
                txn.num_spec_tokens as usize <= self.max_spec_tokens,
                "GDN speculative suffix exceeds candidate state capacity"
            );
            if txn.num_spec_tokens > 0 {
                assert!(
                    txn.num_total_tokens as usize
                        <= self
                            .max_spec_tokens
                            .checked_add(1)
                            .expect("GDN speculative candidate capacity must fit usize"),
                    "GDN candidate states require decode-sized token batches"
                );
            }
        }
    }
}

impl GDNStatePageIO {
    fn new(device: &Device, num_page_ids: usize, max_state_io_requests: usize) -> Self {
        Self {
            page_ids: Buffer::new_zeroed_elements(device, num_page_ids, inference_backend_metal::metal::Dtype::Uint32),
            state_slots: Buffer::new_zeroed_elements(
                device,
                max_state_io_requests,
                inference_backend_metal::metal::Dtype::Uint32,
            ),
            read: GDNStatePageBatchRead::new(device),
            write: GDNStatePageBatchWrite::new(device),
        }
    }

    fn prepare_restore(&self, restores: &[GDNStateRestore], num_pages_per_state_slot: usize) {
        self.state_slots.write_typed(
            0,
            &restores
                .iter()
                .map(|restore| restore.dst_state_slot)
                .collect::<Vec<_>>(),
        );
        for (state_io_request_index, restore) in restores.iter().enumerate() {
            self.write_page_ids(state_io_request_index, &restore.page_ids, num_pages_per_state_slot);
        }
    }

    fn prepare_publish(&self, publishes: &[GDNStatePublish], num_pages_per_state_slot: usize) {
        self.state_slots.write_typed(
            0,
            &publishes
                .iter()
                .map(|publish| publish.src_state_slot)
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
            Self::shape(layout, num_state_io_requests),
            GDNStatePageBatchReadBuffers {
                pages,
                recurrent_states,
                conv_states,
                page_ids: &self.page_ids,
                state_slots: &self.state_slots,
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
            Self::shape(layout, num_state_io_requests),
            GDNStatePageBatchWriteBuffers {
                pages,
                recurrent_states,
                conv_states,
                page_ids: &self.page_ids,
                state_slots: &self.state_slots,
            },
        )));
    }

    fn shape(layout: GDNStateLayout, num_state_io_requests: u32) -> GDNStatePageBatchShape {
        GDNStatePageBatchShape {
            num_gdn_layers: layout.num_gdn_layers.try_into().expect("GDN layer count must fit u32"),
            num_state_io_requests,
            num_state_slots: layout
                .num_state_slots
                .try_into()
                .expect("GDN state slot count must fit u32"),
            page_bytes: layout.page_bytes.try_into().expect("GDN page bytes must fit u32"),
        }
    }

    fn write_page_ids(&self, state_io_request_index: usize, page_ids: &[u32], pages_per_state_slot: usize) {
        assert_eq!(page_ids.len(), pages_per_state_slot);
        let start = state_io_request_index
            .checked_mul(pages_per_state_slot)
            .expect("GDN page-ID staging offset must fit usize");
        let end = start
            .checked_add(page_ids.len())
            .expect("GDN page-ID staging end must fit usize");
        assert!(
            end <= self.page_ids.len_bytes() / size_of::<u32>(),
            "GDN page-ID staging exceeds capacity"
        );
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
        let mut txn_publish_pages = vec![Vec::new(); self.req_slots.len()];
        for req_index in 0..self.req_slots.len() {
            let token_index = self.token_indices[req_index] as usize;
            let base_block_index = self.block_indices[req_index];
            for (block_offset, block_page_ids) in self.state_page_ids_by_req[req_index].iter().enumerate() {
                assert_eq!(
                    block_page_ids.len(),
                    self.num_pages_per_state_slot,
                    "GDN block state page count must cover every GDN layer"
                );
                let block_index = base_block_index
                    .checked_add(block_offset)
                    .expect("GDN cache block index must fit usize");
                let block_end = block_index
                    .checked_add(1)
                    .and_then(|block_count| block_count.checked_mul(self.num_tokens_per_block))
                    .expect("GDN cache block end must fit usize");
                let state_version = block_end.try_into().expect("GDN state version must fit u32");
                if state_version <= self.request_table.current_state_version(self.req_slots[req_index]) {
                    continue;
                }
                if block_end <= token_index {
                    if self.request_table.current_state_version(self.req_slots[req_index])
                        < token_index.try_into().expect("GDN token index must fit u32")
                    {
                        restore_targets[req_index] = Some((state_version, block_page_ids.clone()));
                    }
                } else {
                    txn_publish_pages[req_index].push(GDNStatePages {
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

        let mut candidate_versions_by_req = Vec::with_capacity(self.req_slots.len());
        for (req_index, &req_slot) in self.req_slots.iter().enumerate() {
            let txn = self.state_txns[req_index];
            let mut candidate_versions = Vec::with_capacity(
                self.max_spec_tokens
                    .checked_add(1)
                    .expect("GDN candidate-version capacity must fit usize"),
            );
            candidate_versions.push(txn.last_candidate_state_version());
            candidate_versions.extend(
                self.request_table
                    .txn_publish_state_versions(req_slot)
                    .filter(|&state_version| state_version <= txn.last_candidate_state_version()),
            );
            if txn.num_spec_tokens > 0 {
                for num_tokens_since_txn_start in 1..=txn.num_total_tokens {
                    candidate_versions.push(
                        txn.token_index
                            .checked_add(num_tokens_since_txn_start)
                            .expect("GDN candidate state version must fit u32"),
                    );
                }
            }
            for pages in &txn_publish_pages[req_index] {
                if pages.state_version <= txn.last_candidate_state_version() {
                    candidate_versions.push(pages.state_version);
                }
            }
            candidate_versions.sort_unstable();
            candidate_versions.dedup();
            assert!(
                candidate_versions.len() <= self.max_candidate_states_per_req,
                "GDN candidate states exceed scheduler-derived per-request capacity"
            );
            self.request_table
                .begin_txn(req_slot, &candidate_versions, take(&mut txn_publish_pages[req_index]));
            candidate_versions_by_req.push(candidate_versions);
        }

        let src_state_slots = self
            .req_slots
            .iter()
            .map(|&req_slot| self.request_table.current_state_slot(req_slot))
            .collect::<Vec<_>>();
        let dst_state_slots = self
            .req_slots
            .iter()
            .enumerate()
            .map(|(req_index, &req_slot)| {
                self.request_table
                    .candidate_state_slot(req_slot, self.state_txns[req_index].last_candidate_state_version())
            })
            .collect::<Vec<_>>();
        let num_tokens = self.cu_tokens.last().copied().unwrap_or_default() as usize;
        let mut flat_candidate_state_slots = Vec::with_capacity(num_tokens);
        for (req_index, candidate_versions) in candidate_versions_by_req.iter().enumerate() {
            let txn = self.state_txns[req_index];
            let req_slot = self.req_slots[req_index];
            let flat_start = self.cu_tokens[req_index];
            let flat_end = self.cu_tokens[req_index + 1];
            for flat_index in flat_start..flat_end {
                let state_version = txn
                    .token_index
                    .checked_add(flat_index - flat_start + 1)
                    .expect("GDN candidate state version must fit u32");
                flat_candidate_state_slots.push(if candidate_versions.contains(&state_version) {
                    self.request_table.candidate_state_slot(req_slot, state_version)
                } else {
                    u32::MAX
                });
            }
        }
        GDNPrepareOutput {
            prepared: GDNPreparedRequestState {
                src_state_slots,
                dst_state_slots,
                flat_candidate_state_slots,
            },
            request_table: self.request_table,
            restores,
            publishes,
            pending_request_txns,
        }
    }
}

fn assert_u32_element_index_domain(len_bytes: u64, item_size: usize, name: &str) {
    let item_size_u64 = u64::try_from(item_size).expect("dtype item size must fit u64");
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
mod tests {
    use std::mem::size_of;

    use inference_backend_metal::metal::Device;
    use inference_executor_core::attn::GDNCore;
    use inference_executor_core::attn::GDNReplayShape;
    use inference_executor_core::attn::gdn::state::GDNStateTxn;

    use super::GDNRequestStateTable;
    use super::GDNStateCapacity;
    use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;

    const TEST_PAGE_BYTES: usize = 32 * 1024;

    fn normal_forward_capacity(token_index: usize, num_tokens: usize, num_tokens_per_block: usize) -> (usize, usize) {
        let final_state_version = token_index
            .checked_add(num_tokens)
            .expect("test GDN state version overflow");
        let num_publish_jobs = final_state_version / num_tokens_per_block - token_index / num_tokens_per_block;
        let num_candidate_states =
            num_publish_jobs + usize::from(!final_state_version.is_multiple_of(num_tokens_per_block));
        (num_candidate_states, num_publish_jobs)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_state(
        state: &GDNRequestStateTable,
        metadata: &GDNMetadataBuffers,
        req_slots: &[u32],
        block_indices: &[usize],
        token_indices: &[u32],
        cu_tokens: &[u32],
        state_txns: &[GDNStateTxn],
        state_page_ids_by_req: &[Vec<Vec<u32>>],
    ) -> GDNReplayShape {
        let prepared = state.prepare(
            req_slots,
            block_indices,
            token_indices,
            cu_tokens,
            state_txns,
            state_page_ids_by_req,
        );
        metadata.update(
            cu_tokens,
            &prepared.src_state_slots,
            &prepared.dst_state_slots,
            &prepared.flat_candidate_state_slots,
        )
    }

    #[test]
    fn test_capacity_for_aligned_and_unaligned_cache_block_starts() {
        let capacity = GDNStateCapacity::derive(0, 2048, 1024);

        assert_eq!(normal_forward_capacity(0, 2048, 1024), (2, 2));
        assert_eq!(normal_forward_capacity(1, 2048, 1024), (3, 2));

        // A 2-block aligned request shares its final candidate with the last
        // boundary; starting one token later adds a distinct final candidate.
        assert_eq!(capacity.max_candidate_states_per_req, 3);
        assert_eq!(capacity.num_state_slots_per_req, 4);
        assert_eq!(capacity.max_publish_jobs_per_req, 2);
    }

    #[test]
    fn test_capacity_prefers_speculative_prefix_candidates() {
        let capacity = GDNStateCapacity::derive(4, 1024, 1024);

        assert_eq!(capacity.max_candidate_states_per_req, 5);
        assert_eq!(capacity.num_state_slots_per_req, 6);
        assert_eq!(capacity.max_publish_jobs_per_req, 1);
    }

    #[test]
    fn test_layout() {
        let device = Device::system_default();
        let state = GDNRequestStateTable::new(&device, &[core(0), core(1)], 2, 2, 16, 16, TEST_PAGE_BYTES);

        assert_eq!(state.layer_bindings(0).recurrent_layer_offset_bytes, 0);
        assert_eq!(state.layer_bindings(0).conv_layer_offset_bytes, 0);
        assert_eq!(
            state.layer_bindings(1).recurrent_layer_offset_bytes,
            8 * 4 * size_of::<f32>() as u64
        );
        assert_eq!(
            state.layer_bindings(1).conv_layer_offset_bytes,
            8 * 12 * size_of::<f32>() as u64
        );
        assert_eq!(
            state.layer_bindings(0).recurrent_states.as_raw_ptr(),
            state.layer_bindings(1).recurrent_states.as_raw_ptr()
        );
        assert_eq!(
            state.layer_bindings(0).conv_states.as_raw_ptr(),
            state.layer_bindings(1).conv_states.as_raw_ptr()
        );
        assert_eq!(state.num_pages_per_state_slot(), 4);
    }

    #[test]
    #[should_panic(expected = "runtime supplied a page ID outside the cache-page buffer")]
    fn test_page_id_domain_panics() {
        let device = Device::system_default();
        let state = GDNRequestStateTable::new(&device, &[core(0)], 1, 1, 2, 2, 16);
        let pages = inference_backend_metal::metal::Buffer::new_zeroed(&device, 2 * 16);
        let page_ids = [2_u32];

        state
            .page_io
            .assert_page_buffer_and_ids(&pages, state.layout.page_bytes, page_ids.iter());
    }

    #[test]
    fn test_transaction() {
        let device = Device::system_default();
        let state = GDNRequestStateTable::new(&device, &[core(0), core(1)], 2, 2, 16, 16, TEST_PAGE_BYTES);
        let batch_metadata = GDNMetadataBuffers::new(&device, 2, 8);
        prepare_state(
            &state,
            &batch_metadata,
            &[0],
            &[0],
            &[0],
            &[0, 1],
            &[GDNStateTxn::new(0, 1, 0)],
            &[Vec::new()],
        );

        assert_eq!(batch_metadata.src_state_slots().read_typed::<u32>(0, 1), vec![0]);
        assert_eq!(batch_metadata.dst_state_slots().read_typed::<u32>(0, 1), vec![2]);
        assert_eq!(
            batch_metadata.flat_candidate_state_slots().read_typed::<u32>(0, 1),
            vec![2]
        );
        state.commit(&[1]);
        assert_eq!(state.request_table.borrow().current_state_slot(0), 2);
        assert_eq!(state.request_table.borrow().current_state_version(0), 1);
    }

    #[test]
    fn test_future_publish_page_ids() {
        let device = Device::system_default();
        let state = GDNRequestStateTable::new(&device, &[core(0), core(1)], 1, 3, 6, 2, 16);
        let batch_metadata = GDNMetadataBuffers::new(&device, 1, 8);
        prepare_state(
            &state,
            &batch_metadata,
            &[0],
            &[0],
            &[0],
            &[0, 1],
            &[GDNStateTxn::new(0, 1, 0)],
            &[vec![
                vec![10, 11, 12, 13, 14, 15, 16, 17],
                vec![20, 21, 22, 23, 24, 25, 26, 27],
            ]],
        );
        state.commit(&[1]);

        prepare_state(
            &state,
            &batch_metadata,
            &[0],
            &[0],
            &[1],
            &[0, 3],
            &[GDNStateTxn::new(1, 3, 0)],
            &[Vec::new()],
        );
        let state_version_2_slot = state.request_table.borrow().candidate_state_slot(0, 2);
        let state_version_4_slot = state.request_table.borrow().candidate_state_slot(0, 4);
        assert_eq!(
            batch_metadata.flat_candidate_state_slots().read_typed::<u32>(0, 3),
            vec![state_version_2_slot, u32::MAX, state_version_4_slot]
        );

        state.commit(&[4]);
        assert_eq!(
            state
                .publishes()
                .iter()
                .map(|publish| (publish.state_version, publish.page_ids.clone()))
                .collect::<Vec<_>>(),
            vec![
                (2, vec![10, 11, 12, 13, 14, 15, 16, 17]),
                (4, vec![20, 21, 22, 23, 24, 25, 26, 27]),
            ]
        );
    }

    #[test]
    fn test_snapshot_restore_then_apply_tokens() {
        let device = Device::system_default();
        let state = GDNRequestStateTable::new(&device, &[core(0), core(1)], 1, 2, 2, 2, 16);
        let batch_metadata = GDNMetadataBuffers::new(&device, 1, 8);
        let snapshot_page_ids = vec![10, 11, 12, 13, 14, 15, 16, 17];
        prepare_state(
            &state,
            &batch_metadata,
            &[0],
            &[0],
            &[2],
            &[0, 1],
            &[GDNStateTxn::new(2, 1, 0)],
            &[vec![snapshot_page_ids.clone()]],
        );

        assert_eq!(state.restores().len(), 1);
        assert_eq!(state.restores()[0].state_version, 2);
        assert_eq!(state.restores()[0].page_ids, snapshot_page_ids);
        assert_eq!(batch_metadata.src_state_slots().read_typed::<u32>(0, 1), vec![0]);

        state.commit(&[3]);
        assert_eq!(state.request_table.borrow().current_state_version(0), 3);
    }

    #[test]
    fn test_snapshot_restore_at_a_1024_token_logical_cache_boundary() {
        let device = Device::system_default();
        let state = GDNRequestStateTable::new(&device, &[core(0), core(1)], 1, 2, 1024, 1024, 16);
        let batch_metadata = GDNMetadataBuffers::new(&device, 1, 8);
        let snapshot_page_ids = vec![10, 11, 12, 13, 14, 15, 16, 17];
        prepare_state(
            &state,
            &batch_metadata,
            &[0],
            &[0],
            &[1024],
            &[0, 1],
            &[GDNStateTxn::new(1024, 1, 0)],
            &[vec![snapshot_page_ids.clone()]],
        );

        assert_eq!(state.restores().len(), 1);
        assert_eq!(state.restores()[0].state_version, 1024);
        assert_eq!(state.restores()[0].page_ids, snapshot_page_ids);
        state.commit(&[1025]);
        assert_eq!(state.request_table.borrow().current_state_version(0), 1025);
    }

    fn core(model_layer_index: usize) -> GDNCore {
        GDNCore {
            model_layer_index,
            hidden_dim: 4,
            num_qk_heads: 1,
            qk_head_dim: 2,
            num_v_heads: 1,
            v_head_dim: 2,
            conv_kernel_size: 3,
            q_scale: 1.0,
        }
    }
}
