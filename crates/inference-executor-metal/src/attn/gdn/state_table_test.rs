use std::mem::size_of;
use std::path::PathBuf;

use half::bf16;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::attn::gdn::state::GDNStateTxn;
use inference_runtime_core::compute::ExecutorHibernationPlan;

use super::GDNRequestStateTable;
use super::GDNStateCapacity;
use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::request_slots::GDNRequestSlots;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GDNStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotFile;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

const TEST_PAGE_BYTES: usize = 32 * 1024;
const LIFECYCLE_PAGE_BYTES: usize = 16;
const TEST_NUM_CACHE_PAGES: usize = 1024;
const SNAPSHOT_FILES: GDNStateSnapshotFiles = GDNStateSnapshotFiles::new(
    StateSnapshotFile::MainGDNRequestStateTable,
    StateSnapshotFile::MainGDNRecurrentState,
    StateSnapshotFile::MainGDNConvState,
);

#[derive(Debug, PartialEq)]
struct GDNStateReference {
    request_state: GDNRequestSlots,
    recurrent_state: Vec<u16>,
    conv_state: Vec<u16>,
}

#[test]
fn test_full_state_unload_load_fixed() {
    let device = Device::system_default();
    let mut state = new_lifecycle_state(&device);
    let pages_per_state = state.num_pages_per_state_slot();
    let page_ids = (0..2 * pages_per_state)
        .map(|index| u32::try_from(10 + index).unwrap())
        .collect::<Vec<_>>();
    populate_durable_request_state(&state, &device, &page_ids);
    advance_to_distinct_state_slots(&state, 1);

    let (num_recurrent_values, num_conv_values) = state_value_counts(&state);
    let recurrent_state = fixed_values(num_recurrent_values, 0.25);
    let conv_state = fixed_values(num_conv_values, -0.5);
    write_state_values(&state, &recurrent_state, &conv_state);
    let reference = capture_state(&state);

    assert_eq!(reference.recurrent_state, recurrent_state);
    assert_eq!(reference.conv_state, conv_state);
    assert_unload_load("fixed", &device, &mut state, reference);
}

#[test]
fn test_full_state_unload_load_random() {
    let device = Device::system_default();
    let mut state = new_lifecycle_state(&device);
    let mut random = TestRandom::new(0x4744_4e5f_5354_4154);
    let page_ids = (0..2 * state.num_pages_per_state_slot())
        .map(|_| random.next_u32() % 1024)
        .collect::<Vec<_>>();
    populate_durable_request_state(&state, &device, &page_ids);
    advance_to_distinct_state_slots(&state, 1);

    let (num_recurrent_values, num_conv_values) = state_value_counts(&state);
    let recurrent_state = random.values(num_recurrent_values);
    let conv_state = random.values(num_conv_values);
    write_state_values(&state, &recurrent_state, &conv_state);
    let reference = capture_state(&state);

    assert_unload_load("random", &device, &mut state, reference);
}

#[test]
fn test_selected_state_unload_load() {
    let device = Device::system_default();
    let mut state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        3,
        GDNStateCapacity::new(3, 2, 1),
        2,
        TEST_NUM_CACHE_PAGES,
        LIFECYCLE_PAGE_BYTES,
    );
    advance_to_distinct_state_slots_for_req(&state, 1, 0);
    let selected_recurrent_state_slot =
        usize::try_from(state.request_table().borrow().current_recurrent_state_slot(1)).unwrap();
    let selected_conv_state_slot = usize::try_from(state.request_table().borrow().current_conv_state_slot(1)).unwrap();
    assert_ne!(selected_recurrent_state_slot, selected_conv_state_slot);
    let request_state = state.request_table().borrow().clone();
    let (num_recurrent_values, num_conv_values) = state_value_counts(&state);
    let recurrent_state = fixed_values(num_recurrent_values, 0.25);
    let conv_state = fixed_values(num_conv_values, -0.5);
    write_state_values(&state, &recurrent_state, &conv_state);

    let selected_request_slot_ranges = std::iter::once(1..2).collect::<Vec<_>>();
    let plan = ExecutorHibernationPlan::selected(selected_request_slot_ranges.to_vec(), Vec::new());
    let snapshot_path = snapshot_path("selected");
    let buffer_io = inference_backend_metal::metal::BufferIO::new(&device);
    let snapshot_files = [
        SNAPSHOT_FILES.request_state_table(),
        SNAPSHOT_FILES.recurrent_state(),
        SNAPSHOT_FILES.conv_state(),
    ];
    let mut writer = StateSnapshotWriter::new(&snapshot_path, &snapshot_files, &plan, &buffer_io).unwrap();
    state
        .write_selected_state(&mut writer, SNAPSHOT_FILES, &selected_request_slot_ranges)
        .unwrap();
    writer.commit().unwrap();

    state.release_resources();
    state.allocate_resources(&device);
    let mut reader = StateSnapshotReader::open(&snapshot_path, &snapshot_files, &plan, &buffer_io).unwrap();
    state
        .read_selected_state(&mut reader, SNAPSHOT_FILES, &selected_request_slot_ranges)
        .unwrap();
    reader.finish().unwrap();

    let restored = capture_state(&state);
    assert_eq!(restored.request_state, request_state);
    assert_selected_state_values(
        &restored.recurrent_state,
        &recurrent_state,
        state.layout.num_gdn_layers,
        state.layout.num_state_slots,
        selected_recurrent_state_slot,
        state.recurrent_state_bytes() / size_of::<u16>(),
    );
    assert_selected_state_values(
        &restored.conv_state,
        &conv_state,
        state.layout.num_gdn_layers,
        state.layout.num_state_slots,
        selected_conv_state_slot,
        state.conv_state_bytes() / size_of::<u16>(),
    );
    std::fs::remove_dir_all(snapshot_path).unwrap();
}

#[test]
#[should_panic(expected = "GDN state slots must include current state and all materialized states")]
fn test_capacity_requires_current_state_slot() {
    let _ = GDNStateCapacity::new(3, 3, 1);
}

#[test]
fn test_layout() {
    let device = Device::system_default();
    let state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        2,
        GDNStateCapacity::new(4, 3, 1),
        16,
        TEST_NUM_CACHE_PAGES,
        TEST_PAGE_BYTES,
    );

    assert_eq!(state.layer_bindings(0).recurrent_layer_offset_bytes, 0);
    assert_eq!(state.layer_bindings(0).conv_layer_offset_bytes, 0);
    assert_eq!(
        state.layer_bindings(1).recurrent_layer_offset_bytes,
        8 * 16 * size_of::<u16>() as u64
    );
    assert_eq!(
        state.layer_bindings(1).conv_layer_offset_bytes,
        8 * 24 * size_of::<u16>() as u64
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
    let state = GDNRequestStateTable::new(&device, &[core(0)], 1, GDNStateCapacity::new(3, 2, 1), 2, 2, 16);
    let pages = inference_backend_metal::metal::Buffer::new_zeroed(&device, 2 * 16);
    let page_ids = [2_u32];

    state
        .resources()
        .page_io
        .assert_page_buffer_and_ids(&pages, state.layout.page_bytes, page_ids.iter());
}

#[test]
fn test_transaction_lifecycle_handles_mixed_commit_modes_and_deferred_publish() {
    let device = Device::system_default();
    let state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        3,
        GDNStateCapacity::new(5, 4, 2),
        2,
        TEST_NUM_CACHE_PAGES,
        LIFECYCLE_PAGE_BYTES,
    );
    let batch_metadata = GDNMetadataBuffers::new(&device, 3, 12);
    let pages_per_state = u32::try_from(state.num_pages_per_state_slot()).unwrap();
    let pages_at_version_2 = (10..10 + pages_per_state).collect::<Vec<_>>();
    let pages_at_version_4 = (20..20 + pages_per_state).collect::<Vec<_>>();
    let pages_at_version_6 = (30..30 + pages_per_state).collect::<Vec<_>>();
    prepare_state(
        &state,
        &batch_metadata,
        &[0, 1, 2],
        &[0, 0, 0],
        &[0, 0, 0],
        &[0, 4, 8, 12],
        &[
            GDNStateTxn::from_state_versions(1, 5, 1, 0),
            GDNStateTxn::from_state_versions(1, 5, 3, 1),
            GDNStateTxn::from_state_versions(1, 5, 4, 1),
        ],
        &[
            vec![
                pages_at_version_2.clone(),
                pages_at_version_4.clone(),
                pages_at_version_6.clone(),
            ],
            Vec::new(),
            Vec::new(),
        ],
    );

    let table = state.request_table().borrow();
    let req_0_recurrent_state_2 = table.candidate_recurrent_state_slot(0, 2);
    let req_0_recurrent_state_4 = table.candidate_recurrent_state_slot(0, 4);
    let req_0_conv_state_2 = table.candidate_conv_state_slot(0, 2);
    let req_0_conv_state_4 = table.candidate_conv_state_slot(0, 4);
    let req_1_recurrent_state_1 = table.candidate_recurrent_state_slot(1, 1);
    let req_1_recurrent_state_2 = table.candidate_recurrent_state_slot(1, 2);
    let req_1_recurrent_state_3 = table.candidate_recurrent_state_slot(1, 3);
    let req_1_conv_state_1 = table.candidate_conv_state_slot(1, 1);
    let req_1_conv_state_2 = table.candidate_conv_state_slot(1, 2);
    let req_1_conv_state_3 = table.candidate_conv_state_slot(1, 3);
    let req_2_current_recurrent = table.current_recurrent_state_slot(2);
    let req_2_current_conv = table.current_conv_state_slot(2);
    let req_2_recurrent_state_1 = table.candidate_recurrent_state_slot(2, 1);
    let req_2_recurrent_state_2 = table.candidate_recurrent_state_slot(2, 2);
    let req_2_recurrent_state_3 = table.candidate_recurrent_state_slot(2, 3);
    let req_2_conv_state_1 = table.candidate_conv_state_slot(2, 1);
    let req_2_conv_state_2 = table.candidate_conv_state_slot(2, 2);
    let req_2_conv_state_3 = table.candidate_conv_state_slot(2, 3);
    drop(table);
    assert_eq!(
        batch_metadata
            .flat_materialized_recurrent_state_slots()
            .read_typed::<u32>(0, 12),
        vec![
            u32::MAX,
            req_0_recurrent_state_2,
            u32::MAX,
            req_0_recurrent_state_4,
            req_1_recurrent_state_1,
            req_1_recurrent_state_2,
            req_1_recurrent_state_3,
            u32::MAX,
            req_2_recurrent_state_1,
            req_2_recurrent_state_2,
            req_2_recurrent_state_3,
            u32::MAX,
        ]
    );
    assert_eq!(
        batch_metadata
            .flat_materialized_conv_state_slots()
            .read_typed::<u32>(0, 12),
        vec![
            u32::MAX,
            req_0_conv_state_2,
            u32::MAX,
            req_0_conv_state_4,
            req_1_conv_state_1,
            req_1_conv_state_2,
            req_1_conv_state_3,
            u32::MAX,
            req_2_conv_state_1,
            req_2_conv_state_2,
            req_2_conv_state_3,
            u32::MAX,
        ]
    );

    state.commit(&[4, 3, 1]);
    let table = state.request_table().borrow();
    assert_eq!(table.current_state_version(0), 4);
    assert_eq!(table.current_recurrent_state_slot(0), req_0_recurrent_state_4);
    assert_eq!(table.current_conv_state_slot(0), req_0_conv_state_4);
    assert_eq!(table.current_state_version(1), 2);
    assert_eq!(table.current_recurrent_state_slot(1), req_1_recurrent_state_2);
    assert_eq!(table.current_conv_state_slot(1), req_1_conv_state_2);
    assert_eq!(table.current_state_version(2), 0);
    assert_eq!(table.current_recurrent_state_slot(2), req_2_current_recurrent);
    assert_eq!(table.current_conv_state_slot(2), req_2_current_conv);
    drop(table);
    assert_eq!(
        state
            .publishes()
            .iter()
            .map(|publish| {
                (
                    publish.req_slot,
                    publish.src_recurrent_state_slot,
                    publish.src_conv_state_slot,
                    publish.state_version,
                    publish.page_ids.clone(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (0, req_0_recurrent_state_2, req_0_conv_state_2, 2, pages_at_version_2),
            (0, req_0_recurrent_state_4, req_0_conv_state_4, 4, pages_at_version_4),
        ]
    );

    prepare_state(
        &state,
        &batch_metadata,
        &[0],
        &[2],
        &[4],
        &[0, 2],
        &[GDNStateTxn::new(4, 2, 1)],
        &[Vec::new()],
    );
    let (req_0_recurrent_state_6, req_0_conv_state_6) = {
        let table = state.request_table().borrow();
        (
            table.candidate_recurrent_state_slot(0, 6),
            table.candidate_conv_state_slot(0, 6),
        )
    };
    state.commit(&[6]);
    assert_eq!(state.request_table().borrow().current_state_version(0), 6);
    assert_eq!(
        state
            .publishes()
            .iter()
            .map(|publish| {
                (
                    publish.req_slot,
                    publish.src_recurrent_state_slot,
                    publish.src_conv_state_slot,
                    publish.state_version,
                    publish.page_ids.clone(),
                )
            })
            .collect::<Vec<_>>(),
        vec![(0, req_0_recurrent_state_6, req_0_conv_state_6, 6, pages_at_version_6)]
    );
}

#[test]
fn test_restore_and_reset_preserve_neighbor_request_state() {
    let device = Device::system_default();
    let state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        2,
        GDNStateCapacity::new(4, 3, 1),
        1024,
        TEST_NUM_CACHE_PAGES,
        LIFECYCLE_PAGE_BYTES,
    );
    let batch_metadata = GDNMetadataBuffers::new(&device, 1, 1);
    let snapshot_page_ids = (10..10 + u32::try_from(state.num_pages_per_state_slot()).unwrap()).collect::<Vec<_>>();
    advance_to_distinct_state_slots_for_req(&state, 0, 0);
    advance_to_distinct_state_slots_for_req(&state, 1, 0);
    let (
        restore_recurrent_state_slot,
        restore_conv_state_slot,
        neighbor_recurrent_state_slot,
        neighbor_conv_state_slot,
        neighbor_state_version,
    ) = {
        let table = state.request_table().borrow();
        (
            table.current_recurrent_state_slot(0),
            table.current_conv_state_slot(0),
            table.current_recurrent_state_slot(1),
            table.current_conv_state_slot(1),
            table.current_state_version(1),
        )
    };
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
    assert_eq!(
        state.restores()[0].dst_recurrent_state_slot,
        restore_recurrent_state_slot
    );
    assert_eq!(state.restores()[0].dst_conv_state_slot, restore_conv_state_slot);
    assert_eq!(
        batch_metadata.src_recurrent_state_slots().read_typed::<u32>(0, 1),
        [restore_recurrent_state_slot]
    );
    assert_eq!(
        batch_metadata.src_conv_state_slots().read_typed::<u32>(0, 1),
        [restore_conv_state_slot]
    );
    state.finish_restore();
    state.commit(&[1025]);

    let (num_recurrent_values, num_conv_values) = state_value_counts(&state);
    let recurrent_state = fixed_values(num_recurrent_values, 1.0);
    let conv_state = fixed_values(num_conv_values, -1.0);
    write_state_values(&state, &recurrent_state, &conv_state);
    state.reset_req_slot(0);

    let table = state.request_table().borrow();
    let reset_recurrent_state_slot = table.current_recurrent_state_slot(0) as usize;
    let reset_conv_state_slot = table.current_conv_state_slot(0) as usize;
    assert_eq!(table.current_state_version(0), 0);
    assert_eq!(table.current_recurrent_state_slot(1), neighbor_recurrent_state_slot);
    assert_eq!(table.current_conv_state_slot(1), neighbor_conv_state_slot);
    assert_eq!(table.current_state_version(1), neighbor_state_version);
    drop(table);

    assert_reset_state_values(
        &state,
        &recurrent_state,
        &conv_state,
        reset_recurrent_state_slot,
        reset_conv_state_slot,
    );
}

#[test]
#[should_panic(expected = "GDN current state version must match the runtime input token index")]
fn test_prepare_requires_source_state_for_first_input_token() {
    let device = Device::system_default();
    let state = GDNRequestStateTable::new(
        &device,
        &[core(0), core(1)],
        1,
        GDNStateCapacity::new(4, 3, 1),
        2,
        TEST_NUM_CACHE_PAGES,
        16,
    );
    let batch_metadata = GDNMetadataBuffers::new(&device, 1, 1);

    prepare_state(
        &state,
        &batch_metadata,
        &[0],
        &[0],
        &[2],
        &[0, 1],
        &[GDNStateTxn::new(2, 1, 0)],
        &[Vec::new()],
    );
}

fn assert_unload_load(name: &str, device: &Device, state: &mut GDNRequestStateTable, reference: GDNStateReference) {
    let snapshot_path = snapshot_path(name);
    let buffer_io = inference_backend_metal::metal::BufferIO::new(device);
    let snapshot_files = [
        SNAPSHOT_FILES.request_state_table(),
        SNAPSHOT_FILES.recurrent_state(),
        SNAPSHOT_FILES.conv_state(),
    ];
    let mut writer = StateSnapshotWriter::new(
        &snapshot_path,
        &snapshot_files,
        &ExecutorHibernationPlan::All,
        &buffer_io,
    )
    .unwrap();
    state.write_full_state(&mut writer, SNAPSHOT_FILES).unwrap();
    writer.commit().unwrap();

    state.release_resources();
    state.allocate_resources(device);

    let mut reader = StateSnapshotReader::open(
        &snapshot_path,
        &snapshot_files,
        &ExecutorHibernationPlan::All,
        &buffer_io,
    )
    .unwrap();
    state.read_full_state(&mut reader, SNAPSHOT_FILES).unwrap();
    reader.finish().unwrap();

    assert_eq!(capture_state(state), reference);
    std::fs::remove_dir_all(snapshot_path).unwrap();
}

fn new_lifecycle_state(device: &Device) -> GDNRequestStateTable {
    GDNRequestStateTable::new(
        device,
        &[core(0), core(1)],
        1,
        GDNStateCapacity::new(5, 4, 3),
        2,
        TEST_NUM_CACHE_PAGES,
        LIFECYCLE_PAGE_BYTES,
    )
}

fn populate_durable_request_state(state: &GDNRequestStateTable, device: &Device, page_ids: &[u32]) {
    let pages_per_state = state.num_pages_per_state_slot();
    assert_eq!(page_ids.len(), 2 * pages_per_state);
    let metadata = GDNMetadataBuffers::new(device, 1, 1);
    prepare_state(
        state,
        &metadata,
        &[0],
        &[0],
        &[0],
        &[0, 1],
        &[GDNStateTxn::new(0, 1, 0)],
        &[page_ids.chunks_exact(pages_per_state).map(<[u32]>::to_vec).collect()],
    );
    state.commit(&[1]);
}

fn advance_to_distinct_state_slots(state: &GDNRequestStateTable, current_state_version: u32) {
    advance_to_distinct_state_slots_for_req(state, 0, current_state_version);
}

fn advance_to_distinct_state_slots_for_req(state: &GDNRequestStateTable, req_slot: u32, current_state_version: u32) {
    let first_state_version = current_state_version + 1;
    let second_state_version = first_state_version + 1;
    let mut request_table = state.request_table().borrow_mut();
    request_table.begin_txn(
        req_slot,
        &[first_state_version],
        &[first_state_version, second_state_version],
        Vec::new(),
    );
    let _ = request_table.commit_txn(req_slot, first_state_version);
    request_table.begin_txn(req_slot, &[second_state_version], &[second_state_version], Vec::new());
    let _ = request_table.commit_txn(req_slot, second_state_version);
    assert_ne!(
        request_table.current_recurrent_state_slot(req_slot),
        request_table.current_conv_state_slot(req_slot)
    );
}

fn write_state_values(state: &GDNRequestStateTable, recurrent_state: &[u16], conv_state: &[u16]) {
    let resources = state.resources();
    assert_eq!(
        std::mem::size_of_val(recurrent_state),
        resources.recurrent_states.len_bytes()
    );
    assert_eq!(std::mem::size_of_val(conv_state), resources.conv_states.len_bytes());
    resources.recurrent_states.write_typed(0, recurrent_state);
    resources.conv_states.write_typed(0, conv_state);
}

fn capture_state(state: &GDNRequestStateTable) -> GDNStateReference {
    let resources = state.resources();
    let (num_recurrent_values, num_conv_values) = state_value_counts(state);
    GDNStateReference {
        request_state: state.request_table().borrow().clone(),
        recurrent_state: resources.recurrent_states.read_typed(0, num_recurrent_values),
        conv_state: resources.conv_states.read_typed(0, num_conv_values),
    }
}

fn state_value_counts(state: &GDNRequestStateTable) -> (usize, usize) {
    let resources = state.resources();
    (
        resources.recurrent_states.len_bytes() / size_of::<u16>(),
        resources.conv_states.len_bytes() / size_of::<u16>(),
    )
}

fn fixed_values(len: usize, offset: f32) -> Vec<u16> {
    (0..len)
        .map(|index| bf16::from_f32(((index % 17) as f32 - 8.0) * 0.25 + offset).to_bits())
        .collect()
}

fn assert_selected_state_values(
    restored: &[u16],
    source: &[u16],
    num_layers: usize,
    num_state_slots: usize,
    selected_state_slot: usize,
    values_per_state: usize,
) {
    let mut expected = vec![0_u16; source.len()];
    for layer_index in 0..num_layers {
        let start = (layer_index * num_state_slots + selected_state_slot) * values_per_state;
        let end = start + values_per_state;
        expected[start..end].copy_from_slice(&source[start..end]);
    }
    assert_eq!(restored, expected);
}

fn assert_reset_state_values(
    state: &GDNRequestStateTable,
    recurrent_before_reset: &[u16],
    conv_before_reset: &[u16],
    reset_recurrent_state_slot: usize,
    reset_conv_state_slot: usize,
) {
    let mut expected_recurrent = recurrent_before_reset.to_vec();
    let mut expected_conv = conv_before_reset.to_vec();
    let recurrent_values_per_state = state.recurrent_state_bytes() / size_of::<u16>();
    let conv_values_per_state = state.conv_state_bytes() / size_of::<u16>();
    for layer_index in 0..state.layout.num_gdn_layers {
        let recurrent_start =
            (layer_index * state.layout.num_state_slots + reset_recurrent_state_slot) * recurrent_values_per_state;
        expected_recurrent[recurrent_start..recurrent_start + recurrent_values_per_state].fill(0);
        let conv_start = (layer_index * state.layout.num_state_slots + reset_conv_state_slot) * conv_values_per_state;
        expected_conv[conv_start..conv_start + conv_values_per_state].fill(0);
    }
    let restored = capture_state(state);
    assert_eq!(restored.recurrent_state, expected_recurrent);
    assert_eq!(restored.conv_state, expected_conv);
}

fn snapshot_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "psi-dec-gdn-state-unload-load-{}-{name}.state",
        std::process::id()
    ))
}

struct TestRandom(u64);

impl TestRandom {
    fn new(seed: u64) -> Self {
        assert_ne!(seed, 0);
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }

    fn values(&mut self, len: usize) -> Vec<u16> {
        (0..len)
            .map(|_| {
                let value = (self.next_u32() % 20_001) as i32 - 10_000;
                bf16::from_f32(value as f32 / 128.0).to_bits()
            })
            .collect()
    }
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
        &prepared.src_recurrent_state_slots,
        &prepared.src_conv_state_slots,
        &prepared.flat_materialized_recurrent_state_slots,
        &prepared.flat_materialized_conv_state_slots,
        prepared.src_recurrent_state_slots.len() as u32,
        cu_tokens.last().copied().unwrap(),
    )
}

fn core(model_layer_index: usize) -> GDNCore {
    GDNCore {
        model_layer_index,
        hidden_dim: 4,
        num_qk_heads: 1,
        qk_head_dim: 4,
        num_v_heads: 1,
        v_head_dim: 4,
        conv_kernel_size: 3,
        q_scale: 1.0,
    }
}
