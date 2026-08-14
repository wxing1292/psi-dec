use std::cell::RefCell;
use std::ops::Range;

use inference_executor_core::def::ModelExecutorError;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gdn::request_slots::GDNRequestSlots;
use crate::attn::gdn::state_table::GDNRequestStateTable;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GDNStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

impl FullStateIO for GDNRequestStateTable {
    type Files = GDNStateSnapshotFiles;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError> {
        self.assert_snapshot_ready();
        let request_table = self.request_table().borrow();
        request_table.assert_snapshot_ready(
            self.max_publish_jobs_per_req,
            self.num_pages_per_state_slot(),
            self.num_cache_pages,
        );
        writer.write_metadata(files.request_state_table(), &*request_table)?;
        writer.write_full_buffer(files.recurrent_state(), &self.resources().recurrent_states)?;
        writer.write_full_buffer(files.conv_state(), &self.resources().conv_states)?;
        Ok(())
    }

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError> {
        let request_table: GDNRequestSlots = reader.read_metadata(files.request_state_table())?;
        reader.read_full_buffer(files.recurrent_state(), &self.resources().recurrent_states)?;
        reader.read_full_buffer(files.conv_state(), &self.resources().conv_states)?;
        self.request_table = Some(RefCell::new(request_table));
        Ok(())
    }
}

impl SelectedStateIO for GDNRequestStateTable {
    type ID = RawRequestSlot;

    fn write_selected_state(
        &self,
        writer: &mut StateSnapshotWriter,
        files: Self::Files,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        self.assert_snapshot_ready();
        let request_table = self.request_table().borrow();
        request_table.assert_snapshot_ready(
            self.max_publish_jobs_per_req,
            self.num_pages_per_state_slot(),
            self.num_cache_pages,
        );
        let recurrent_state_entry_ranges = selected_state_entry_ranges(
            &request_table,
            request_slot_ranges,
            self.layout,
            GDNRequestSlots::current_recurrent_state_slot,
        );
        let conv_state_entry_ranges = selected_state_entry_ranges(
            &request_table,
            request_slot_ranges,
            self.layout,
            GDNRequestSlots::current_conv_state_slot,
        );
        writer.write_metadata(files.request_state_table(), &*request_table)?;
        writer.write_selected_buffer(
            files.recurrent_state(),
            &self.resources().recurrent_states,
            &recurrent_state_entry_ranges,
            self.recurrent_state_bytes(),
        )?;
        writer.write_selected_buffer(
            files.conv_state(),
            &self.resources().conv_states,
            &conv_state_entry_ranges,
            self.conv_state_bytes(),
        )?;
        Ok(())
    }

    fn read_selected_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        let request_table: GDNRequestSlots = reader.read_metadata(files.request_state_table())?;
        let recurrent_state_entry_ranges = selected_state_entry_ranges(
            &request_table,
            request_slot_ranges,
            self.layout,
            GDNRequestSlots::current_recurrent_state_slot,
        );
        let conv_state_entry_ranges = selected_state_entry_ranges(
            &request_table,
            request_slot_ranges,
            self.layout,
            GDNRequestSlots::current_conv_state_slot,
        );
        reader.read_selected_buffer(
            files.recurrent_state(),
            &self.resources().recurrent_states,
            &recurrent_state_entry_ranges,
            self.recurrent_state_bytes(),
        )?;
        reader.read_selected_buffer(
            files.conv_state(),
            &self.resources().conv_states,
            &conv_state_entry_ranges,
            self.conv_state_bytes(),
        )?;
        self.request_table = Some(RefCell::new(request_table));
        Ok(())
    }
}

fn selected_state_entry_ranges(
    request_table: &GDNRequestSlots,
    request_slot_ranges: &[Range<RawRequestSlot>],
    layout: super::GDNStateLayout,
    current_state_slot: fn(&GDNRequestSlots, RawRequestSlot) -> u32,
) -> Vec<Range<u32>> {
    let selected_request_count = request_slot_ranges
        .iter()
        .map(|range| (range.end - range.start) as usize)
        .sum::<usize>();
    let mut state_slots = request_slot_ranges
        .iter()
        .flat_map(|range| range.clone())
        .map(|req_slot| current_state_slot(request_table, req_slot) as usize)
        .collect::<Vec<_>>();
    state_slots.sort_unstable();
    state_slots.dedup();
    assert_eq!(
        state_slots.len(),
        selected_request_count,
        "selected GDN requests must own distinct current state slots"
    );

    let mut entry_ranges: Vec<Range<u32>> = Vec::new();
    for gdn_layer_index in 0..layout.num_gdn_layers {
        for &state_slot in &state_slots {
            let entry_index = gdn_layer_index
                .checked_mul(layout.num_state_slots)
                .and_then(|index| index.checked_add(state_slot))
                .expect("GDN selected state entry index must fit usize");
            let entry_index = u32::try_from(entry_index).expect("GDN selected state entry index must fit u32");
            let entry_end = entry_index
                .checked_add(1)
                .expect("GDN selected state entry end must fit u32");
            if let Some(last) = entry_ranges.last_mut()
                && last.end == entry_index
            {
                last.end = entry_end;
            } else {
                entry_ranges.push(entry_index..entry_end);
            }
        }
    }
    entry_ranges
}
