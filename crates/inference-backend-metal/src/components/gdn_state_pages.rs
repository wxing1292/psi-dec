use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Kernel;
use crate::metal::Operator;

const GDN_STATE_PAGE_WRITE_SOURCE: &str = include_str!("metal/gdn_state_page_write.metal");
const GDN_STATE_PAGE_READ_SOURCE: &str = include_str!("metal/gdn_state_page_read.metal");

const NUM_THREADS_PER_THREADBLOCK: usize = 256;

/// Static batch geometry for GDN state-page I/O.
///
/// Each state-I/O request selects one logical state version, one recurrent physical
/// slot, one convolution physical slot, and its page IDs across every GDN layer.
/// One `GDNStatePageReadTask` or `GDNStatePageWriteTask` maps 1:1 to one
/// threadblock and has the complete logical coordinates
/// `{ state_io_request_index, gdn_layer_index, state_kind, page_index_in_state }`.
/// The grid and shape values derive every coordinate, so no Task value,
/// TaskTemplate, or ABI buffer is materialized. `page_id`,
/// `recurrent_state_slot`, and `conv_state_slot` remain data inputs rather than
/// Task coordinates. `state_kind` selects the applicable physical slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNStatePageBatchConfig {
    pub num_gdn_layers: u32,
    pub num_state_slots: u32,
    pub recurrent_state_bytes: u32,
    pub conv_state_bytes: u32,
    pub page_bytes: u32,
}

impl GDNStatePageBatchConfig {
    pub fn validate(self) {
        assert!(self.num_gdn_layers > 0);
        assert!(self.num_state_slots > 0);
        assert!(self.recurrent_state_bytes > 0);
        assert_eq!(self.recurrent_state_bytes % (4 * size_of::<f32>() as u32), 0);
        assert!(self.conv_state_bytes > 0);
        assert_eq!(self.conv_state_bytes % (4 * size_of::<f32>() as u32), 0);
        assert!(self.page_bytes > 0);
        assert_eq!(self.page_bytes % (4 * size_of::<f32>() as u32), 0);
    }

    pub fn validate_shape(self, shape: GDNStatePageBatchShape) {
        self.validate();
        shape.validate();
        self.num_total_pages(shape);
    }

    pub fn state_slots_bytes(self, shape: GDNStatePageBatchShape) -> usize {
        checked_product(
            "GDN state-slot metadata byte length",
            &[shape.num_state_io_requests as usize, size_of::<u32>()],
        )
    }

    fn state_arena_bytes(self, state_bytes: u32) -> usize {
        checked_product(
            "GDN state arena byte length",
            &[
                self.num_gdn_layers as usize,
                self.num_state_slots as usize,
                state_bytes as usize,
            ],
        )
    }

    fn num_total_pages(self, shape: GDNStatePageBatchShape) -> usize {
        let pages_per_layer = self
            .recurrent_state_bytes
            .div_ceil(self.page_bytes)
            .checked_add(self.conv_state_bytes.div_ceil(self.page_bytes))
            .expect("GDN pages per layer must fit u32");
        let num_pages = checked_product(
            "GDN state-page batch count",
            &[
                self.num_gdn_layers as usize,
                shape.num_state_io_requests as usize,
                pages_per_layer as usize,
            ],
        );
        assert_u32_count_domain(num_pages, "GDN state-page batch pages");
        num_pages
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNStatePageBatchShape {
    pub num_state_io_requests: u32,
}

impl GDNStatePageBatchShape {
    pub fn validate(self) {
        assert!(self.num_state_io_requests > 0);
    }
}

#[derive(Clone, Copy)]
pub struct GDNStatePageBatchWriteBuffers<'a> {
    pub pages: &'a Buffer,
    pub recurrent_states: &'a Buffer,
    pub conv_states: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub recurrent_state_slots: &'a Buffer,
    pub conv_state_slots: &'a Buffer,
}

pub struct GDNStatePageBatchWrite {
    config: GDNStatePageBatchConfig,
    kernel: Kernel,
}

impl GDNStatePageBatchWrite {
    pub fn new(device: &Device, config: GDNStatePageBatchConfig) -> Self {
        config.validate();
        Self {
            config,
            kernel: Kernel::new(device, GDN_STATE_PAGE_WRITE_SOURCE, "gdn_state_page_batch_write_f32"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GDNStatePageBatchShape,
        buffers: GDNStatePageBatchWriteBuffers<'a>,
    ) -> GDNStatePageBatchWriteInvocation<'a> {
        GDNStatePageBatchWriteInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }
}

pub struct GDNStatePageBatchWriteInvocation<'a> {
    kernel: &'a GDNStatePageBatchWrite,
    shape: GDNStatePageBatchShape,
    buffers: GDNStatePageBatchWriteBuffers<'a>,
}

impl Operator for GDNStatePageBatchWriteInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        let config = self.kernel.config;
        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_write(0, self.buffers.pages, 0);
        builder.set_buffer_read(1, self.buffers.recurrent_states, 0);
        builder.set_buffer_read(2, self.buffers.conv_states, 0);
        builder.set_buffer_read(3, self.buffers.page_ids, 0);
        builder.set_buffer_read(4, self.buffers.recurrent_state_slots, 0);
        builder.set_buffer_read(5, self.buffers.conv_state_slots, 0);
        builder.set_u32(6, config.num_gdn_layers);
        builder.set_u32(7, config.num_state_slots);
        builder.set_u32(8, self.shape.num_state_io_requests);
        builder.set_u32(9, config.recurrent_state_bytes.div_ceil(config.page_bytes));
        builder.set_u32(10, config.recurrent_state_bytes);
        builder.set_u32(11, config.conv_state_bytes.div_ceil(config.page_bytes));
        builder.set_u32(12, config.conv_state_bytes);
        builder.set_u32(13, config.page_bytes);
        builder.dispatch_threadblocks(
            (config.num_total_pages(self.shape), 1, 1),
            (NUM_THREADS_PER_THREADBLOCK, 1, 1),
        );
    }
}

impl GDNStatePageBatchWriteInvocation<'_> {
    fn validate(&self) {
        let config = self.kernel.config;
        config.validate_shape(self.shape);
        assert!(self.buffers.page_ids.len_bytes() >= config.num_total_pages(self.shape) * size_of::<u32>());
        assert!(self.buffers.recurrent_states.len_bytes() >= config.state_arena_bytes(config.recurrent_state_bytes));
        assert!(self.buffers.conv_states.len_bytes() >= config.state_arena_bytes(config.conv_state_bytes));
        assert!(self.buffers.recurrent_state_slots.len_bytes() >= config.state_slots_bytes(self.shape));
        assert!(self.buffers.conv_state_slots.len_bytes() >= config.state_slots_bytes(self.shape));
    }
}

#[derive(Clone, Copy)]
pub struct GDNStatePageBatchReadBuffers<'a> {
    pub pages: &'a Buffer,
    pub recurrent_states: &'a Buffer,
    pub conv_states: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub recurrent_state_slots: &'a Buffer,
    pub conv_state_slots: &'a Buffer,
}

pub struct GDNStatePageBatchRead {
    config: GDNStatePageBatchConfig,
    kernel: Kernel,
}

impl GDNStatePageBatchRead {
    pub fn new(device: &Device, config: GDNStatePageBatchConfig) -> Self {
        config.validate();
        Self {
            config,
            kernel: Kernel::new(device, GDN_STATE_PAGE_READ_SOURCE, "gdn_state_page_batch_read_f32"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GDNStatePageBatchShape,
        buffers: GDNStatePageBatchReadBuffers<'a>,
    ) -> GDNStatePageBatchReadInvocation<'a> {
        GDNStatePageBatchReadInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }
}

pub struct GDNStatePageBatchReadInvocation<'a> {
    kernel: &'a GDNStatePageBatchRead,
    shape: GDNStatePageBatchShape,
    buffers: GDNStatePageBatchReadBuffers<'a>,
}

impl Operator for GDNStatePageBatchReadInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        let config = self.kernel.config;
        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.buffers.pages, 0);
        builder.set_buffer_write(1, self.buffers.recurrent_states, 0);
        builder.set_buffer_write(2, self.buffers.conv_states, 0);
        builder.set_buffer_read(3, self.buffers.page_ids, 0);
        builder.set_buffer_read(4, self.buffers.recurrent_state_slots, 0);
        builder.set_buffer_read(5, self.buffers.conv_state_slots, 0);
        builder.set_u32(6, config.num_gdn_layers);
        builder.set_u32(7, config.num_state_slots);
        builder.set_u32(8, self.shape.num_state_io_requests);
        builder.set_u32(9, config.recurrent_state_bytes.div_ceil(config.page_bytes));
        builder.set_u32(10, config.recurrent_state_bytes);
        builder.set_u32(11, config.conv_state_bytes.div_ceil(config.page_bytes));
        builder.set_u32(12, config.conv_state_bytes);
        builder.set_u32(13, config.page_bytes);
        builder.dispatch_threadblocks(
            (config.num_total_pages(self.shape), 1, 1),
            (NUM_THREADS_PER_THREADBLOCK, 1, 1),
        );
    }
}

impl GDNStatePageBatchReadInvocation<'_> {
    fn validate(&self) {
        let config = self.kernel.config;
        config.validate_shape(self.shape);
        assert!(self.buffers.page_ids.len_bytes() >= config.num_total_pages(self.shape) * size_of::<u32>());
        assert!(self.buffers.recurrent_states.len_bytes() >= config.state_arena_bytes(config.recurrent_state_bytes));
        assert!(self.buffers.conv_states.len_bytes() >= config.state_arena_bytes(config.conv_state_bytes));
        assert!(self.buffers.recurrent_state_slots.len_bytes() >= config.state_slots_bytes(self.shape));
        assert!(self.buffers.conv_state_slots.len_bytes() >= config.state_slots_bytes(self.shape));
    }
}

#[cfg(test)]
mod tests {
    use super::GDNStatePageBatchConfig;
    use super::GDNStatePageBatchRead;
    use super::GDNStatePageBatchReadBuffers;
    use super::GDNStatePageBatchShape;
    use super::GDNStatePageBatchWrite;
    use super::GDNStatePageBatchWriteBuffers;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Stream;

    #[test]
    #[should_panic(expected = "GDN state-page batch pages exceeds the shader u32 count domain")]
    fn test_batch_shape_rejects_shader_count_overflow() {
        GDNStatePageBatchConfig {
            num_gdn_layers: 1 << 30,
            num_state_slots: 1,
            recurrent_state_bytes: 16,
            conv_state_bytes: 48,
            page_bytes: 16,
        }
        .num_total_pages(GDNStatePageBatchShape {
            num_state_io_requests: 1,
        });
    }

    #[test]
    fn test_batch_read() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = GDNStatePageBatchConfig {
            num_gdn_layers: 1,
            num_state_slots: 2,
            recurrent_state_bytes: 12 * size_of::<f32>() as u32,
            conv_state_bytes: 4 * size_of::<f32>() as u32,
            page_bytes: 32,
        };
        let page_read = GDNStatePageBatchRead::new(&device, config);
        let pages = Buffer::from_slice(
            &device,
            &[
                0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, //
                10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, //
                20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, //
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, //
                40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, //
                50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, //
                60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0,
            ],
        );
        let recurrent_states = Buffer::new_zeroed(&device, 2 * 12 * size_of::<f32>());
        let conv_states = Buffer::new_zeroed(&device, 2 * 4 * size_of::<f32>());
        let page_ids = Buffer::from_slice(&device, &[1_u32, 3, 5, 2, 4, 6]);
        let recurrent_state_slots = Buffer::from_slice(&device, &[1_u32, 0]);
        let conv_state_slots = Buffer::from_slice(&device, &[0_u32, 1]);
        let shape = GDNStatePageBatchShape {
            num_state_io_requests: 2,
        };

        let mut builder = stream.create_replay_program();
        builder.record(page_read.invoke(
            shape,
            GDNStatePageBatchReadBuffers {
                pages: &pages,
                recurrent_states: &recurrent_states,
                conv_states: &conv_states,
                page_ids: &page_ids,
                recurrent_state_slots: &recurrent_state_slots,
                conv_state_slots: &conv_state_slots,
            },
        ));
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(
            recurrent_states.read_typed::<f32>(0, 24),
            vec![
                20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 40.0, 41.0, 42.0, 43.0, //
                10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 30.0, 31.0, 32.0, 33.0,
            ]
        );
        assert_eq!(
            conv_states.read_typed::<f32>(0, 8),
            vec![
                50.0, 51.0, 52.0, 53.0, //
                60.0, 61.0, 62.0, 63.0,
            ]
        );
    }

    #[test]
    fn test_batch_write() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = GDNStatePageBatchConfig {
            num_gdn_layers: 1,
            num_state_slots: 2,
            recurrent_state_bytes: 12 * size_of::<f32>() as u32,
            conv_state_bytes: 4 * size_of::<f32>() as u32,
            page_bytes: 32,
        };
        let page_write = GDNStatePageBatchWrite::new(&device, config);
        let pages = Buffer::from_slice(&device, &[-1.0_f32; 7 * 8]);
        let recurrent_states = Buffer::from_slice(
            &device,
            &[
                100.0_f32, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, //
                200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0, 211.0,
            ],
        );
        let conv_states = Buffer::from_slice(
            &device,
            &[
                300.0_f32, 301.0, 302.0, 303.0, //
                400.0, 401.0, 402.0, 403.0,
            ],
        );
        let page_ids = Buffer::from_slice(&device, &[1_u32, 3, 5, 2, 4, 6]);
        let recurrent_state_slots = Buffer::from_slice(&device, &[1_u32, 0]);
        let conv_state_slots = Buffer::from_slice(&device, &[0_u32, 1]);
        let shape = GDNStatePageBatchShape {
            num_state_io_requests: 2,
        };

        let mut builder = stream.create_replay_program();
        builder.record(page_write.invoke(
            shape,
            GDNStatePageBatchWriteBuffers {
                pages: &pages,
                recurrent_states: &recurrent_states,
                conv_states: &conv_states,
                page_ids: &page_ids,
                recurrent_state_slots: &recurrent_state_slots,
                conv_state_slots: &conv_state_slots,
            },
        ));
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(
            pages.read_typed::<f32>(0, 56),
            vec![
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, //
                200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, //
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, //
                208.0, 209.0, 210.0, 211.0, 0.0, 0.0, 0.0, 0.0, //
                108.0, 109.0, 110.0, 111.0, 0.0, 0.0, 0.0, 0.0, //
                300.0, 301.0, 302.0, 303.0, 0.0, 0.0, 0.0, 0.0, //
                400.0, 401.0, 402.0, 403.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
    }

    #[test]
    fn test_batch_layers() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = GDNStatePageBatchConfig {
            num_gdn_layers: 2,
            num_state_slots: 3,
            recurrent_state_bytes: 8 * size_of::<f32>() as u32,
            conv_state_bytes: 4 * size_of::<f32>() as u32,
            page_bytes: 32,
        };
        let page_read = GDNStatePageBatchRead::new(&device, config);
        let page_write = GDNStatePageBatchWrite::new(&device, config);
        let shape = GDNStatePageBatchShape {
            num_state_io_requests: 2,
        };
        let recurrent_values = (0..48).map(|value| value as f32 + 10.0).collect::<Vec<_>>();
        let conv_values = (0..24).map(|value| value as f32 + 100.0).collect::<Vec<_>>();
        let recurrent_source = Buffer::from_slice(&device, &recurrent_values);
        let conv_source = Buffer::from_slice(&device, &conv_values);
        let recurrent_target = Buffer::new_zeroed(&device, recurrent_values.len() * size_of::<f32>());
        let conv_target = Buffer::new_zeroed(&device, conv_values.len() * size_of::<f32>());
        let pages = Buffer::new_zeroed(&device, 9 * 8 * size_of::<f32>());
        let page_ids = Buffer::from_slice(&device, &[1_u32, 3, 5, 7, 2, 4, 6, 8]);
        let recurrent_state_slots = Buffer::from_slice(&device, &[2_u32, 0]);
        let conv_state_slots = Buffer::from_slice(&device, &[1_u32, 0]);

        let mut write = stream.create_replay_program();
        write.record(page_write.invoke(
            shape,
            GDNStatePageBatchWriteBuffers {
                pages: &pages,
                recurrent_states: &recurrent_source,
                conv_states: &conv_source,
                page_ids: &page_ids,
                recurrent_state_slots: &recurrent_state_slots,
                conv_state_slots: &conv_state_slots,
            },
        ));
        stream.submit_replay(&write.build()).wait();

        let mut read = stream.create_replay_program();
        read.record(page_read.invoke(
            shape,
            GDNStatePageBatchReadBuffers {
                pages: &pages,
                recurrent_states: &recurrent_target,
                conv_states: &conv_target,
                page_ids: &page_ids,
                recurrent_state_slots: &recurrent_state_slots,
                conv_state_slots: &conv_state_slots,
            },
        ));
        stream.submit_replay(&read.build()).wait();

        for layer in 0..2 {
            for state_slot in [0, 2] {
                let recurrent_start = (layer * 3 + state_slot) * 8;
                assert_eq!(
                    recurrent_target.read_typed::<f32>(recurrent_start, 8),
                    recurrent_values[recurrent_start..recurrent_start + 8]
                );
            }
            for state_slot in [0, 1] {
                let conv_start = (layer * 3 + state_slot) * 4;
                assert_eq!(
                    conv_target.read_typed::<f32>(conv_start, 4),
                    conv_values[conv_start..conv_start + 4]
                );
            }
        }
    }
}
