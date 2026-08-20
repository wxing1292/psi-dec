use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Kernel;
use crate::metal::Operator;

const GDN_STATE_PAGE_WRITE_SOURCE: &str = include_str!("../metal/gdn_state_page_write.metal");
const GDN_STATE_PAGE_READ_SOURCE: &str = include_str!("../metal/gdn_state_page_read.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current() -> Self {
        Self {
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

/// Static batch geometry for GDN state-page I/O.
///
/// Each state-I/O request selects one logical state version, one recurrent physical
/// slot, one convolution physical slot, and its page IDs across every GDN layer.
/// One `ReadThreadBlockTask` or `WriteThreadBlockTask` maps 1:1 to one
/// threadblock and has the complete logical coordinates
/// `{ state_io_request_index, gdn_layer_index, state_kind, page_index_in_state }`.
/// The grid and shape values derive every coordinate, so no Task value,
/// TaskTemplate, or ABI buffer is materialized. `page_id`,
/// `recurrent_state_slot`, and `conv_state_slot` remain data inputs rather than
/// Task coordinates. `state_kind` selects the applicable physical slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_gdn_layers: u32,
    pub num_state_slots: u32,
    pub recurrent_state_bytes: u32,
    pub conv_state_bytes: u32,
    pub page_bytes: u32,
}

impl Config {
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

    pub fn validate_shape(self, shape: Shape) {
        self.validate();
        shape.validate();
        self.num_total_pages(shape);
    }

    pub fn state_slots_bytes(self, shape: Shape) -> usize {
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

    fn num_total_pages(self, shape: Shape) -> usize {
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
pub struct Shape {
    pub num_state_io_requests: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_state_io_requests > 0);
    }
}

#[derive(Clone, Copy)]
pub struct WriteBuffers<'a> {
    pub pages: &'a Buffer,
    pub recurrent_states: &'a Buffer,
    pub conv_states: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub recurrent_state_slots: &'a Buffer,
    pub conv_state_slots: &'a Buffer,
}

pub struct Write {
    config: Config,
    constants: KernelConstants,
    kernel: Kernel,
}

impl Write {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            constants: KernelConstants::current(),
            kernel: Kernel::new(device, GDN_STATE_PAGE_WRITE_SOURCE, "gdn_state_page_batch_write_f32"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: WriteBuffers<'a>) -> WriteInvocation<'a> {
        WriteInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }
}

pub struct WriteInvocation<'a> {
    kernel: &'a Write,
    shape: Shape,
    buffers: WriteBuffers<'a>,
}

impl Operator for WriteInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let config = self.kernel.config;
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_write(0, self.buffers.pages, 0);
        recorder.set_buffer_read(1, self.buffers.recurrent_states, 0);
        recorder.set_buffer_read(2, self.buffers.conv_states, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.recurrent_state_slots, 0);
        recorder.set_buffer_read(5, self.buffers.conv_state_slots, 0);
        recorder.set_u32(6, config.num_gdn_layers);
        recorder.set_u32(7, config.num_state_slots);
        recorder.set_u32(8, self.shape.num_state_io_requests);
        recorder.set_u32(9, config.recurrent_state_bytes.div_ceil(config.page_bytes));
        recorder.set_u32(10, config.recurrent_state_bytes);
        recorder.set_u32(11, config.conv_state_bytes.div_ceil(config.page_bytes));
        recorder.set_u32(12, config.conv_state_bytes);
        recorder.set_u32(13, config.page_bytes);
        recorder.dispatch_threadblocks(
            (config.num_total_pages(self.shape), 1, 1),
            (self.kernel.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl WriteInvocation<'_> {
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
pub struct ReadBuffers<'a> {
    pub pages: &'a Buffer,
    pub recurrent_states: &'a Buffer,
    pub conv_states: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub recurrent_state_slots: &'a Buffer,
    pub conv_state_slots: &'a Buffer,
}

pub struct Read {
    config: Config,
    constants: KernelConstants,
    kernel: Kernel,
}

impl Read {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            constants: KernelConstants::current(),
            kernel: Kernel::new(device, GDN_STATE_PAGE_READ_SOURCE, "gdn_state_page_batch_read_f32"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: ReadBuffers<'a>) -> ReadInvocation<'a> {
        ReadInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }
}

pub struct ReadInvocation<'a> {
    kernel: &'a Read,
    shape: Shape,
    buffers: ReadBuffers<'a>,
}

impl Operator for ReadInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let config = self.kernel.config;
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.pages, 0);
        recorder.set_buffer_write(1, self.buffers.recurrent_states, 0);
        recorder.set_buffer_write(2, self.buffers.conv_states, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.recurrent_state_slots, 0);
        recorder.set_buffer_read(5, self.buffers.conv_state_slots, 0);
        recorder.set_u32(6, config.num_gdn_layers);
        recorder.set_u32(7, config.num_state_slots);
        recorder.set_u32(8, self.shape.num_state_io_requests);
        recorder.set_u32(9, config.recurrent_state_bytes.div_ceil(config.page_bytes));
        recorder.set_u32(10, config.recurrent_state_bytes);
        recorder.set_u32(11, config.conv_state_bytes.div_ceil(config.page_bytes));
        recorder.set_u32(12, config.conv_state_bytes);
        recorder.set_u32(13, config.page_bytes);
        recorder.dispatch_threadblocks(
            (config.num_total_pages(self.shape), 1, 1),
            (self.kernel.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl ReadInvocation<'_> {
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
    use super::Config;
    use super::Read;
    use super::ReadBuffers;
    use super::Shape;
    use super::Write;
    use super::WriteBuffers;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Stream;

    #[test]
    fn test_read_and_write_share_the_state_page_constants() {
        let constants = super::KernelConstants::current();
        assert_eq!(constants.thread_block.required_threads, 256);
    }

    #[test]
    #[should_panic(expected = "GDN state-page batch pages exceeds the shader u32 count domain")]
    fn test_batch_shape_rejects_shader_count_overflow() {
        Config {
            num_gdn_layers: 1 << 30,
            num_state_slots: 1,
            recurrent_state_bytes: 16,
            conv_state_bytes: 48,
            page_bytes: 16,
        }
        .num_total_pages(Shape {
            num_state_io_requests: 1,
        });
    }

    #[test]
    fn test_multi_layer_write_read_preserves_page_layout_and_unselected_slots() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            num_gdn_layers: 2,
            num_state_slots: 3,
            recurrent_state_bytes: 8 * size_of::<f32>() as u32,
            conv_state_bytes: 4 * size_of::<f32>() as u32,
            page_bytes: 32,
        };
        let page_read = Read::new(&device, config);
        let page_write = Write::new(&device, config);
        let shape = Shape {
            num_state_io_requests: 2,
        };
        let recurrent_values = (0..48).map(|value| value as f32 + 10.0).collect::<Vec<_>>();
        let conv_values = (0..24).map(|value| value as f32 + 100.0).collect::<Vec<_>>();
        let recurrent_source = Buffer::from_slice(&device, &recurrent_values);
        let conv_source = Buffer::from_slice(&device, &conv_values);
        let state_canary = -777.0_f32;
        let page_canary = -999.0_f32;
        let recurrent_target = Buffer::from_slice(&device, &vec![state_canary; recurrent_values.len()]);
        let conv_target = Buffer::from_slice(&device, &vec![state_canary; conv_values.len()]);
        let pages = Buffer::from_slice(&device, &[page_canary; 9 * 8]);
        let page_ids = Buffer::from_slice(&device, &[1_u32, 3, 5, 7, 2, 4, 6, 8]);
        let recurrent_state_slots = Buffer::from_slice(&device, &[2_u32, 0]);
        let conv_state_slots = Buffer::from_slice(&device, &[1_u32, 0]);

        let mut write = stream.create_replay_program();
        write.record(page_write.invoke(
            shape,
            WriteBuffers {
                pages: &pages,
                recurrent_states: &recurrent_source,
                conv_states: &conv_source,
                page_ids: &page_ids,
                recurrent_state_slots: &recurrent_state_slots,
                conv_state_slots: &conv_state_slots,
            },
        ));
        stream.submit_replay(&write.build()).wait();

        let expected_pages = [
            vec![page_canary; 8],
            recurrent_values[16..24].to_vec(),
            recurrent_values[0..8].to_vec(),
            [conv_values[4..8].to_vec(), vec![0.0; 4]].concat(),
            [conv_values[0..4].to_vec(), vec![0.0; 4]].concat(),
            recurrent_values[40..48].to_vec(),
            recurrent_values[24..32].to_vec(),
            [conv_values[16..20].to_vec(), vec![0.0; 4]].concat(),
            [conv_values[12..16].to_vec(), vec![0.0; 4]].concat(),
        ]
        .concat();
        assert_eq!(pages.read_typed::<f32>(0, expected_pages.len()), expected_pages);

        let mut read = stream.create_replay_program();
        read.record(page_read.invoke(
            shape,
            ReadBuffers {
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
            for state_slot in 0..3 {
                let recurrent_start = (layer * 3 + state_slot) * 8;
                let expected = if matches!(state_slot, 0 | 2) {
                    &recurrent_values[recurrent_start..recurrent_start + 8]
                } else {
                    &[state_canary; 8]
                };
                assert_eq!(recurrent_target.read_typed::<f32>(recurrent_start, 8), expected);
            }
            for state_slot in 0..3 {
                let conv_start = (layer * 3 + state_slot) * 4;
                let expected = if matches!(state_slot, 0 | 1) {
                    &conv_values[conv_start..conv_start + 4]
                } else {
                    &[state_canary; 4]
                };
                assert_eq!(conv_target.read_typed::<f32>(conv_start, 4), expected);
            }
        }
    }
}
