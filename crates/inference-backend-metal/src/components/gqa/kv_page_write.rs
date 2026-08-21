use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/gqa_kv_page_write.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    config: Config,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current(config: Config) -> Self {
        Self {
            config,
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageTableLayout {
    pub num_req_slots: u32,
    pub num_gqa_layers: u32,
    pub num_blocks: u32,
    pub num_page_ids_per_block: u32,
}

impl PageTableLayout {
    pub fn validate(self) {
        assert!(self.num_req_slots > 0);
        assert!(self.num_blocks > 0);
        assert!(self.num_gqa_layers > 0);
        assert!(self.num_page_ids_per_block > 0);
    }

    pub fn bytes(self) -> usize {
        (self.num_req_slots as usize)
            .checked_mul(self.num_gqa_layers as usize)
            .and_then(|count| count.checked_mul(self.num_blocks as usize))
            .and_then(|count| count.checked_mul(self.num_page_ids_per_block as usize))
            .and_then(|count| count.checked_mul(size_of::<u32>()))
            .expect("GQA page-table byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_bytes: u32,
    pub dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_kv_heads > 0);
        assert!(self.num_tokens_per_page() > 0);
        assert!(self.head_dim > 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_tokens_per_page(self) -> u32 {
        // KV[2][num_kv_heads][num_tokens_per_page][head_dim], where 0 is K and 1 is V.
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(self.dtype.item_size().try_into().expect("dtype size must fit u32")))
            .expect("GQA K/V bytes per token must fit u32");
        assert!(
            self.page_bytes.is_multiple_of(kv_bytes_per_token),
            "GQA page_bytes must be divisible by the K/V bytes per token"
        );
        self.page_bytes / kv_bytes_per_token
    }

    pub fn index_bytes(self, shape: Shape) -> usize {
        (shape.num_total_token_writes as usize)
            .checked_mul(size_of::<u32>())
            .expect("GQA KV page-write index bytes must fit usize")
    }

    pub fn flat_kv_bytes(self, shape: Shape) -> usize {
        self.num_total_threads(shape)
            .checked_mul(self.dtype.item_size())
            .expect("GQA flattened K/V byte length must fit usize")
    }

    pub fn num_total_threads(self, shape: Shape) -> usize {
        checked_product(
            "GQA KV page-write thread count",
            &[
                shape.num_total_token_writes as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_token_writes: u32,
    pub page_table_layout: PageTableLayout,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_token_writes > 0);
        self.page_table_layout.validate();
        assert_u32_count_domain(config.num_total_threads(self), "GQA KV page-write threads");
    }

    pub fn page_ids_bytes(self) -> usize {
        self.page_table_layout.bytes()
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub pages: &'a Buffer,
    pub flat_k: &'a Buffer,
    pub flat_v: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub page_ids: &'a Buffer,
}

pub struct Compute {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let constants = KernelConstants::current(config);
        let source = kv_page_write_source(constants);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_kv_page_write_f32",
            Dtype::Bfloat16 => "gqa_kv_page_write_u16",
            dtype => panic!("unsupported GQA KV page write dtype {dtype:?}"),
        };
        Self {
            constants,
            kernel: CompiledKernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>, page_table_index: ReplayU32) -> Invocation<'a> {
        Invocation {
            constants: self.constants,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_token_writes: ReplayU32::Fixed(shape.num_total_token_writes),
            page_table_index,
        }
    }

    pub fn invoke_bucketed<'a>(
        &'a self,
        shape: Shape,
        buffers: Buffers<'a>,
        num_active_token_writes: ReplayU32,
        page_table_index: ReplayU32,
    ) -> Invocation<'a> {
        Invocation {
            constants: self.constants,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_token_writes,
            page_table_index,
        }
    }
}

fn kv_page_write_source(kernel_constants: KernelConstants) -> String {
    let config = kernel_constants.config;
    let source_constants = format!(
        "using namespace metal;\n\nconstant uint num_kv_heads = {}u;\nconstant uint head_dim = {}u;\nconstant uint \
         num_tokens_per_page = {}u;\nconstant uint page_bytes = {}u;",
        config.num_kv_heads,
        config.head_dim,
        config.num_tokens_per_page(),
        config.page_bytes,
    );
    SOURCE.replacen("using namespace metal;", &source_constants, 1)
}

pub struct Invocation<'a> {
    constants: KernelConstants,
    kernel: &'a CompiledKernel,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_token_writes: ReplayU32,
    page_table_index: ReplayU32,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let config = self.constants.config;
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_write(0, self.buffers.pages, 0);
        recorder.set_buffer_read(1, self.buffers.flat_k, 0);
        recorder.set_buffer_read(2, self.buffers.flat_v, 0);
        recorder.set_buffer_read(3, self.buffers.req_slots, 0);
        recorder.set_buffer_read(4, self.buffers.flat_token_indices, 0);
        recorder.set_buffer_read(5, self.buffers.page_ids, 0);
        match self.num_active_token_writes {
            ReplayU32::Fixed(num_active_token_writes) => {
                assert!(num_active_token_writes > 0);
                assert!(num_active_token_writes <= self.shape.num_total_token_writes);
                recorder.set_u32(6, num_active_token_writes);
            },
            ReplayU32::Parameter(key) => recorder.bind_u32(6, key, 1, self.shape.num_total_token_writes),
        }
        let max_page_table_index = self.shape.page_table_layout.num_gqa_layers - 1;
        match self.page_table_index {
            ReplayU32::Fixed(page_table_index) => {
                assert!(
                    page_table_index <= max_page_table_index,
                    "GQA page-table index exceeds layer count"
                );
                recorder.set_u32(7, page_table_index);
            },
            ReplayU32::Parameter(key) => recorder.bind_u32(7, key, 0, max_page_table_index),
        }
        recorder.set_u32(8, self.shape.page_table_layout.num_gqa_layers);
        recorder.set_u32(9, self.shape.page_table_layout.num_blocks);
        recorder.set_u32(10, self.shape.page_table_layout.num_page_ids_per_block);
        recorder.dispatch_1d(
            config.num_total_threads(self.shape),
            self.constants.thread_block.required_threads as usize,
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        let config = self.constants.config;
        self.shape.validate(config);
        assert!(self.buffers.flat_k.len_bytes() >= config.flat_kv_bytes(self.shape));
        assert!(self.buffers.flat_v.len_bytes() >= config.flat_kv_bytes(self.shape));
        assert!(self.buffers.req_slots.len_bytes() >= config.index_bytes(self.shape));
        assert!(self.buffers.flat_token_indices.len_bytes() >= config.index_bytes(self.shape));
        assert!(self.buffers.page_ids.len_bytes() >= self.shape.page_ids_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const NUM_ACTIVE_WRITES: ReplayParameterKey = ReplayParameterKey::new("test.gqa.kv.active_writes");

    #[test]
    fn test_constants_have_explicit_thread_block_scope() {
        let config = Config {
            num_kv_heads: 2,
            head_dim: 2,
            page_bytes: 16,
            dtype: Dtype::Bfloat16,
        };
        let constants = KernelConstants::current(config);
        assert_eq!(constants.config, config);
        assert_eq!(constants.thread_block.required_threads, 256);
    }

    #[test]
    #[should_panic(expected = "GQA KV page-write threads exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        let config = Config {
            num_kv_heads: 2,
            head_dim: 2,
            page_bytes: 16,
            dtype: Dtype::Bfloat16,
        };
        Shape {
            num_total_token_writes: 1 << 30,
            page_table_layout: PageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
        }
        .validate(config);
    }

    #[test]
    fn test_fixed() {
        test_u16();
        test_f32();
    }

    #[test]
    fn test_bucketed_replay_preserves_inactive_kv_page() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let sentinel = 0x7777_u16;
        let page_values = 16;
        let pages = Buffer::from_slice(&device, &vec![sentinel; 2 * page_values]);
        let flat_k = Buffer::from_slice(&device, &[10_u16, 11, 12, 13, 14, 15, 16, 17]);
        let flat_v = Buffer::from_slice(&device, &[20_u16, 21, 22, 23, 24, 25, 26, 27]);
        let req_slots = Buffer::from_slice(&device, &[0_u32, 0, 0, 1]);
        let flat_token_indices = Buffer::from_slice(&device, &[0_u32, 1, 2, 0]);
        let page_ids = Buffer::from_slice(&device, &[0_u32, 1]);
        let config = Config {
            num_kv_heads: 1,
            head_dim: 2,
            page_bytes: (page_values * size_of::<u16>()) as u32,
            dtype: Dtype::Bfloat16,
        };
        let shape = Shape {
            num_total_token_writes: 4,
            page_table_layout: PageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
        };
        let kernel = Compute::new(&device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_bucketed(
            shape,
            Buffers {
                pages: &pages,
                flat_k: &flat_k,
                flat_v: &flat_v,
                req_slots: &req_slots,
                flat_token_indices: &flat_token_indices,
                page_ids: &page_ids,
            },
            ReplayU32::Parameter(NUM_ACTIVE_WRITES),
            ReplayU32::Fixed(0),
        ));
        let replay = builder.build();
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_WRITES, 3))
            .wait();
        assert_eq!(
            pages.read_typed::<u16>(page_values, page_values),
            vec![sentinel; page_values]
        );

        pages.write_typed(0, &vec![sentinel; 2 * page_values]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_WRITES, 4))
            .wait();
        assert_ne!(
            pages.read_typed::<u16>(page_values, page_values),
            vec![sentinel; page_values]
        );
    }

    fn test_u16() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let expected = [0, 0, 10, 11, 0, 0, 20, 21];
        let pages = Buffer::new_zeroed(&device, expected.len() * size_of::<u16>());
        let k = Buffer::from_slice(&device, &[10u16, 11]);
        let v = Buffer::from_slice(&device, &[20u16, 21]);
        run(&device, &stream, Dtype::Bfloat16, &pages, &k, &v);
        assert_eq!(pages.read_typed::<u16>(0, expected.len()), expected);
    }

    fn test_f32() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let expected = [0.0, 0.0, 10.0, 11.0, 0.0, 0.0, 20.0, 21.0];
        let pages = Buffer::new_zeroed(&device, expected.len() * size_of::<f32>());
        let k = Buffer::from_slice(&device, &[10.0f32, 11.0]);
        let v = Buffer::from_slice(&device, &[20.0f32, 21.0]);
        run(&device, &stream, Dtype::Float32, &pages, &k, &v);
        assert_eq!(pages.read_typed::<f32>(0, expected.len()), expected);
    }

    fn run(device: &Device, stream: &Stream, dtype: Dtype, pages: &Buffer, k: &Buffer, v: &Buffer) {
        let req_slots = Buffer::from_slice(device, &[0u32]);
        let flat_token_indices = Buffer::from_slice(device, &[1u32]);
        let page_ids = Buffer::from_slice(device, &[0u32]);
        let config = Config {
            num_kv_heads: 1,
            head_dim: 2,
            page_bytes: pages.len_bytes() as u32,
            dtype,
        };
        let shape = Shape {
            num_total_token_writes: 1,
            page_table_layout: PageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
        };
        let kernel = Compute::new(device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            Buffers {
                pages,
                flat_k: k,
                flat_v: v,
                req_slots: &req_slots,
                flat_token_indices: &flat_token_indices,
                page_ids: &page_ids,
            },
            ReplayU32::Fixed(0),
        ));
        stream.submit_replay(&builder.build()).wait();
    }
}
