use super::assert_u32_count_domain;
use super::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const GQA_KV_PAGE_UPDATE_SOURCE: &str = include_str!("metal/gqa_kv_pages.metal");

const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQAPageTableLayout {
    pub num_req_slots: u32,
    pub num_gqa_layers: u32,
    pub num_blocks: u32,
    pub num_page_ids_per_block: u32,
}

impl GQAPageTableLayout {
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
pub struct GQAKVPageUpdateConfig {
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_bytes: u32,
    pub dtype: Dtype,
}

impl GQAKVPageUpdateConfig {
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

    pub fn index_bytes(self, shape: GQAKVPageUpdateShape) -> usize {
        (shape.num_token_writes as usize)
            .checked_mul(size_of::<u32>())
            .expect("GQA KV page-update index bytes must fit usize")
    }

    pub fn flat_kv_bytes(self, shape: GQAKVPageUpdateShape) -> usize {
        self.num_total_threads(shape)
            .checked_mul(self.dtype.item_size())
            .expect("GQA flattened K/V byte length must fit usize")
    }

    pub fn num_total_threads(self, shape: GQAKVPageUpdateShape) -> usize {
        checked_product(
            "GQA KV page-update thread count",
            &[
                shape.num_token_writes as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQAKVPageUpdateShape {
    pub num_token_writes: u32,
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
}

impl GQAKVPageUpdateShape {
    pub fn validate(self, config: GQAKVPageUpdateConfig) {
        config.validate();
        assert!(self.num_token_writes > 0);
        self.page_table_layout.validate();
        assert!(self.gqa_layer_index < self.page_table_layout.num_gqa_layers);
        assert_u32_count_domain(config.num_total_threads(self), "GQA KV page-update threads");
    }

    pub fn page_ids_bytes(self) -> usize {
        self.page_table_layout.bytes()
    }
}

#[derive(Clone, Copy)]
pub struct GQAKVPageUpdateBuffers<'a> {
    pub pages: &'a Buffer,
    pub flat_k: &'a Buffer,
    pub flat_v: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub page_ids: &'a Buffer,
}

pub struct GQAKVPageUpdate {
    config: GQAKVPageUpdateConfig,
    kernel: Kernel,
}

impl GQAKVPageUpdate {
    pub fn new(device: &Device, config: GQAKVPageUpdateConfig) -> Self {
        config.validate();
        let source = kv_page_update_source(config);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_kv_page_update_f32",
            Dtype::Bfloat16 => "gqa_kv_page_update_u16",
            dtype => panic!("unsupported GQA KV page update dtype {dtype:?}"),
        };
        Self {
            config,
            kernel: Kernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GQAKVPageUpdateShape,
        buffers: GQAKVPageUpdateBuffers<'a>,
    ) -> GQAKVPageUpdateInvocation<'a> {
        GQAKVPageUpdateInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

fn kv_page_update_source(config: GQAKVPageUpdateConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_kv_heads = {}u;\nconstant uint head_dim = {}u;\nconstant uint \
         num_tokens_per_page = {}u;\nconstant uint page_bytes = {}u;",
        config.num_kv_heads,
        config.head_dim,
        config.num_tokens_per_page(),
        config.page_bytes,
    );
    GQA_KV_PAGE_UPDATE_SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct GQAKVPageUpdateInvocation<'a> {
    config: GQAKVPageUpdateConfig,
    kernel: &'a Kernel,
    shape: GQAKVPageUpdateShape,
    buffers: GQAKVPageUpdateBuffers<'a>,
}

impl Operator for GQAKVPageUpdateInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl GQAKVPageUpdateInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.flat_k.len_bytes() >= self.config.flat_kv_bytes(self.shape));
        assert!(self.buffers.flat_v.len_bytes() >= self.config.flat_kv_bytes(self.shape));
        assert!(self.buffers.req_slots.len_bytes() >= self.config.index_bytes(self.shape));
        assert!(self.buffers.flat_token_indices.len_bytes() >= self.config.index_bytes(self.shape));
        assert!(self.buffers.page_ids.len_bytes() >= self.shape.page_ids_bytes());
    }

    fn record_compute(self, builder: &CommandRecorder) {
        builder.set_kernel(self.kernel);
        builder.set_buffer_write(0, self.buffers.pages, 0);
        builder.set_buffer_read(1, self.buffers.flat_k, 0);
        builder.set_buffer_read(2, self.buffers.flat_v, 0);
        builder.set_buffer_read(3, self.buffers.req_slots, 0);
        builder.set_buffer_read(4, self.buffers.flat_token_indices, 0);
        builder.set_buffer_read(5, self.buffers.page_ids, 0);
        builder.set_u32(6, self.shape.num_token_writes);
        builder.set_u32(7, self.shape.gqa_layer_index);
        builder.set_u32(8, self.shape.page_table_layout.num_gqa_layers);
        builder.set_u32(9, self.shape.page_table_layout.num_blocks);
        builder.set_u32(10, self.shape.page_table_layout.num_page_ids_per_block);
        builder.dispatch_1d(self.config.num_total_threads(self.shape), NUM_THREADS_PER_THREADBLOCK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Stream;

    #[test]
    #[should_panic(expected = "GQA KV page-update threads exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        let config = GQAKVPageUpdateConfig {
            num_kv_heads: 2,
            head_dim: 2,
            page_bytes: 16,
            dtype: Dtype::Bfloat16,
        };
        GQAKVPageUpdateShape {
            num_token_writes: 1 << 30,
            page_table_layout: GQAPageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
            gqa_layer_index: 0,
        }
        .validate(config);
    }

    #[test]
    fn test_fixed() {
        test_u16();
        test_f32();
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
        let config = GQAKVPageUpdateConfig {
            num_kv_heads: 1,
            head_dim: 2,
            page_bytes: pages.len_bytes() as u32,
            dtype,
        };
        let shape = GQAKVPageUpdateShape {
            num_token_writes: 1,
            page_table_layout: GQAPageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
            gqa_layer_index: 0,
        };
        let kernel = GQAKVPageUpdate::new(device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            GQAKVPageUpdateBuffers {
                pages,
                flat_k: k,
                flat_v: v,
                req_slots: &req_slots,
                flat_token_indices: &flat_token_indices,
                page_ids: &page_ids,
            },
        ));
        stream.submit_replay(&builder.build()).wait();
    }
}
