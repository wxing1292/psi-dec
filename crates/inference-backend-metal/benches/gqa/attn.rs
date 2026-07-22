use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use half::bf16;
use inference_backend_metal::components::GQAPageTableLayout;
use inference_backend_metal::components::GQAPagedSDPAConfig;
use inference_backend_metal::components::GQAPagedSDPAKernels;
use inference_backend_metal::components::GQAPagedSDPAScratch;
use inference_backend_metal::components::GQAPagedSDPAShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;

const GQA_CONTEXT_LENS: [u32; 4] = [128, 1024, 2048, 4096];
const GQA_NUM_Q_HEADS: u32 = 32;
const GQA_NUM_KV_HEADS: u32 = 8;
const GQA_HEAD_DIM: u32 = 128;
const GQA_TOKENS_PER_PAGE: u32 = 16;
const GQA_KV_TOKEN_TILE_SIZE: u32 = 512;
const GQA_MAX_SDPA_MAP_TASK_TEMPLATES: u32 = 64;
const GQA_NUM_THREADS_PER_THREADBLOCK: u32 = 128;
const GQA_Q_HEAD_TILE_SIZE_CAP: u32 = 8;

fn bench_gqa_attn(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/gqa-attn");

    for context_len in GQA_CONTEXT_LENS {
        let fixture = GQAFixture::new(&device, &[context_len]);
        group.throughput(Throughput::Elements(context_len as u64));
        group.bench_function(format!("paged-sdpa/replay/ctx{context_len}"), |b| {
            b.iter(|| {
                fixture.run_replay();
                black_box(&fixture.output);
            });
        });
    }

    let ragged_context_lens = [128, 128, 128, 128, 128, 128, 128, 4096];
    let fixture = GQAFixture::new(&device, &ragged_context_lens);
    group.throughput(Throughput::Elements(
        ragged_context_lens.iter().map(|&context_len| context_len as u64).sum(),
    ));
    group.bench_function("paged-sdpa/replay/ragged-7x128-1x4096", |b| {
        b.iter(|| {
            fixture.run_replay();
            black_box(&fixture.output);
        });
    });

    group.finish();
}

struct GQAFixture {
    stream: Stream,
    output: Buffer,
    replay: ReplayProgram,
}

impl GQAFixture {
    fn new(device: &Device, context_lens: &[u32]) -> Self {
        assert!(!context_lens.is_empty());
        let max_context_len = context_lens.iter().copied().max().unwrap();
        let num_tokens = context_lens.len() as u32;
        let num_blocks = max_context_len.div_ceil(GQA_TOKENS_PER_PAGE);
        let (mut sdpa_map_task_template_values, cu_sdpa_partial_output_values) =
            sdpa_map_task_template_plan(context_lens);
        let num_sdpa_map_task_templates = sdpa_map_task_template_values.len() as u32 / 3;
        let total_sdpa_map_task_templates = num_sdpa_map_task_templates.next_power_of_two();
        sdpa_map_task_template_values.resize(total_sdpa_map_task_templates as usize * 3, u32::MAX);
        let page_table_layout = GQAPageTableLayout {
            num_req_slots: num_tokens,
            num_blocks,
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        };
        let config = GQAPagedSDPAConfig {
            num_q_heads: GQA_NUM_Q_HEADS,
            num_kv_heads: GQA_NUM_KV_HEADS,
            head_dim: GQA_HEAD_DIM,
            scale: 1.0 / (GQA_HEAD_DIM as f32).sqrt(),
            page_bytes: 2 * GQA_NUM_KV_HEADS * GQA_TOKENS_PER_PAGE * GQA_HEAD_DIM * Dtype::Bfloat16.item_size() as u32,
            page_table_layout,
            gqa_layer_index: 0,
            kv_token_tile_size: GQA_KV_TOKEN_TILE_SIZE,
            num_threads_per_threadblock: GQA_NUM_THREADS_PER_THREADBLOCK,
            q_head_tile_size: (GQA_NUM_Q_HEADS / GQA_NUM_KV_HEADS).min(GQA_Q_HEAD_TILE_SIZE_CAP),
            dtype: Dtype::Bfloat16,
        };
        let shape = GQAPagedSDPAShape {
            num_tokens,
            total_sdpa_map_task_templates,
        };
        shape.validate(config);

        let q = bf16_pattern_buffer(device, config.num_output_values(shape), 0.003);
        let kv_pages = bf16_pattern_buffer(
            device,
            page_table_layout.num_req_slots as usize
                * page_table_layout.num_blocks as usize
                * config.page_bytes as usize
                / Dtype::Bfloat16.item_size(),
            0.002,
        );
        let req_slots = Buffer::from_slice(device, &(0..num_tokens).collect::<Vec<_>>());
        let num_page_ids = page_table_layout.num_req_slots * page_table_layout.num_blocks;
        let page_ids = Buffer::from_slice(device, &(0..num_page_ids).collect::<Vec<_>>());
        let sdpa_map_task_templates = Buffer::from_slice(device, &sdpa_map_task_template_values);
        let cu_sdpa_partial_outputs = Buffer::from_slice(device, &cu_sdpa_partial_output_values);
        let scratch = GQAPagedSDPAScratch::new(device, config, shape);
        let output = Buffer::new_zeroed(device, config.q_bytes(shape));
        let stream = Stream::new(device);
        let kernels = GQAPagedSDPAKernels::new(device);
        let replay = build_gqa_replay(
            &stream,
            &kernels,
            config,
            shape,
            &q,
            &kv_pages,
            &req_slots,
            &page_ids,
            &sdpa_map_task_templates,
            &cu_sdpa_partial_outputs,
            &scratch,
            &output,
        );

        let fixture = Self { stream, output, replay };
        fixture.run_replay();
        fixture
    }

    fn run_replay(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

fn sdpa_map_task_template_plan(context_lens: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let num_kv_token_tiles = context_lens
        .iter()
        .map(|&context_len| context_len.div_ceil(GQA_KV_TOKEN_TILE_SIZE) as usize)
        .collect::<Vec<_>>();
    let mut num_sdpa_map_task_templates_by_q_token_tile = vec![1_usize; context_lens.len()];
    let mut num_sdpa_map_task_templates = num_sdpa_map_task_templates_by_q_token_tile.len();
    while num_sdpa_map_task_templates < GQA_MAX_SDPA_MAP_TASK_TEMPLATES as usize {
        let Some(q_token_tile_index) = (0..num_kv_token_tiles.len())
            .filter(|&q_token_tile_index| {
                num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index] < num_kv_token_tiles[q_token_tile_index]
            })
            .max_by_key(|&q_token_tile_index| {
                num_kv_token_tiles[q_token_tile_index]
                    .div_ceil(num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index])
            })
        else {
            break;
        };
        num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index] += 1;
        num_sdpa_map_task_templates += 1;
    }

    let mut sdpa_map_task_templates = Vec::with_capacity(num_sdpa_map_task_templates * 3);
    let mut cu_sdpa_partial_outputs = vec![0];
    for (q_token_tile_index, &context_len) in context_lens.iter().enumerate() {
        for sdpa_map_task_template_index in 0..num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index] {
            let kv_token_tile_begin = num_kv_token_tiles[q_token_tile_index] * sdpa_map_task_template_index
                / num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index];
            let kv_token_tile_end = num_kv_token_tiles[q_token_tile_index] * (sdpa_map_task_template_index + 1)
                / num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index];
            sdpa_map_task_templates.extend_from_slice(&[
                q_token_tile_index as u32,
                kv_token_tile_begin as u32 * GQA_KV_TOKEN_TILE_SIZE,
                context_len.min(kv_token_tile_end as u32 * GQA_KV_TOKEN_TILE_SIZE),
            ]);
        }
        cu_sdpa_partial_outputs.push((sdpa_map_task_templates.len() / 3) as u32);
    }
    (sdpa_map_task_templates, cu_sdpa_partial_outputs)
}

#[allow(clippy::too_many_arguments)]
fn build_gqa_replay(
    stream: &Stream,
    kernels: &GQAPagedSDPAKernels,
    config: GQAPagedSDPAConfig,
    shape: GQAPagedSDPAShape,
    q: &Buffer,
    kv_pages: &Buffer,
    req_slots: &Buffer,
    page_ids: &Buffer,
    sdpa_map_task_templates: &Buffer,
    cu_sdpa_partial_outputs: &Buffer,
    scratch: &GQAPagedSDPAScratch,
    output: &Buffer,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_map(
        config,
        shape,
        scratch.map_buffers(q, kv_pages, req_slots, page_ids, sdpa_map_task_templates),
    ));
    builder.record(kernels.invoke_reduce(config, shape, scratch.reduce_buffers(cu_sdpa_partial_outputs, output)));
    builder.build()
}

fn bf16_pattern_buffer(device: &Device, len: usize, scale: f32) -> Buffer {
    let v = (0..len)
        .map(|index| {
            let value = (index % 251) as f32 - 125.0;
            bf16::from_f32(value * scale).to_bits()
        })
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &v)
}

criterion_group!(benches, bench_gqa_attn);
criterion_main!(benches);
