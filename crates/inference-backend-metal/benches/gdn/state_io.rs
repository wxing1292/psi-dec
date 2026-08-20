use std::hint::black_box;
use std::mem::size_of;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::gdn::state_pages;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;

const STATE_PAGE_FLOATS: usize = 32 * 1024 / size_of::<f32>();
const STATE_IO_REQUEST_COUNTS: [u32; 3] = [1, 4, 16];
const QKV_DIM: usize = 4096;
const V_HEADS: usize = 16;
const V_HEAD_DIM: usize = 128;
const QK_HEAD_DIM: usize = 128;
const CONV_STATE_LEN: usize = 3;

fn bench_gdn_state_io(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/gdn-state-io");
    for num_state_io_requests in STATE_IO_REQUEST_COUNTS {
        let fixture = StateIOFixture::new(&device, num_state_io_requests);
        group.throughput(Throughput::Elements(fixture.total_pages() as u64));
        group.bench_function(format!("restore/io_requests{num_state_io_requests}"), |b| {
            b.iter(|| {
                fixture.restore();
                black_box(&fixture.recurrent_state_arena);
            });
        });
        group.bench_function(format!("publish/io_requests{num_state_io_requests}"), |b| {
            b.iter(|| {
                fixture.publish();
                black_box(&fixture.pages);
            });
        });
    }
    group.finish();
}

struct StateIOFixture {
    stream: Stream,
    pages: Buffer,
    recurrent_state_arena: Buffer,
    restore: ReplayProgram,
    publish: ReplayProgram,
    total_pages: usize,
}

impl StateIOFixture {
    fn new(device: &Device, num_state_io_requests: u32) -> Self {
        let config = state_io_config(num_state_io_requests);
        let page_bytes = config.page_bytes as usize;
        let recurrent_state_bytes = config.recurrent_state_bytes as usize;
        let conv_state_bytes = config.conv_state_bytes as usize;
        let recurrent_pages_per_state = recurrent_state_bytes.div_ceil(page_bytes);
        let conv_pages_per_state = conv_state_bytes.div_ceil(page_bytes);
        let recurrent_page_count = num_state_io_requests as usize * recurrent_pages_per_state;
        let conv_page_count = num_state_io_requests as usize * conv_pages_per_state;
        let total_pages = recurrent_page_count + conv_page_count;
        let stream = Stream::new(device);
        let pages = f32_pattern_buffer(device, total_pages * STATE_PAGE_FLOATS, 0.0001);
        let recurrent_state_arena = Buffer::new_zeroed(device, num_state_io_requests as usize * recurrent_state_bytes);
        let conv_state_arena = Buffer::new_zeroed(device, num_state_io_requests as usize * conv_state_bytes);
        let page_ids = Buffer::from_slice(
            device,
            &(0..num_state_io_requests as usize)
                .flat_map(|state_io_request_index| {
                    let recurrent_start = state_io_request_index * recurrent_pages_per_state;
                    let conv_start = recurrent_page_count + state_io_request_index * conv_pages_per_state;
                    (recurrent_start..recurrent_start + recurrent_pages_per_state)
                        .chain(conv_start..conv_start + conv_pages_per_state)
                })
                .map(|page_id| page_id as u32)
                .collect::<Vec<_>>(),
        );
        let state_slots = Buffer::from_slice(device, &(0..num_state_io_requests).collect::<Vec<_>>());
        let bindings = StateIOBindings {
            pages: &pages,
            page_ids: &page_ids,
            recurrent_state_slots: &state_slots,
            conv_state_slots: &state_slots,
            recurrent_states: &recurrent_state_arena,
            conv_states: &conv_state_arena,
        };
        let fixture = Self {
            restore: build_restore_replay(&stream, num_state_io_requests, config, bindings, device),
            publish: build_publish_replay(&stream, num_state_io_requests, config, bindings, device),
            stream,
            pages,
            recurrent_state_arena,
            total_pages,
        };
        fixture.restore();
        fixture.publish();
        fixture
    }

    fn total_pages(&self) -> usize {
        self.total_pages
    }

    fn restore(&self) {
        self.stream.submit_replay(&self.restore).wait();
    }

    fn publish(&self) {
        self.stream.submit_replay(&self.publish).wait();
    }
}

#[derive(Clone, Copy)]
struct StateIOBindings<'a> {
    pages: &'a Buffer,
    page_ids: &'a Buffer,
    recurrent_state_slots: &'a Buffer,
    conv_state_slots: &'a Buffer,
    recurrent_states: &'a Buffer,
    conv_states: &'a Buffer,
}

fn build_restore_replay(
    stream: &Stream,
    num_state_io_requests: u32,
    config: state_pages::Config,
    bindings: StateIOBindings<'_>,
    device: &Device,
) -> ReplayProgram {
    let read = state_pages::Read::new(device, config);
    let mut builder = stream.create_replay_program();
    builder.record(read.invoke(
        state_pages::Shape { num_state_io_requests },
        state_pages::ReadBuffers {
            pages: bindings.pages,
            recurrent_states: bindings.recurrent_states,
            conv_states: bindings.conv_states,
            page_ids: bindings.page_ids,
            recurrent_state_slots: bindings.recurrent_state_slots,
            conv_state_slots: bindings.conv_state_slots,
        },
    ));
    builder.build()
}

fn build_publish_replay(
    stream: &Stream,
    num_state_io_requests: u32,
    config: state_pages::Config,
    bindings: StateIOBindings<'_>,
    device: &Device,
) -> ReplayProgram {
    let write = state_pages::Write::new(device, config);
    let mut builder = stream.create_replay_program();
    builder.record(write.invoke(
        state_pages::Shape { num_state_io_requests },
        state_pages::WriteBuffers {
            pages: bindings.pages,
            recurrent_states: bindings.recurrent_states,
            conv_states: bindings.conv_states,
            page_ids: bindings.page_ids,
            recurrent_state_slots: bindings.recurrent_state_slots,
            conv_state_slots: bindings.conv_state_slots,
        },
    ));
    builder.build()
}

fn state_io_config(num_state_slots: u32) -> state_pages::Config {
    state_pages::Config {
        num_gdn_layers: 1,
        num_state_slots,
        recurrent_state_bytes: (V_HEADS * V_HEAD_DIM * QK_HEAD_DIM * size_of::<f32>()) as u32,
        conv_state_bytes: (QKV_DIM * CONV_STATE_LEN * size_of::<f32>()) as u32,
        page_bytes: (STATE_PAGE_FLOATS * size_of::<f32>()) as u32,
    }
}

fn f32_pattern_buffer(device: &Device, len: usize, scale: f32) -> Buffer {
    Buffer::from_slice(
        device,
        &(0..len)
            .map(|index| ((index % 257) as f32 - 128.0) * scale)
            .collect::<Vec<_>>(),
    )
}

criterion_group!(benches, bench_gdn_state_io);
criterion_main!(benches);
