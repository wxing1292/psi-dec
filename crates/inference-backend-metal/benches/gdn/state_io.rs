use std::hint::black_box;
use std::mem::size_of;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::BufferCopy32Buffers;
use inference_backend_metal::components::BufferCopy32Shape;
use inference_backend_metal::components::F32BufferCopyKernel;
use inference_backend_metal::components::GDNStatePageBatchRead;
use inference_backend_metal::components::GDNStatePageBatchReadBuffers;
use inference_backend_metal::components::GDNStatePageBatchShape;
use inference_backend_metal::components::GDNStatePageBatchWrite;
use inference_backend_metal::components::GDNStatePageBatchWriteBuffers;
use inference_backend_metal::components::GDNStatePageRead;
use inference_backend_metal::components::GDNStatePageReadBuffers;
use inference_backend_metal::components::GDNStatePageShape;
use inference_backend_metal::components::GDNStatePageWrite;
use inference_backend_metal::components::GDNStatePageWriteBuffers;
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
        group.bench_function(format!("restore/legacy/io_requests{num_state_io_requests}"), |b| {
            b.iter(|| {
                fixture.restore_legacy();
                black_box(&fixture.recurrent_state_arena);
            });
        });
        group.bench_function(format!("restore/batch/io_requests{num_state_io_requests}"), |b| {
            b.iter(|| {
                fixture.restore_batch();
                black_box(&fixture.recurrent_state_arena);
            });
        });
        group.bench_function(format!("publish/legacy/io_requests{num_state_io_requests}"), |b| {
            b.iter(|| {
                fixture.publish_legacy();
                black_box(&fixture.pages);
            });
        });
        group.bench_function(format!("publish/batch/io_requests{num_state_io_requests}"), |b| {
            b.iter(|| {
                fixture.publish_batch();
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
    legacy_restore: ReplayProgram,
    batch_restore: ReplayProgram,
    legacy_publish: ReplayProgram,
    batch_publish: ReplayProgram,
    total_pages: usize,
}

impl StateIOFixture {
    fn new(device: &Device, num_state_io_requests: u32) -> Self {
        let page_bytes = STATE_PAGE_FLOATS * size_of::<f32>();
        let recurrent_state_bytes = V_HEADS * V_HEAD_DIM * QK_HEAD_DIM * size_of::<f32>();
        let conv_state_bytes = QKV_DIM * CONV_STATE_LEN * size_of::<f32>();
        let recurrent_pages_per_state = recurrent_state_bytes.div_ceil(page_bytes);
        let conv_pages_per_state = conv_state_bytes.div_ceil(page_bytes);
        let recurrent_page_count = num_state_io_requests as usize * recurrent_pages_per_state;
        let conv_page_count = num_state_io_requests as usize * conv_pages_per_state;
        let total_pages = recurrent_page_count + conv_page_count;
        let stream = Stream::new(device);
        let pages = f32_pattern_buffer(device, total_pages * STATE_PAGE_FLOATS, 0.0001);
        let recurrent_state_arena = Buffer::new_zeroed(device, num_state_io_requests as usize * recurrent_state_bytes);
        let conv_state = Buffer::new_zeroed(device, num_state_io_requests as usize * conv_state_bytes);
        let recurrent_scratch = Buffer::new_zeroed(device, recurrent_state_bytes);
        let conv_scratch = Buffer::new_zeroed(device, conv_state_bytes);
        let recurrent_page_ids = Buffer::from_slice(device, &(0..recurrent_page_count as u32).collect::<Vec<_>>());
        let conv_page_ids = Buffer::from_slice(
            device,
            &(recurrent_page_count as u32..(recurrent_page_count + conv_page_count) as u32).collect::<Vec<_>>(),
        );
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
        let fixture = Self {
            legacy_restore: legacy_restore(
                &stream,
                num_state_io_requests,
                &pages,
                &recurrent_scratch,
                &conv_scratch,
                &recurrent_page_ids,
                &conv_page_ids,
                &recurrent_state_arena,
                &conv_state,
                recurrent_state_bytes,
                conv_state_bytes,
                recurrent_pages_per_state,
                conv_pages_per_state,
                device,
            ),
            batch_restore: batch_restore(
                &stream,
                num_state_io_requests,
                BatchStateIO {
                    pages: &pages,
                    page_ids: &page_ids,
                    state_slots: &state_slots,
                    recurrent_states: &recurrent_state_arena,
                    conv_states: &conv_state,
                },
                device,
            ),
            legacy_publish: legacy_publish(
                &stream,
                num_state_io_requests,
                &pages,
                &recurrent_scratch,
                &conv_scratch,
                &recurrent_page_ids,
                &conv_page_ids,
                &recurrent_state_arena,
                &conv_state,
                recurrent_state_bytes,
                conv_state_bytes,
                recurrent_pages_per_state,
                conv_pages_per_state,
                device,
            ),
            batch_publish: batch_publish(
                &stream,
                num_state_io_requests,
                BatchStateIO {
                    pages: &pages,
                    page_ids: &page_ids,
                    state_slots: &state_slots,
                    recurrent_states: &recurrent_state_arena,
                    conv_states: &conv_state,
                },
                device,
            ),
            stream,
            pages,
            recurrent_state_arena,
            total_pages,
        };
        fixture.restore_legacy();
        fixture.restore_batch();
        fixture.publish_legacy();
        fixture.publish_batch();
        fixture
    }

    fn total_pages(&self) -> usize {
        self.total_pages
    }
    fn restore_legacy(&self) {
        self.stream.submit_replay(&self.legacy_restore).wait();
    }
    fn restore_batch(&self) {
        self.stream.submit_replay(&self.batch_restore).wait();
    }
    fn publish_legacy(&self) {
        self.stream.submit_replay(&self.legacy_publish).wait();
    }
    fn publish_batch(&self) {
        self.stream.submit_replay(&self.batch_publish).wait();
    }
}

#[allow(clippy::too_many_arguments)]
fn legacy_restore(
    stream: &Stream,
    num_state_io_requests: u32,
    pages: &Buffer,
    recurrent_scratch: &Buffer,
    conv_scratch: &Buffer,
    recurrent_page_ids: &Buffer,
    conv_page_ids: &Buffer,
    recurrent_states: &Buffer,
    conv_states: &Buffer,
    recurrent_bytes: usize,
    conv_bytes: usize,
    recurrent_pages: usize,
    conv_pages: usize,
    device: &Device,
) -> ReplayProgram {
    let read = GDNStatePageRead::new(device);
    let copy = F32BufferCopyKernel::new(device);
    let mut builder = stream.create_replay_program();
    for state_io_request_index in 0..num_state_io_requests as usize {
        builder.record(read.invoke(
            GDNStatePageShape {
                state_bytes: recurrent_bytes as u32,
                page_bytes: (STATE_PAGE_FLOATS * size_of::<f32>()) as u32,
            },
            GDNStatePageReadBuffers {
                pages,
                flat_state: recurrent_scratch,
                page_ids: recurrent_page_ids,
                page_id_start_index: (state_io_request_index * recurrent_pages) as u32,
            },
        ));
        builder.record(copy.invoke(
            BufferCopy32Shape {
                num_values: (recurrent_bytes / size_of::<f32>()) as u32,
            },
            BufferCopy32Buffers {
                input: recurrent_scratch,
                output: recurrent_states,
                input_offset_bytes: 0,
                output_offset_bytes: state_io_request_index * recurrent_bytes,
            },
        ));
        builder.record(read.invoke(
            GDNStatePageShape {
                state_bytes: conv_bytes as u32,
                page_bytes: (STATE_PAGE_FLOATS * size_of::<f32>()) as u32,
            },
            GDNStatePageReadBuffers {
                pages,
                flat_state: conv_scratch,
                page_ids: conv_page_ids,
                page_id_start_index: (state_io_request_index * conv_pages) as u32,
            },
        ));
        builder.record(copy.invoke(
            BufferCopy32Shape {
                num_values: (conv_bytes / size_of::<f32>()) as u32,
            },
            BufferCopy32Buffers {
                input: conv_scratch,
                output: conv_states,
                input_offset_bytes: 0,
                output_offset_bytes: state_io_request_index * conv_bytes,
            },
        ));
    }
    builder.build()
}

struct BatchStateIO<'a> {
    pages: &'a Buffer,
    page_ids: &'a Buffer,
    state_slots: &'a Buffer,
    recurrent_states: &'a Buffer,
    conv_states: &'a Buffer,
}

fn batch_restore(
    stream: &Stream,
    num_state_io_requests: u32,
    state_io: BatchStateIO<'_>,
    device: &Device,
) -> ReplayProgram {
    let read = GDNStatePageBatchRead::new(device);
    let mut builder = stream.create_replay_program();
    builder.record(read.invoke(
        GDNStatePageBatchShape {
            num_gdn_layers: 1,
            num_state_io_requests,
            num_state_slots: num_state_io_requests,
            page_bytes: (STATE_PAGE_FLOATS * size_of::<f32>()) as u32,
        },
        GDNStatePageBatchReadBuffers {
            pages: state_io.pages,
            recurrent_states: state_io.recurrent_states,
            conv_states: state_io.conv_states,
            page_ids: state_io.page_ids,
            state_slots: state_io.state_slots,
        },
    ));
    builder.build()
}

#[allow(clippy::too_many_arguments)]
fn legacy_publish(
    stream: &Stream,
    num_state_io_requests: u32,
    pages: &Buffer,
    recurrent_scratch: &Buffer,
    conv_scratch: &Buffer,
    recurrent_page_ids: &Buffer,
    conv_page_ids: &Buffer,
    recurrent_states: &Buffer,
    conv_states: &Buffer,
    recurrent_bytes: usize,
    conv_bytes: usize,
    recurrent_pages: usize,
    conv_pages: usize,
    device: &Device,
) -> ReplayProgram {
    let write = GDNStatePageWrite::new(device);
    let copy = F32BufferCopyKernel::new(device);
    let mut builder = stream.create_replay_program();
    for state_io_request_index in 0..num_state_io_requests as usize {
        builder.record(copy.invoke(
            BufferCopy32Shape {
                num_values: (recurrent_bytes / size_of::<f32>()) as u32,
            },
            BufferCopy32Buffers {
                input: recurrent_states,
                output: recurrent_scratch,
                input_offset_bytes: state_io_request_index * recurrent_bytes,
                output_offset_bytes: 0,
            },
        ));
        builder.record(write.invoke(
            GDNStatePageShape {
                state_bytes: recurrent_bytes as u32,
                page_bytes: (STATE_PAGE_FLOATS * size_of::<f32>()) as u32,
            },
            GDNStatePageWriteBuffers {
                pages,
                flat_state: recurrent_scratch,
                page_ids: recurrent_page_ids,
                page_id_start_index: (state_io_request_index * recurrent_pages) as u32,
            },
        ));
        builder.record(copy.invoke(
            BufferCopy32Shape {
                num_values: (conv_bytes / size_of::<f32>()) as u32,
            },
            BufferCopy32Buffers {
                input: conv_states,
                output: conv_scratch,
                input_offset_bytes: state_io_request_index * conv_bytes,
                output_offset_bytes: 0,
            },
        ));
        builder.record(write.invoke(
            GDNStatePageShape {
                state_bytes: conv_bytes as u32,
                page_bytes: (STATE_PAGE_FLOATS * size_of::<f32>()) as u32,
            },
            GDNStatePageWriteBuffers {
                pages,
                flat_state: conv_scratch,
                page_ids: conv_page_ids,
                page_id_start_index: (state_io_request_index * conv_pages) as u32,
            },
        ));
    }
    builder.build()
}

fn batch_publish(
    stream: &Stream,
    num_state_io_requests: u32,
    state_io: BatchStateIO<'_>,
    device: &Device,
) -> ReplayProgram {
    let write = GDNStatePageBatchWrite::new(device);
    let mut builder = stream.create_replay_program();
    builder.record(write.invoke(
        GDNStatePageBatchShape {
            num_gdn_layers: 1,
            num_state_io_requests,
            num_state_slots: num_state_io_requests,
            page_bytes: (STATE_PAGE_FLOATS * size_of::<f32>()) as u32,
        },
        GDNStatePageBatchWriteBuffers {
            pages: state_io.pages,
            recurrent_states: state_io.recurrent_states,
            conv_states: state_io.conv_states,
            page_ids: state_io.page_ids,
            state_slots: state_io.state_slots,
        },
    ));
    builder.build()
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
