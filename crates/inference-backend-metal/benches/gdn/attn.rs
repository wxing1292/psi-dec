use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::GDNCoreBuffers;
use inference_backend_metal::components::GDNCoreConfig;
use inference_backend_metal::components::GDNCoreForwardCandidateStateUpdateBuffers;
use inference_backend_metal::components::GDNCoreKernels;
use inference_backend_metal::components::GDNCoreShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;

const GDN_CASES: [(u32, u32); 4] = [(1, 1), (4, 1), (1, 16), (4, 4)];
const GDN_CONV_DIM: u32 = 4096;
const GDN_V_HEADS: u32 = 16;
const GDN_V_HEAD_DIM: u32 = 128;
const GDN_QK_HEAD_DIM: u32 = 128;
const GDN_QK_HEADS: u32 = 8;
const GDN_CONV_STATE_LEN: u32 = 3;
const GDN_CONV_KERNEL_SIZE: u32 = 4;

fn bench_gdn_attn(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/gdn-attn");

    for (batch, tokens) in GDN_CASES {
        let fixture = GDNFixture::new(&device, batch, tokens);
        let num_tokens = batch * tokens;
        group.throughput(Throughput::Elements(num_tokens as u64));
        group.bench_function(
            format!("core-ragged_recurrent/replay/batch{batch}/tokens{tokens}"),
            |b| {
                b.iter(|| {
                    fixture.run_with_state_replay();
                    black_box(&fixture.pre_output_hidden_states);
                });
            },
        );
        group.bench_function(
            format!("forward_candidate_state_update/replay/batch{batch}/tokens{tokens}"),
            |b| {
                b.iter(|| {
                    fixture.run_forward_candidate_state_update_replay();
                    black_box(&fixture.recurrent_state_arena);
                });
            },
        );
    }

    group.finish();
}

struct GDNFixture {
    stream: Stream,
    recurrent_state_arena: Buffer,
    pre_output_hidden_states: Buffer,
    with_state_replay: ReplayProgram,
    forward_candidate_state_update_replay: ReplayProgram,
}

impl GDNFixture {
    fn new(device: &Device, batch: u32, tokens: u32) -> Self {
        let shape = GDNCoreShape {
            num_reqs: batch,
            num_tokens: batch * tokens,
        };
        let config = GDNCoreConfig {
            num_qk_heads: GDN_QK_HEADS,
            qk_head_dim: GDN_QK_HEAD_DIM,
            num_v_heads: GDN_V_HEADS,
            v_head_dim: GDN_V_HEAD_DIM,
            conv_kernel_size: GDN_CONV_KERNEL_SIZE,
            v_dim_tile_size: 8,
        };
        let cu_token_values = (0..=shape.num_reqs)
            .map(|req_index| (req_index * tokens) as i32)
            .collect::<Vec<_>>();
        let src_state_slot_values = (0..shape.num_reqs).collect::<Vec<_>>();
        let dst_slot_id_values = (shape.num_reqs..shape.num_reqs * 2).collect::<Vec<_>>();
        let candidate_dst_slot_id_values = (0..shape.num_tokens)
            .map(|flat_token_index| shape.num_reqs * 2 + flat_token_index)
            .collect::<Vec<_>>();
        let state_slot_count = shape.num_reqs * 2 + shape.num_tokens;

        let stream = Stream::new(device);
        let kernels = GDNCoreKernels::new(device, config);
        let q_scale = 1.0 / (config.qk_head_dim as f32).sqrt();
        let projected_qkv = f32_pattern_buffer(device, config.num_qkv_values(shape), 0.001);
        let a = f32_pattern_buffer(device, shape.num_tokens as usize * config.num_v_heads as usize, 0.002);
        let b = f32_pattern_buffer(device, shape.num_tokens as usize * config.num_v_heads as usize, -0.001);
        let z = f32_pattern_buffer(device, config.num_recurrent_output_values(shape), 0.0015);
        let conv_weight = f32_pattern_buffer(
            device,
            config.qkv_dim() as usize * config.conv_kernel_size as usize,
            0.0005,
        );
        let norm_weight = Buffer::from_slice(device, &vec![1.0_f32; config.v_head_dim as usize]);
        let a_log_decay = Buffer::from_slice(device, &vec![-0.01_f32; config.num_v_heads as usize]);
        let dt_bias = Buffer::from_slice(device, &vec![0.02_f32; config.num_v_heads as usize]);
        let cu_tokens = Buffer::from_slice(device, &cu_token_values);
        let src_state_slots = Buffer::from_slice(device, &src_state_slot_values);
        let dst_slot_ids = Buffer::from_slice(device, &dst_slot_id_values);
        let candidate_dst_slot_ids = Buffer::from_slice(device, &candidate_dst_slot_id_values);
        let conv_state = f32_pattern_buffer(device, config.num_conv_state_values(shape), 0.001);
        let next_conv_state = Buffer::new_zeroed(
            device,
            state_slot_count as usize * config.qkv_dim() as usize * config.conv_state_len() as usize * size_of::<f32>(),
        );
        let recurrent_state_arena = f32_pattern_buffer(
            device,
            config.recurrent_state_stride() * state_slot_count as usize,
            0.0001,
        );
        let conv_qkv = Buffer::new_zeroed(device, config.num_qkv_values(shape) * size_of::<f32>());
        let recurrent_output = Buffer::new_zeroed(device, config.num_recurrent_output_values(shape) * size_of::<f32>());
        let pre_output_hidden_states =
            Buffer::new_zeroed(device, config.num_recurrent_output_values(shape) * size_of::<f32>());
        let buffers = GDNCoreBuffers {
            projected_qkv: &projected_qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log_decay: &a_log_decay,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_state_slots: &src_state_slots,
            dst_state_slots: &dst_slot_ids,
            conv_state: &conv_state,
            conv_state_offset_bytes: 0,
            next_conv_state: &next_conv_state,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &recurrent_state_arena,
            recurrent_state_arena_offset_bytes: 0,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            pre_output_hidden_states: &pre_output_hidden_states,
        };
        let with_state_replay = build_gdn_with_state_replay(&stream, &kernels, shape, buffers, q_scale, 1.0e-6);
        let forward_candidate_state_update_replay = build_gdn_forward_candidate_state_update_replay(
            &stream,
            &kernels,
            shape,
            buffers,
            &candidate_dst_slot_ids,
            q_scale,
            1.0e-6,
        );

        let fixture = Self {
            stream,
            recurrent_state_arena,
            pre_output_hidden_states,
            with_state_replay,
            forward_candidate_state_update_replay,
        };
        fixture.run_with_state_replay();
        fixture.run_forward_candidate_state_update_replay();
        fixture
    }

    fn run_with_state_replay(&self) {
        self.stream.submit_replay(&self.with_state_replay).wait();
    }

    fn run_forward_candidate_state_update_replay(&self) {
        self.stream
            .submit_replay(&self.forward_candidate_state_update_replay)
            .wait();
    }
}

fn build_gdn_with_state_replay(
    stream: &Stream,
    kernels: &GDNCoreKernels,
    shape: GDNCoreShape,
    buffers: GDNCoreBuffers<'_>,
    q_scale: f32,
    eps: f32,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke(shape, buffers, q_scale, eps));
    builder.build()
}

fn build_gdn_forward_candidate_state_update_replay(
    stream: &Stream,
    kernels: &GDNCoreKernels,
    shape: GDNCoreShape,
    buffers: GDNCoreBuffers<'_>,
    candidate_dst_slot_ids: &Buffer,
    q_scale: f32,
    eps: f32,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_forward_candidate_state_update(
        shape,
        GDNCoreForwardCandidateStateUpdateBuffers {
            core: buffers,
            flat_candidate_state_slots: candidate_dst_slot_ids,
        },
        q_scale,
        eps,
    ));
    builder.build()
}

fn f32_pattern_buffer(device: &Device, len: usize, scale: f32) -> Buffer {
    let values = (0..len)
        .map(|index| {
            let value = (index % 251) as f32 - 125.0;
            value * scale
        })
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &values)
}

criterion_group!(benches, bench_gdn_attn);
criterion_main!(benches);
