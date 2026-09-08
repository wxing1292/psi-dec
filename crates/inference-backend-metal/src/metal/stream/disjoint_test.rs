use super::CommandRecorder;
use super::Operator;
use super::ReplayArguments;
use super::ReplayParameterKey;
use super::Stream;
use crate::metal::Buffer;
use crate::metal::CompiledKernel;
use crate::metal::Device;

const NUM_TOTAL_VALUES: u32 = 4096;
const CANARY: u32 = 0xDEAD_BEEF;
const NUM_ACTIVE_VALUES: ReplayParameterKey = ReplayParameterKey::new("test.disjoint.num_active_values");
const SPLIT: ReplayParameterKey = ReplayParameterKey::new("test.disjoint.split");
const SEED: ReplayParameterKey = ReplayParameterKey::new("test.disjoint.seed");

const SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void prepare(
        device const uint* input [[buffer(0)]],
        device uint* prepared [[buffer(1)]],
        device const uint* joined [[buffer(2)]],
        constant uint& num_active_values [[buffer(3)]],
        constant uint& seed [[buffer(4)]],
        constant uint& layer [[buffer(5)]],
        uint id [[thread_position_in_grid]]) {
        if (id >= num_active_values) return;
        const uint value = layer == 0 ? input[id] : joined[id + 1];
        prepared[id] = value * 3 + seed + layer;
    }

    kernel void branch(
        device uint* output [[buffer(0)]],
        device const uint* prepared [[buffer(1)]],
        constant uint& num_active_values [[buffer(2)]],
        constant uint& split [[buffer(3)]],
        constant uint& prefix [[buffer(4)]],
        uint id [[thread_position_in_grid]]) {
        if (id >= num_active_values || ((id < split) != (prefix != 0))) return;
        output[id + 1] = prepared[id] + (prefix != 0 ? 11 : 29);
    }

    kernel void join(
        device const uint* branches [[buffer(0)]],
        device uint* output [[buffer(1)]],
        constant uint& num_active_values [[buffer(2)]],
        uint id [[thread_position_in_grid]]) {
        if (id >= num_active_values) return;
        output[id + 1] = branches[id + 1] + branches[num_active_values - id];
    }
"#;

struct DisjointInvocation<'a> {
    prepare: &'a CompiledKernel,
    branch: &'a CompiledKernel,
    join: &'a CompiledKernel,
    input: &'a Buffer,
    prepared: &'a Buffer,
    branches: &'a Buffer,
    joined: &'a Buffer,
}

impl Operator for DisjointInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        for layer in 0..2 {
            recorder.set_kernel(self.prepare);
            recorder.set_buffer_read(0, self.input, 0);
            recorder.set_buffer_write(1, self.prepared, 0);
            recorder.set_buffer_read(2, self.joined, 0);
            recorder.bind_u32(3, NUM_ACTIVE_VALUES, 1, NUM_TOTAL_VALUES);
            recorder.bind_u32(4, SEED, 0, 100);
            recorder.set_u32(5, layer);
            recorder.dispatch_1d(NUM_TOTAL_VALUES as usize, 256);

            recorder.record_disjoint_buffers(&[self.branches], || {
                for prefix in [true, false] {
                    recorder.set_kernel(self.branch);
                    if prefix {
                        recorder.set_barrier_before();
                    }
                    recorder.set_buffer_write(0, self.branches, 0);
                    recorder.set_buffer_read(1, self.prepared, 0);
                    recorder.bind_u32(2, NUM_ACTIVE_VALUES, 1, NUM_TOTAL_VALUES);
                    recorder.bind_u32(3, SPLIT, 0, NUM_TOTAL_VALUES);
                    recorder.set_u32(4, u32::from(prefix));
                    recorder.dispatch_1d(NUM_TOTAL_VALUES as usize, 256);
                }
            });

            recorder.set_kernel(self.join);
            recorder.set_barrier_before();
            recorder.set_buffer_read(0, self.branches, 0);
            recorder.set_buffer_write(1, self.joined, 0);
            recorder.bind_u32(2, NUM_ACTIVE_VALUES, 1, NUM_TOTAL_VALUES);
            recorder.dispatch_1d(NUM_TOTAL_VALUES as usize, 256);
        }
    }
}

#[test]
fn test_disjoint_replay_fork_join_with_dynamic_regions_and_reused_storage() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let prepare = CompiledKernel::new(&device, SOURCE, "prepare");
    let branch = CompiledKernel::new(&device, SOURCE, "branch");
    let join = CompiledKernel::new(&device, SOURCE, "join");
    let input_values = (0..NUM_TOTAL_VALUES).map(|index| index * 17 + 3).collect::<Vec<_>>();
    let input = Buffer::from_slice(&device, &input_values);
    let prepared = Buffer::from_slice(&device, &vec![CANARY; NUM_TOTAL_VALUES as usize]);
    let branches = Buffer::from_slice(&device, &vec![CANARY; NUM_TOTAL_VALUES as usize + 2]);
    let joined = Buffer::from_slice(&device, &vec![CANARY; NUM_TOTAL_VALUES as usize + 2]);
    let mut builder = stream.create_replay_program();
    builder.record(DisjointInvocation {
        prepare: &prepare,
        branch: &branch,
        join: &join,
        input: &input,
        prepared: &prepared,
        branches: &branches,
        joined: &joined,
    });
    let program = builder.build();
    assert_eq!(program.stats().command_count, 8);

    for (seed, (num_active, split)) in [
        (64, 0),
        (64, 64),
        (64, 1),
        (NUM_TOTAL_VALUES, NUM_TOTAL_VALUES / 2),
        (257, 17),
        (1, 0),
        (1, 1),
        (NUM_TOTAL_VALUES, NUM_TOTAL_VALUES),
    ]
    .into_iter()
    .enumerate()
    {
        let seed = seed as u32;
        let mut expected_prepared = vec![CANARY; NUM_TOTAL_VALUES as usize];
        let mut expected_branches = vec![CANARY; NUM_TOTAL_VALUES as usize + 2];
        let mut expected_joined = expected_branches.clone();
        prepared.write_typed(0, &expected_prepared);
        branches.write_typed(0, &expected_branches);
        joined.write_typed(0, &expected_joined);
        for layer in 0..2 {
            for index in 0..num_active as usize {
                let value = if layer == 0 {
                    input_values[index]
                } else {
                    expected_joined[index + 1]
                };
                expected_prepared[index] = value * 3 + seed + layer;
                expected_branches[index + 1] = expected_prepared[index] + if index < split as usize { 11 } else { 29 };
            }
            for index in 0..num_active as usize {
                expected_joined[index + 1] =
                    expected_branches[index + 1] + expected_branches[num_active as usize - index];
            }
        }

        stream
            .submit_replay_with_arguments(
                &program,
                &ReplayArguments::new()
                    .with_u32(NUM_ACTIVE_VALUES, num_active)
                    .with_u32(SPLIT, split)
                    .with_u32(SEED, seed),
            )
            .wait();
        assert_eq!(
            prepared.read_typed::<u32>(0, expected_prepared.len()),
            expected_prepared
        );
        assert_eq!(
            branches.read_typed::<u32>(0, expected_branches.len()),
            expected_branches
        );
        assert_eq!(joined.read_typed::<u32>(0, expected_joined.len()), expected_joined);
    }
}
