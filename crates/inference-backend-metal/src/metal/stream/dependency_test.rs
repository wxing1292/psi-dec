use objc2_metal::MTLResourceUsage;

use super::CommandDependencyTracker;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;
use crate::metal::stream::operation::CommandMetadata;
use crate::metal::stream::operation::record_operator;
use crate::metal::stream::parameter::CommandParameterLayoutBuilder;

const SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void noop(device uint* values [[buffer(0)]], uint id [[thread_position_in_grid]]) {
        values[id] += 1;
    }
"#;

struct Recording<F>(F);

impl<F: FnOnce(&CommandRecorder<'_>)> Operator for Recording<F> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        (self.0)(recorder);
    }
}

fn commands(record: impl FnOnce(&CommandRecorder<'_>)) -> Vec<CommandMetadata> {
    record_operator(&CommandParameterLayoutBuilder::default(), Recording(record)).commands
}

fn record_access(
    recorder: &CommandRecorder<'_>,
    kernel: &CompiledKernel,
    bindings: &[(&Buffer, MTLResourceUsage)],
    barrier_before: bool,
) {
    recorder.set_kernel(kernel);
    if barrier_before {
        recorder.set_barrier_before();
    }
    for (index, &(buffer, usage)) in bindings.iter().enumerate() {
        if usage == MTLResourceUsage::Read {
            recorder.set_buffer_read(index, buffer, 0);
        } else if usage == MTLResourceUsage::Write {
            recorder.set_buffer_write(index, buffer, 0);
        } else {
            recorder.set_buffer_read_write(index, buffer, 0);
        }
    }
    recorder.dispatch_1d(1, 1);
}

fn barriers(commands: &[CommandMetadata]) -> Vec<bool> {
    let mut tracker = CommandDependencyTracker::default();
    commands.iter().map(|command| tracker.barrier_before(command)).collect()
}

#[test]
fn test_disjoint_commands_preserve_entry_and_join_dependencies() {
    let device = Device::system_default();
    let kernel = CompiledKernel::new(&device, SOURCE, "noop");
    let prepared = Buffer::from_slice(&device, &[0_u32; 2]);
    let output = Buffer::from_slice(&device, &[0_u32; 2]);
    let recorded = commands(|recorder| {
        record_access(recorder, &kernel, &[(&prepared, MTLResourceUsage::Write)], false);
        recorder.record_disjoint_buffers(&[&output], || {
            for _ in 0..2 {
                record_access(
                    recorder,
                    &kernel,
                    &[(&prepared, MTLResourceUsage::Read), (&output, MTLResourceUsage::Write)],
                    false,
                );
            }
        });
        record_access(recorder, &kernel, &[(&output, MTLResourceUsage::Read)], false);
    });

    assert_eq!(barriers(&recorded), vec![false, true, false, true]);
}

#[test]
fn test_disjoint_scope_keeps_each_partition_access() {
    let device = Device::system_default();
    let kernel = CompiledKernel::new(&device, SOURCE, "noop");
    let values = Buffer::from_slice(&device, &[0_u32; 2]);
    for usages in [
        [MTLResourceUsage::Write, MTLResourceUsage::Read],
        [MTLResourceUsage::Read, MTLResourceUsage::Write],
    ] {
        let recorded = commands(|recorder| {
            recorder.record_disjoint_buffers(&[&values], || {
                for usage in usages {
                    record_access(recorder, &kernel, &[(&values, usage)], false);
                }
            });
            record_access(recorder, &kernel, &[(&values, MTLResourceUsage::Read)], false);
        });
        assert_eq!(barriers(&recorded), vec![false, false, true]);

        // Revisit the same recorded domain. A later disjoint access must not
        // replace its earlier read/write history or inherit another domain's writes.
        for (index, usage) in usages.into_iter().enumerate() {
            let mut tracker = CommandDependencyTracker::default();
            assert!(!tracker.barrier_before(&recorded[0]));
            assert!(!tracker.barrier_before(&recorded[1]));
            assert_eq!(
                tracker.barrier_before(&recorded[index]),
                usage == MTLResourceUsage::Write
            );
        }
    }
}

#[test]
fn test_disjoint_scope_does_not_cross_recording_boundaries() {
    let device = Device::system_default();
    let kernel = CompiledKernel::new(&device, SOURCE, "noop");
    let values = Buffer::from_slice(&device, &[0_u32; 2]);
    let recorded = commands(|recorder| {
        for _ in 0..2 {
            recorder.record_disjoint_buffers(&[&values], || {
                for _ in 0..2 {
                    record_access(recorder, &kernel, &[(&values, MTLResourceUsage::Write)], false);
                }
            });
        }
    });

    assert_eq!(barriers(&recorded), vec![false, false, true, false]);
}

#[test]
fn test_disjoint_scope_preserves_unselected_hazards_and_explicit_barriers() {
    let device = Device::system_default();
    let kernel = CompiledKernel::new(&device, SOURCE, "noop");
    let selected = Buffer::from_slice(&device, &[0_u32; 2]);
    let unselected = Buffer::from_slice(&device, &[0_u32; 2]);
    for explicit_barrier in [false, true] {
        let recorded = commands(|recorder| {
            recorder.record_disjoint_buffers(&[&selected], || {
                for command_index in 0..2 {
                    let mut bindings = vec![(&selected, MTLResourceUsage::Write)];
                    if !explicit_barrier {
                        bindings.push((&unselected, MTLResourceUsage::Read | MTLResourceUsage::Write));
                    }
                    record_access(recorder, &kernel, &bindings, explicit_barrier && command_index == 1);
                }
            });
        });
        assert_eq!(barriers(&recorded), vec![false, true]);
    }
}
