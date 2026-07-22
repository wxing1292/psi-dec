use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLResourceUsage;

use crate::metal::stream::operation::CommandBinding;
use crate::metal::stream::operation::CommandMetadata;

#[derive(Debug, Default)]
pub struct CommandDependencyTracker {
    accesses: Vec<BufferAccess>,
}

impl CommandDependencyTracker {
    pub fn barrier_before(&mut self, command: &CommandMetadata) -> bool {
        let barrier_before = command.barrier_before || self.has_dependency(command);
        if barrier_before {
            self.accesses.clear();
        }
        self.update(command);
        barrier_before
    }

    fn has_dependency(&self, command: &CommandMetadata) -> bool {
        for binding in &command.bindings {
            let Some(CommandBinding::Buffer { buffer, usage, .. }) = binding else {
                continue;
            };
            let current_writes = usage.contains(MTLResourceUsage::Write);
            for access in &self.accesses {
                if Retained::as_ptr(&access.buffer) != Retained::as_ptr(buffer) {
                    continue;
                }
                let previous_wrote = access.usage.contains(MTLResourceUsage::Write);
                if previous_wrote || current_writes {
                    return true;
                }
            }
        }
        false
    }

    fn update(&mut self, command: &CommandMetadata) {
        for binding in &command.bindings {
            let Some(CommandBinding::Buffer { buffer, usage, .. }) = binding else {
                continue;
            };
            if let Some(existing) = self
                .accesses
                .iter_mut()
                .find(|existing| Retained::as_ptr(&existing.buffer) == Retained::as_ptr(buffer))
            {
                existing.usage |= *usage;
                continue;
            }
            self.accesses.push(BufferAccess {
                buffer: buffer.clone(),
                usage: *usage,
            });
        }
    }
}

#[derive(Debug)]
struct BufferAccess {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    usage: MTLResourceUsage,
}
