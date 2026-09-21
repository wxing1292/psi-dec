use std::cell::RefCell;
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::MTLCompileOptions;
use objc2_metal::MTLComputePipelineDescriptor;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLDevice;
use objc2_metal::MTLLanguageVersion;
use objc2_metal::MTLLibrary;
use objc2_metal::MTLPipelineOption;

use crate::metal::Device;

#[derive(Debug)]
pub struct CompiledKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CompilerKey {
    device: usize,
    language_version: MTLLanguageVersion,
}

struct CompiledLibrary {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

thread_local! {
    // Compare complete source text. Hash collisions must not select another shader.
    static LIBRARY_CACHE: RefCell<HashMap<CompilerKey, HashMap<String, CompiledLibrary>>> =
        RefCell::new(HashMap::new());
}

impl CompiledKernel {
    pub fn new(device: &Device, source: &str, function_name: &str) -> Self {
        Self::compile(device, source, function_name, MTLCompileOptions::new())
    }

    pub fn new_tensor_ops(device: &Device, source: &str, function_name: &str) -> Self {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);
        Self::compile(device, source, function_name, options)
    }

    fn compile(device: &Device, source: &str, function_name: &str, options: Retained<MTLCompileOptions>) -> Self {
        let key = CompilerKey {
            device: device.as_raw() as *const _ as *const () as usize,
            language_version: options.languageVersion(),
        };
        let pipeline = LIBRARY_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let libraries = cache.entry(key).or_default();
            if let Some(library) = libraries.get_mut(source) {
                return library.pipeline(device, function_name);
            }
            let mut library = CompiledLibrary {
                library: compile_library(device, source, &options),
                pipelines: HashMap::new(),
            };
            let pipeline = library.pipeline(device, function_name);
            libraries.insert(source.to_owned(), library);
            pipeline
        });
        Self { pipeline }
    }

    pub fn as_raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.pipeline
    }

    pub fn as_raw_retained(&self) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        self.pipeline.clone()
    }

    pub fn max_total_threads_per_threadblock(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
    }

    pub fn thread_execution_width(&self) -> usize {
        self.pipeline.threadExecutionWidth()
    }

    pub fn static_threadblock_memory_length(&self) -> usize {
        self.pipeline.staticThreadgroupMemoryLength()
    }
}

impl CompiledLibrary {
    fn pipeline(
        &mut self,
        device: &Device,
        function_name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        if let Some(pipeline) = self.pipelines.get(function_name) {
            return pipeline.clone();
        }
        let function = self
            .library
            .newFunctionWithName(&NSString::from_str(function_name))
            .expect("Metal function lookup failed");
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setSupportIndirectCommandBuffers(true);
        let pipeline = device
            .as_raw()
            .newComputePipelineStateWithDescriptor_options_reflection_error(&descriptor, MTLPipelineOption::None, None)
            .expect("Metal compute pipeline creation failed");
        self.pipelines.insert(function_name.to_owned(), pipeline.clone());
        pipeline
    }
}

fn compile_library(
    device: &Device,
    source: &str,
    options: &MTLCompileOptions,
) -> Retained<ProtocolObject<dyn MTLLibrary>> {
    // Match MLX JIT compilation so MLX-derived qdot/math kernels keep parity.
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    device
        .as_raw()
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(options))
        .expect("Metal library compile failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Buffer;
    use crate::metal::CommandRecorder;
    use crate::metal::Operator;
    use crate::metal::Stream;

    #[test]
    fn test_compile_reuses_library_and_pipeline() {
        let device = Device::system_default();
        let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void add(device uint* v [[buffer(0)]], \
                      uint i [[thread_position_in_grid]]) { v[i] += 1; }\nkernel void multiply(device uint* v \
                      [[buffer(0)]], uint i [[thread_position_in_grid]]) { v[i] *= 2; }";
        let compile = |source: &str, name: &str| {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version3_1);
            CompiledKernel::compile(&device, source, name, options)
        };
        let add = compile(source, "add");
        let key = CompilerKey {
            device: device.as_raw() as *const _ as *const () as usize,
            language_version: MTLLanguageVersion::Version3_1,
        };
        let library = LIBRARY_CACHE.with(|cache| cache.borrow()[&key][source].library.clone());
        let multiply = compile(source, "multiply");
        let add_again = compile(source, "add");
        assert!(std::ptr::eq(add.as_raw(), add_again.as_raw()));
        LIBRARY_CACHE.with(|cache| {
            let cache = cache.borrow();
            let cached = &cache[&key][source];
            assert!(std::ptr::eq(&*library, &*cached.library));
            assert_eq!(cached.pipelines.len(), 2);
        });
        let changed = compile(&source.replace("+= 1", "+= 3"), "add");
        let tensor_ops = CompiledKernel::new_tensor_ops(&device, source, "multiply");
        assert!(!std::ptr::eq(add.as_raw(), changed.as_raw()));
        assert!(!std::ptr::eq(multiply.as_raw(), tensor_ops.as_raw()));
        let stream = Stream::new(&device);
        let values = Buffer::from_slice(&device, &[1_u32, 2, 3]);
        let mut recorder = stream.create_replay_program();
        for kernel in [&add_again, &multiply, &changed, &tensor_ops] {
            recorder.record_with_barrier_before(Invocation {
                kernel,
                values: &values,
            });
        }
        stream.submit_replay(&recorder.build()).wait();
        assert_eq!(values.read_typed::<u32>(0, 3), [14, 18, 22]);
    }

    struct Invocation<'a> {
        kernel: &'a CompiledKernel,
        values: &'a Buffer,
    }

    impl Operator for Invocation<'_> {
        fn record(self, recorder: &CommandRecorder<'_>) {
            recorder.set_kernel(self.kernel);
            recorder.set_buffer_read_write(0, self.values, 0);
            recorder.dispatch_threadblocks((3, 1, 1), (1, 1, 1));
        }
    }
}
