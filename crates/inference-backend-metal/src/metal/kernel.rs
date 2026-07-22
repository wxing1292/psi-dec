use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::MTLCompileOptions;
use objc2_metal::MTLComputePipelineDescriptor;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLDevice;
use objc2_metal::MTLLibrary;
use objc2_metal::MTLPipelineOption;

use crate::metal::Device;

#[derive(Debug)]
pub struct Kernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct KernelCacheKey {
    device: usize,
    source_hash: u64,
    function_name: String,
}

thread_local! {
    static KERNEL_CACHE: RefCell<HashMap<KernelCacheKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>> =
        RefCell::new(HashMap::new());
}

impl Kernel {
    pub fn new(device: &Device, source: &str, function_name: &str) -> Self {
        let key = KernelCacheKey {
            device: device.as_raw() as *const _ as *const () as usize,
            source_hash: stable_hash(source),
            function_name: function_name.to_string(),
        };
        let pipeline = KERNEL_CACHE.with(|cache| {
            if let Some(pipeline) = cache.borrow().get(&key) {
                return pipeline.clone();
            }

            let library = compile_library(device, source);
            let function = library
                .newFunctionWithName(&NSString::from_str(function_name))
                .expect("Metal function lookup failed");
            let descriptor = MTLComputePipelineDescriptor::new();
            descriptor.setComputeFunction(Some(&function));
            descriptor.setSupportIndirectCommandBuffers(true);
            let pipeline = device
                .as_raw()
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &descriptor,
                    MTLPipelineOption::None,
                    None,
                )
                .expect("Metal compute pipeline creation failed");
            cache.borrow_mut().insert(key, pipeline.clone());
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
}

fn stable_hash(value: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn compile_library(device: &Device, source: &str) -> Retained<ProtocolObject<dyn MTLLibrary>> {
    let options = MTLCompileOptions::new();
    // Match MLX JIT compilation so MLX-derived qdot/math kernels keep parity.
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    device
        .as_raw()
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .expect("Metal library compile failed")
}
