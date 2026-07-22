use inference_backend_metal::components::REJECTION_NUM_ACTIVE_THREADS_KEY;
use inference_backend_metal::components::REJECTION_NUM_DRAFT_DISTRIBUTIONS_KEY;
use inference_backend_metal::components::REJECTION_NUM_TARGET_DISTRIBUTIONS_KEY;
use inference_backend_metal::components::SAMPLING_NUM_THREADS_PER_THREADBLOCK;
use inference_backend_metal::components::SparseRejectionSampleBuffers;
use inference_backend_metal::components::SparseRejectionSampleKernel;
use inference_backend_metal::components::SparseRejectionSampleShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::sampling::SparseRejectionSamplingReqParams;
use inference_executor_core::sampling::SparseRejectionSamplingShape;

use crate::def::replay_op::ReplayOp;
use crate::sampling::RuntimeParamRows;

#[derive(Clone, Copy)]
pub struct SparseRejectionSamplingOutput<'a> {
    pub flat_accepted_token_ids: &'a Buffer,
    pub flat_accepted_probs: &'a Buffer,
    pub num_accepted_tokens: &'a Buffer,
    pub sampled_token_ids: &'a Buffer,
    pub sampled_token_probs: &'a Buffer,
}

pub struct SparseRejectionSampling {
    kernel: SparseRejectionSampleKernel,
    max_shape: SparseRejectionSamplingShape,
    runtime_params: Buffer,
    runtime_param_rows: RuntimeParamRows,
}

impl SparseRejectionSampling {
    pub fn new(device: &Device, max_shape: SparseRejectionSamplingShape) -> Self {
        max_shape.validate();
        Self {
            kernel: SparseRejectionSampleKernel::new(device),
            max_shape,
            runtime_params: Buffer::new_zeroed_elements(
                device,
                (max_shape.num_total_reqs as usize)
                    .checked_mul(4)
                    .expect("sparse rejection runtime parameter capacity must fit usize"),
                Dtype::Uint32,
            ),
            runtime_param_rows: RuntimeParamRows::default(),
        }
    }

    pub fn set_runtime_params(&self, params: &[SparseRejectionSamplingReqParams]) {
        assert!(
            params.len() <= self.max_shape.num_total_reqs as usize,
            "sparse rejection sampling runtime request params exceed total requests"
        );
        for (req_index, params) in params.iter().enumerate() {
            assert!(
                params.top_k > 0 && params.top_k <= self.max_shape.top_k,
                "sparse rejection sampling request top_k must fit capacity"
            );
            self.runtime_params.write_typed(
                req_index * 4,
                &[
                    params.seed,
                    params.sample_position,
                    params.top_k,
                    0, // Padding keeps each request entry 16-byte aligned.
                ],
            );
        }
        self.runtime_param_rows.set(params.len(), "sparse rejection sampling");
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: SparseRejectionSamplingShape,
        inputs: SparseRejectionSamplingInputs<'a>,
        output: SparseRejectionSamplingOutput<'a>,
    ) {
        self.validate_input(shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kernel.invoke_replay(
            component_shape(shape),
            SparseRejectionSampleBuffers {
                target_distribution_token_ids: inputs.target_distribution_token_ids,
                target_distribution_probs: inputs.target_distribution_probs,
                draft_distribution_token_ids: inputs.draft_distribution_token_ids,
                draft_distribution_probs: inputs.draft_distribution_probs,
                flat_draft_token_ids: inputs.flat_draft_token_ids,
                cu_target_distributions: inputs.cu_target_distributions,
                cu_draft_distributions: inputs.cu_draft_distributions,
                flat_draft_distribution_indices: inputs.flat_draft_distribution_indices,
                flat_accepted_token_ids: output.flat_accepted_token_ids,
                flat_accepted_probs: output.flat_accepted_probs,
                num_accepted_tokens: output.num_accepted_tokens,
                sampled_token_ids: output.sampled_token_ids,
                sampled_token_probs: output.sampled_token_probs,
                runtime_params: &self.runtime_params,
            },
        )));
    }

    pub fn add_replay_arguments(&self, shape: SparseRejectionSamplingShape, arguments: &mut ReplayArguments) {
        self.validate_input(shape);
        self.runtime_param_rows
            .consume(shape.num_active_reqs, "sparse rejection sampling");
        if shape.num_total_reqs > 1 {
            let num_active_threads = shape
                .num_active_reqs
                .checked_mul(SAMPLING_NUM_THREADS_PER_THREADBLOCK)
                .expect("sparse rejection active thread count must fit u32");
            let num_total_threads = shape
                .num_total_reqs
                .checked_mul(SAMPLING_NUM_THREADS_PER_THREADBLOCK)
                .expect("sparse rejection total thread count must fit u32");
            assert!(num_active_threads <= num_total_threads);
            assert_eq!(num_active_threads % SAMPLING_NUM_THREADS_PER_THREADBLOCK, 0);
            arguments.set_u32(REJECTION_NUM_ACTIVE_THREADS_KEY, num_active_threads);
        }
        if shape.num_total_target_distributions > 1 {
            assert!(shape.num_active_target_distributions <= shape.num_total_target_distributions);
            arguments.set_u32(
                REJECTION_NUM_TARGET_DISTRIBUTIONS_KEY,
                shape.num_active_target_distributions,
            );
        }
        if shape.num_total_draft_distributions > 0 {
            assert!(shape.num_active_draft_distributions <= shape.num_total_draft_distributions);
            arguments.set_u32(
                REJECTION_NUM_DRAFT_DISTRIBUTIONS_KEY,
                shape.num_active_draft_distributions,
            );
        }
    }

    fn validate_input(&self, shape: SparseRejectionSamplingShape) {
        shape.validate();
        assert!(
            shape.num_total_reqs <= self.max_shape.num_total_reqs,
            "sparse rejection sampling total requests={} exceeds max={}",
            shape.num_total_reqs,
            self.max_shape.num_total_reqs
        );
        assert!(
            shape.num_total_draft_distributions <= self.max_shape.num_total_draft_distributions,
            "sparse rejection sampling total draft distributions={} exceeds max={}",
            shape.num_total_draft_distributions,
            self.max_shape.num_total_draft_distributions
        );
        assert!(
            shape.num_total_target_distributions <= self.max_shape.num_total_target_distributions,
            "sparse rejection sampling total target distributions={} exceeds max={}",
            shape.num_total_target_distributions,
            self.max_shape.num_total_target_distributions
        );
        assert!(
            shape.top_k <= self.max_shape.top_k,
            "sparse rejection sampling top_k={} exceed capacity={}",
            shape.top_k,
            self.max_shape.top_k
        );
        assert_eq!(
            shape.max_target_k, self.max_shape.max_target_k,
            "sparse rejection target distribution slots must match capacity"
        );
        assert_eq!(
            shape.max_draft_k, self.max_shape.max_draft_k,
            "sparse rejection draft distribution slots must match capacity"
        );
    }
}

fn component_shape(shape: SparseRejectionSamplingShape) -> SparseRejectionSampleShape {
    SparseRejectionSampleShape {
        num_total_reqs: shape.num_total_reqs,
        num_total_draft_distributions: shape.num_total_draft_distributions,
        num_total_target_distributions: shape.num_total_target_distributions,
        top_k: shape.top_k,
        max_target_k: shape.max_target_k,
        max_draft_k: shape.max_draft_k,
    }
}

#[derive(Clone, Copy)]
pub struct SparseRejectionSamplingInputs<'a> {
    pub target_distribution_token_ids: &'a Buffer,
    pub target_distribution_probs: &'a Buffer,
    pub draft_distribution_token_ids: &'a Buffer,
    pub draft_distribution_probs: &'a Buffer,
    pub flat_draft_token_ids: &'a Buffer,
    pub cu_target_distributions: &'a Buffer,
    pub cu_draft_distributions: &'a Buffer,
    pub flat_draft_distribution_indices: &'a Buffer,
}
