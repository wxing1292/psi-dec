use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::sampling::SparseRejectionSamplingReqParams;
use inference_executor_core::sampling::SparseRejectionSamplingShape;
use inference_executor_core::sampling::TopKSamplingShape;

use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::replay::ReplayComponent;
use crate::sampling::rejection_sampling::SparseRejectionSampling;
use crate::sampling::rejection_sampling::SparseRejectionSamplingInputs;
use crate::sampling::rejection_sampling::SparseRejectionSamplingOutput;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingInputs;
use crate::sampling::top_k_sampling::TopKSamplingSparseDistributionOutput;

pub struct Qwen35RejectionSampler {
    sparse_sampler: SparseRejectionSampling,
    max_requests: usize,
    max_num_spec_tokens: usize,
    max_k: u32,
    cu_target_distributions: Buffer,
    cu_draft_distributions: Buffer,
    flat_draft_token_ids: Buffer,
    flat_draft_distribution_indices: Buffer,
    flat_accepted_token_ids: Buffer,
    flat_accepted_probs: Buffer,
    num_accepted_tokens: Buffer,
    sampled_token_ids: Buffer,
    sampled_token_probs: Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen35RejectionSamplingInput<'a> {
    pub num_active_decode_reqs: usize,
    pub num_decode_req_capacity: usize,
    pub num_target_distribution_capacity: usize,
    pub num_active_draft_distributions: usize,
    pub num_draft_distribution_capacity: usize,
    pub top_k: u32,
    pub target_token_ids: &'a Buffer,
    pub target_probs: &'a Buffer,
    pub draft_token_ids: &'a Buffer,
    pub draft_probs: &'a Buffer,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Qwen35PreparedRejection {
    pub decode_req_indices: Vec<usize>,
    pub num_active_draft_distributions: usize,
}

impl Qwen35RejectionSamplingInput<'_> {
    fn num_active_target_distributions(self) -> usize {
        self.num_active_draft_distributions
            .checked_add(self.num_active_decode_reqs)
            .expect("qwen3.5 rejection target-distribution count overflow")
    }
}

pub struct Qwen35RejectionResults {
    flat_accepted_token_ids: Vec<i32>,
    flat_accepted_probs: Vec<f32>,
    num_accepted_tokens: Vec<u32>,
    sampled_token_ids: Vec<i32>,
    sampled_token_probs: Vec<f32>,
}

impl Qwen35RejectionResults {
    pub fn num_accepted_tokens(&self, decode_req_index: usize) -> usize {
        self.num_accepted_tokens[decode_req_index]
            .try_into()
            .expect("qwen3.5 accepted-token count must fit host usize")
    }

    pub fn accepted_token_ids(&self, flat_draft_index: usize, num_tokens: usize) -> &[i32] {
        &self.flat_accepted_token_ids[flat_draft_index..flat_draft_index + num_tokens]
    }

    pub fn accepted_probs(&self, flat_draft_index: usize, num_tokens: usize) -> &[f32] {
        &self.flat_accepted_probs[flat_draft_index..flat_draft_index + num_tokens]
    }

    pub fn sampled_token_id(&self, decode_req_index: usize) -> i32 {
        self.sampled_token_ids[decode_req_index]
    }

    pub fn sampled_prob(&self, decode_req_index: usize) -> f32 {
        self.sampled_token_probs[decode_req_index]
    }
}

impl Qwen35PreparedRejection {
    pub fn num_active_decode_reqs(&self) -> usize {
        self.decode_req_indices.len()
    }

    pub fn num_active_target_distributions(&self) -> usize {
        self.num_active_draft_distributions
            .checked_add(self.num_active_decode_reqs())
            .expect("qwen3.5 rejection target-distribution count overflow")
    }
}

impl Qwen35RejectionSampler {
    fn validate_input(&self, input: Qwen35RejectionSamplingInput<'_>) {
        assert!(input.top_k > 0, "qwen3.5 rejection top_k is empty");
        assert!(
            input.num_active_target_distributions() <= input.num_target_distribution_capacity,
            "qwen3.5 active target distributions exceed replay capacity"
        );
    }

    pub fn new(device: &Device, max_num_spec_tokens: usize, max_requests: usize, max_k: u32) -> Self {
        assert!(max_requests > 0, "qwen3.5 rejection sampler requires requests");
        let max_draft_distributions: u32 = max_requests
            .checked_mul(max_num_spec_tokens.max(1))
            .expect("qwen3.5 rejection draft distributions overflow")
            .try_into()
            .expect("qwen3.5 rejection draft-distribution count must fit u32");
        let max_requests_u32: u32 = max_requests
            .try_into()
            .expect("qwen3.5 rejection request capacity must fit u32");
        let max_target_distributions = max_draft_distributions
            .checked_add(max_requests_u32)
            .expect("qwen3.5 rejection target-distribution count must fit u32");
        let max_shape = SparseRejectionSamplingShape {
            num_active_reqs: max_requests_u32,
            num_total_reqs: max_requests_u32,
            num_active_draft_distributions: max_draft_distributions,
            num_total_draft_distributions: max_draft_distributions,
            num_active_target_distributions: max_target_distributions,
            num_total_target_distributions: max_target_distributions,
            top_k: max_k,
            max_target_k: max_k,
            max_draft_k: max_k,
        };
        Self {
            sparse_sampler: SparseRejectionSampling::new(device, max_shape),
            max_requests,
            max_num_spec_tokens,
            max_k,
            cu_target_distributions: Buffer::new_zeroed_elements(
                device,
                max_requests
                    .checked_add(1)
                    .expect("qwen3.5 rejection cumulative target length overflow"),
                Dtype::Uint32,
            ),
            cu_draft_distributions: Buffer::new_zeroed_elements(
                device,
                max_requests
                    .checked_add(1)
                    .expect("qwen3.5 rejection cumulative draft length overflow"),
                Dtype::Uint32,
            ),
            flat_draft_token_ids: Buffer::new_zeroed_elements(device, max_draft_distributions as usize, Dtype::Int32),
            flat_draft_distribution_indices: Buffer::new_zeroed_elements(
                device,
                max_draft_distributions as usize,
                Dtype::Uint32,
            ),
            flat_accepted_token_ids: Buffer::new_zeroed_elements(
                device,
                max_draft_distributions as usize,
                Dtype::Int32,
            ),
            flat_accepted_probs: Buffer::new_zeroed_elements(device, max_draft_distributions as usize, Dtype::Float32),
            num_accepted_tokens: Buffer::new_zeroed_elements(device, max_requests, Dtype::Uint32),
            sampled_token_ids: Buffer::new_zeroed_elements(device, max_requests, Dtype::Int32),
            sampled_token_probs: Buffer::new_zeroed_elements(device, max_requests, Dtype::Float32),
        }
    }

    pub fn prepare_inputs(
        &self,
        microbatch: &Qwen35Microbatch,
        flat_draft_distribution_indices: &[u32],
    ) -> Qwen35PreparedRejection {
        let decode_req_indices = (0..microbatch.num_reqs())
            .filter(|&req_index| microbatch.is_decode_req(req_index))
            .collect::<Vec<_>>();
        let num_decode_reqs = decode_req_indices.len();
        assert!(num_decode_reqs > 0, "qwen3.5 rejection requires decode requests");
        assert!(
            num_decode_reqs <= self.max_requests,
            "qwen3.5 rejection requests exceed sampler capacity"
        );
        let num_draft_distributions = decode_req_indices
            .iter()
            .map(|&req_index| microbatch.num_spec_tokens(req_index) as usize)
            .sum::<usize>();
        assert_eq!(
            flat_draft_distribution_indices.len(),
            num_draft_distributions,
            "qwen3.5 rejection draft-distribution indices must match flat drafts"
        );
        let draft_capacity = self
            .max_requests
            .checked_mul(self.max_num_spec_tokens)
            .expect("qwen3.5 rejection draft-distribution capacity overflow");
        assert!(
            flat_draft_distribution_indices.iter().all(|&index| {
                usize::try_from(index).expect("qwen3.5 draft-distribution index must fit host usize") < draft_capacity
            }),
            "qwen3.5 rejection draft-distribution index exceeds sampler capacity"
        );
        let mut cu_target = Vec::with_capacity(num_decode_reqs + 1);
        let mut cu_draft = Vec::with_capacity(num_decode_reqs + 1);
        let mut flat_draft_tokens = Vec::with_capacity(num_draft_distributions);
        cu_target.push(0_u32);
        cu_draft.push(0_u32);
        for &req_index in &decode_req_indices {
            let draft_len = microbatch.num_spec_tokens(req_index) as usize;
            assert!(
                draft_len <= self.max_num_spec_tokens,
                "qwen3.5 rejection num_spec_tokens exceeds sampler capacity"
            );
            let q_end = microbatch.cu_tokens()[req_index + 1] as usize;
            let q_start = q_end
                .checked_sub(draft_len)
                .expect("qwen3.5 rejection draft suffix exceeds request flat num_tokens");
            flat_draft_tokens.extend_from_slice(&microbatch.flat_token_ids()[q_start..q_end]);
            cu_draft.push(
                flat_draft_tokens
                    .len()
                    .try_into()
                    .expect("qwen3.5 rejection cumulative draft count must fit u32"),
            );
            cu_target.push(
                flat_draft_tokens
                    .len()
                    .checked_add(cu_target.len())
                    .and_then(|count| count.try_into().ok())
                    .expect("qwen3.5 rejection cumulative target count must fit u32"),
            );
        }
        self.cu_target_distributions.write_typed(0, &cu_target);
        self.cu_draft_distributions.write_typed(0, &cu_draft);
        self.flat_draft_token_ids.write_typed(0, &flat_draft_tokens);
        self.flat_draft_distribution_indices
            .write_typed(0, flat_draft_distribution_indices);
        Qwen35PreparedRejection {
            decode_req_indices,
            num_active_draft_distributions: num_draft_distributions,
        }
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, input: Qwen35RejectionSamplingInput<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_input(input);
        self.sparse_sampler.record(
            recorder,
            self.sampling_shape(input),
            SparseRejectionSamplingInputs {
                target_distribution_token_ids: input.target_token_ids,
                target_distribution_probs: input.target_probs,
                draft_distribution_token_ids: input.draft_token_ids,
                draft_distribution_probs: input.draft_probs,
                flat_draft_token_ids: &self.flat_draft_token_ids,
                cu_target_distributions: &self.cu_target_distributions,
                cu_draft_distributions: &self.cu_draft_distributions,
                flat_draft_distribution_indices: &self.flat_draft_distribution_indices,
            },
            SparseRejectionSamplingOutput {
                flat_accepted_token_ids: &self.flat_accepted_token_ids,
                flat_accepted_probs: &self.flat_accepted_probs,
                num_accepted_tokens: &self.num_accepted_tokens,
                sampled_token_ids: &self.sampled_token_ids,
                sampled_token_probs: &self.sampled_token_probs,
            },
        );
    }

    pub fn add_replay_arguments(&self, input: Qwen35RejectionSamplingInput<'_>, arguments: &mut ReplayArguments) {
        self.validate_input(input);
        self.sparse_sampler
            .add_replay_arguments(self.sampling_shape(input), arguments);
    }

    fn sampling_shape(&self, input: Qwen35RejectionSamplingInput<'_>) -> SparseRejectionSamplingShape {
        SparseRejectionSamplingShape {
            num_active_reqs: input
                .num_active_decode_reqs
                .try_into()
                .expect("qwen3.5 active decode request count must fit u32"),
            num_total_reqs: input
                .num_decode_req_capacity
                .try_into()
                .expect("qwen3.5 decode request capacity must fit u32"),
            num_active_draft_distributions: input
                .num_active_draft_distributions
                .try_into()
                .expect("qwen3.5 active draft-distribution count must fit u32"),
            num_total_draft_distributions: input
                .num_draft_distribution_capacity
                .try_into()
                .expect("qwen3.5 draft-distribution capacity must fit u32"),
            num_active_target_distributions: input
                .num_active_target_distributions()
                .try_into()
                .expect("qwen3.5 active target-distribution count must fit u32"),
            num_total_target_distributions: input
                .num_target_distribution_capacity
                .try_into()
                .expect("qwen3.5 target-distribution capacity must fit u32"),
            top_k: input.top_k,
            max_target_k: self.max_k,
            max_draft_k: self.max_k,
        }
    }

    pub fn set_runtime_params(&self, params: &[SparseRejectionSamplingReqParams]) {
        self.sparse_sampler.set_runtime_params(params);
    }

    pub fn read_results(&self, num_decode_reqs: usize, num_draft_distributions: usize) -> Qwen35RejectionResults {
        debug_assert!(num_decode_reqs <= self.max_requests);
        debug_assert!(num_draft_distributions <= self.max_requests * self.max_num_spec_tokens);
        Qwen35RejectionResults {
            flat_accepted_token_ids: self
                .flat_accepted_token_ids
                .read_typed::<i32>(0, num_draft_distributions),
            flat_accepted_probs: self.flat_accepted_probs.read_typed::<f32>(0, num_draft_distributions),
            num_accepted_tokens: self.num_accepted_tokens.read_typed::<u32>(0, num_decode_reqs),
            sampled_token_ids: self.sampled_token_ids.read_typed::<i32>(0, num_decode_reqs),
            sampled_token_probs: self.sampled_token_probs.read_typed::<f32>(0, num_decode_reqs),
        }
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;
    use inference_executor_core::attn::gdn::state::GDNStateTxn;
    use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
    use inference_executor_core::sampling::SamplerConfig;

    use super::Qwen35RejectionSampler;

    #[test]
    fn test_ragged_inputs() {
        let device = Device::system_default();
        let sampler = Qwen35RejectionSampler::new(&device, 3, 4, 4);
        let batch = Qwen35Microbatch::new(
            vec![0, 2],
            vec![0, 0],
            vec![5, 8],
            vec![10, 11, 12, 20, 21],
            vec![0, 3, 5],
            vec![GDNStateTxn::new(5, 3, 2), GDNStateTxn::new(8, 2, 1)],
            vec![Vec::new(), Vec::new()],
            vec![SamplerConfig::default(), SamplerConfig::default()],
            vec![true; 5],
        );

        let prepared = sampler.prepare_inputs(&batch, &[7, 8, 1]);
        assert_eq!(prepared.decode_req_indices, vec![0, 1]);
        assert_eq!(prepared.num_active_draft_distributions, 3);
        assert_eq!(prepared.num_active_target_distributions(), 5);
        assert_eq!(sampler.cu_draft_distributions.read_typed::<u32>(0, 3), vec![0, 2, 3]);
        assert_eq!(sampler.cu_target_distributions.read_typed::<u32>(0, 3), vec![0, 3, 5]);
        assert_eq!(sampler.flat_draft_token_ids.read_typed::<i32>(0, 3), vec![11, 12, 21]);
        assert_eq!(
            sampler.flat_draft_distribution_indices.read_typed::<u32>(0, 3),
            vec![7, 8, 1]
        );
    }

    #[test]
    fn test_mixed_inputs() {
        let device = Device::system_default();
        let sampler = Qwen35RejectionSampler::new(&device, 2, 4, 4);
        let batch = Qwen35Microbatch::new(
            vec![0, 1, 2],
            vec![0, 0, 0],
            vec![3, 7, 9],
            vec![10, 11, 20, 30, 31, 32],
            vec![0, 2, 3, 6],
            vec![
                GDNStateTxn::new(3, 2, 0),
                GDNStateTxn::new(7, 1, 0),
                GDNStateTxn::new(9, 3, 2),
            ],
            vec![Vec::new(), Vec::new(), Vec::new()],
            vec![SamplerConfig::default(); 3],
            vec![false, false, true, true, true, true],
        );

        let prepared = sampler.prepare_inputs(&batch, &[5, 6]);

        assert_eq!(prepared.decode_req_indices, vec![1, 2]);
        assert_eq!(prepared.num_active_draft_distributions, 2);
        assert_eq!(prepared.num_active_target_distributions(), 4);
        assert_eq!(sampler.cu_draft_distributions.read_typed::<u32>(0, 3), vec![0, 0, 2]);
        assert_eq!(sampler.cu_target_distributions.read_typed::<u32>(0, 3), vec![0, 1, 4]);
        assert_eq!(sampler.flat_draft_token_ids.read_typed::<i32>(0, 2), vec![31, 32]);
    }

    #[test]
    fn test_zero_drafts() {
        let device = Device::system_default();
        let sampler = Qwen35RejectionSampler::new(&device, 2, 4, 4);
        let batch = Qwen35Microbatch::new(
            vec![0, 1],
            vec![0, 0],
            vec![3, 7],
            vec![10, 11, 20],
            vec![0, 2, 3],
            vec![GDNStateTxn::new(3, 2, 0), GDNStateTxn::new(7, 1, 0)],
            vec![Vec::new(), Vec::new()],
            vec![SamplerConfig::default(); 2],
            vec![false, true, true],
        );

        let prepared = sampler.prepare_inputs(&batch, &[]);

        assert_eq!(prepared.decode_req_indices, vec![0, 1]);
        assert_eq!(prepared.num_active_draft_distributions, 0);
        assert_eq!(prepared.num_active_target_distributions(), 2);
        assert_eq!(sampler.cu_draft_distributions.read_typed::<u32>(0, 3), vec![0, 0, 0]);
        assert_eq!(sampler.cu_target_distributions.read_typed::<u32>(0, 3), vec![0, 1, 2]);
    }

    #[test]
    fn test_result_prefixes() {
        let device = Device::system_default();
        let sampler = Qwen35RejectionSampler::new(&device, 2, 4, 4);
        sampler.flat_accepted_token_ids.write_typed(0, &[11_i32, 12, 21]);
        sampler.flat_accepted_probs.write_typed(0, &[0.1_f32, 0.2, 0.3]);
        sampler.num_accepted_tokens.write_typed(0, &[2_u32, 1]);
        sampler.sampled_token_ids.write_typed(0, &[13_i32, 22]);
        sampler.sampled_token_probs.write_typed(0, &[0.4_f32, 0.5]);

        let results = sampler.read_results(2, 3);

        assert_eq!(results.num_accepted_tokens(0), 2);
        assert_eq!(results.num_accepted_tokens(1), 1);
        assert_eq!(results.accepted_token_ids(0, 2), &[11, 12]);
        assert_eq!(results.accepted_token_ids(2, 1), &[21]);
        assert_eq!(results.accepted_probs(0, 3), &[0.1, 0.2, 0.3]);
        assert_eq!(results.sampled_token_id(0), 13);
        assert_eq!(results.sampled_token_id(1), 22);
        assert_eq!(results.sampled_prob(0), 0.4);
        assert_eq!(results.sampled_prob(1), 0.5);
    }
}

pub struct RejectionSampling {
    sampler: Rc<TopKSampling>,
    rejector: Rc<Qwen35RejectionSampler>,
}

impl RejectionSampling {
    pub fn new(sampler: Rc<TopKSampling>, rejector: Rc<Qwen35RejectionSampler>) -> Self {
        Self { sampler, rejector }
    }

    pub fn rejector(&self) -> &Rc<Qwen35RejectionSampler> {
        &self.rejector
    }
}

#[derive(Clone, Copy)]
pub struct RejectionSamplingInput<'a> {
    pub target_shape: TopKSamplingShape,
    pub logits: &'a Buffer,
    pub target_sparse: TopKSamplingSparseDistributionOutput<'a>,
    pub rejection: Qwen35RejectionSamplingInput<'a>,
}

impl ReplayComponent for RejectionSampling {
    type Key = Qwen35TargetRejectionReplayKey;
    type Input<'a> = RejectionSamplingInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen35TargetRejectionReplayKey {
            num_decode_req_capacity: input.rejection.num_decode_req_capacity,
            num_target_distribution_capacity: input.rejection.num_target_distribution_capacity,
            num_draft_distribution_capacity: input.rejection.num_draft_distribution_capacity,
            top_k: input.rejection.top_k,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.sampler.record_sparse_distribution_bf16(
            recorder,
            input.target_shape,
            TopKSamplingInputs {
                logits: input.logits,
                logits_offset_bytes: 0,
            },
            input.target_sparse,
        );
        self.rejector.record(recorder, input.rejection);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35TargetRejectionReplayKey {
    num_decode_req_capacity: usize,
    num_target_distribution_capacity: usize,
    num_draft_distribution_capacity: usize,
    top_k: u32,
}

impl Qwen35TargetRejectionReplayKey {
    pub fn new(
        num_decode_req_capacity: usize,
        num_target_distribution_capacity: usize,
        num_draft_distribution_capacity: usize,
        top_k: u32,
    ) -> Self {
        Self {
            num_decode_req_capacity,
            num_target_distribution_capacity,
            num_draft_distribution_capacity,
            top_k,
        }
    }
}
