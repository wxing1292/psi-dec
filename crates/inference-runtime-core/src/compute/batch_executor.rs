use std::time::Duration;

use crate::compute::BatchDeviceRequest;
use crate::compute::BatchDeviceResponse;
use crate::compute::DeviceRequest;
use crate::runtime::RawRequestSlot;
use crate::runtime::Token;

pub trait ReplayableModelBatchExecutor {
    type ModelBatchReq;
    type ModelBatchHidden;
    type ModelBatchResp;
    type SampledOutput;
    type ModelOpsRecorder;

    fn model_name(&self) -> &str;

    fn default_stop_sequences(&self) -> Vec<Vec<Token>> {
        Vec::new()
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]);

    fn prepare_batch(&mut self, core_batch_req: &BatchDeviceRequest) -> Self::ModelBatchReq;
    fn commit_batch(
        &mut self,
        core_batch_req: BatchDeviceRequest,
        sampled_output: Self::SampledOutput,
    ) -> BatchDeviceResponse;

    fn begin_ops_recording(&mut self, batch_req: &Self::ModelBatchReq) -> Self::ModelOpsRecorder;
    fn finish_ops_recording(
        &mut self,
        recorder: Self::ModelOpsRecorder,
        sampled_output: Self::SampledOutput,
    ) -> Self::SampledOutput {
        let _recorder = recorder;
        sampled_output
    }

    fn embed(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        batch_req: &Self::ModelBatchReq,
    ) -> Self::ModelBatchHidden;
    fn unembed(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchReq,
        model_batch_hidden: &Self::ModelBatchHidden,
    ) -> Self::ModelBatchResp;

    fn forward_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchReq,
        model_batch_hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden;
    fn forward_mtp(
        &mut self,
        _recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchReq,
        _model_batch_hidden: &Self::ModelBatchHidden,
        sampled_output: Self::SampledOutput,
    ) -> Self::SampledOutput {
        sampled_output
    }

    fn sample(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchReq,
        model_batch_resp: &Self::ModelBatchResp,
    ) -> Self::SampledOutput;
    fn rejection_sample(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchReq,
        model_batch_resp: &Self::ModelBatchResp,
    ) -> Self::SampledOutput {
        self.sample(recorder, model_batch_req, model_batch_resp)
    }
    fn empty_sampled_output(&self) -> Self::SampledOutput;
    fn sampled_output_len(&self, sampled_output: &Self::SampledOutput) -> usize;
    fn sampled_output_timing(&self, _sampled_output: &Self::SampledOutput) -> Option<ModelOutputTiming> {
        None
    }

    fn first_pp_stage(&self, _batch_req: &Self::ModelBatchReq) -> bool {
        true
    }
    fn last_pp_stage(&self, _batch_req: &Self::ModelBatchReq) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelOutputTiming {
    pub main_replay_elapsed: Duration,
    pub main_output_replay_elapsed: Duration,
    pub sample_read_elapsed: Duration,
    pub rejection_build_elapsed: Duration,
    pub rejection_read_elapsed: Duration,
    pub mtp_build_elapsed: Duration,
    pub mtp_replay_elapsed: Duration,
    pub mtp_read_elapsed: Duration,
    pub mtp_modules: usize,
}

impl ModelOutputTiming {
    pub fn add_assign(&mut self, other: Self) {
        self.main_replay_elapsed += other.main_replay_elapsed;
        self.main_output_replay_elapsed += other.main_output_replay_elapsed;
        self.sample_read_elapsed += other.sample_read_elapsed;
        self.rejection_build_elapsed += other.rejection_build_elapsed;
        self.rejection_read_elapsed += other.rejection_read_elapsed;
        self.mtp_build_elapsed += other.mtp_build_elapsed;
        self.mtp_replay_elapsed += other.mtp_replay_elapsed;
        self.mtp_read_elapsed += other.mtp_read_elapsed;
        self.mtp_modules += other.mtp_modules;
    }

    pub fn is_zero(self) -> bool {
        self == Self::default()
    }
}

pub fn page_ids_by_layer_for_lane(
    request: &DeviceRequest,
    cache_lane: usize,
    num_gqa_layers: usize,
    num_page_ids_per_block: usize,
    model_name: &str,
) -> Vec<Vec<Vec<u32>>> {
    let page_ids_by_lane_and_block = request.decoder_sync_blocks.kv_page_ids();
    let page_ids_by_block = page_ids_by_lane_and_block
        .get(cache_lane)
        .unwrap_or_else(|| panic!("{model_name} missing cache lane {cache_lane} for kv page ids"));
    let mut page_ids_by_layer = (0..num_gqa_layers)
        .map(|_| Vec::with_capacity(page_ids_by_block.len()))
        .collect::<Vec<_>>();
    for page_ids_for_one_block in page_ids_by_block {
        assert_eq!(
            num_gqa_layers * num_page_ids_per_block,
            page_ids_for_one_block.len(),
            "{model_name} expects {} page ids for each synced kv block in cache lane {cache_lane}, got {}",
            num_gqa_layers * num_page_ids_per_block,
            page_ids_for_one_block.len()
        );
        for (gqa_layer_index, page_ids_by_block) in page_ids_by_layer.iter_mut().enumerate() {
            let page_id_start = gqa_layer_index * num_page_ids_per_block;
            page_ids_by_block
                .push(page_ids_for_one_block[page_id_start..page_id_start + num_page_ids_per_block].to_vec());
        }
    }

    page_ids_by_layer
}
