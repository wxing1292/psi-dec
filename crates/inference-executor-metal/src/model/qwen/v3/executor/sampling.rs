impl Qwen3Executor {
    fn sample_replay_shape(&self, sampler_configs: &[SamplerConfig]) -> TopKSamplingShape {
        let shape = self.sampler.active_shape(sampler_configs);
        shape.with_num_total_sampling_inputs(replay_bucket_capacity(
            shape.num_active_sampling_inputs,
            self.sampler_bounds.max_sampling_inputs,
        ))
    }

    fn prepare_sample_replay(
        &mut self,
        sampler_configs: &[SamplerConfig],
        sample_positions: &[u32],
    ) -> (TopKSamplingReplayKey, ReplayArguments) {
        assert_eq!(
            sampler_configs.len(),
            sample_positions.len(),
            "qwen3 sample runtime configs and positions must have equal lengths"
        );
        let sample_shape = self.sample_replay_shape(sampler_configs);
        let input = SamplingInput {
            shape: sample_shape,
            logits: &self.unembed_logits,
            output: self.sampler_output.as_output(),
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (sample_key, _) = self.sampling.record(&runtime, &input);
        self.sampler
            .set_configs(sampler_configs, sample_positions, SamplingDomain::Target);
        let mut replay_arguments = ReplayArguments::new();
        self.sampler.add_replay_arguments(sample_shape, &mut replay_arguments);
        (sample_key, replay_arguments)
    }

    fn record_sampling(&mut self, microbatch: &Qwen3Microbatch) -> (TopKSamplingReplayKey, ReplayArguments) {
        let sampler_configs = sample_sampler_configs(microbatch);
        let sample_positions = sample_token_positions(microbatch);
        self.prepare_sample_replay(&sampler_configs, &sample_positions)
    }

    fn read_sampled_token_ids(&self, num_decode_reqs: usize) -> Qwen3SampledTokens {
        let output = &self.sampler_output;
        Qwen3SampledTokens::new(
            output.token_ids.read_typed::<i32>(0, num_decode_reqs),
            output.token_probs.read_typed::<f32>(0, num_decode_reqs),
        )
    }

    fn read_sample_decisions(&self, num_decode_reqs: usize) -> Vec<Qwen3DecodeDecision> {
        sample_decisions_from_sampled_tokens(&self.read_sampled_token_ids(num_decode_reqs))
    }

    fn record_rejection_sampling(&mut self, recorder: &mut Qwen3ModelOpsRecorder, microbatch: &Qwen3Microbatch) {
        assert!(self.dspark_block_size > 0, "Qwen3 rejection sampling requires DSpark");
        let sample_positions = sample_token_positions(microbatch);
        let sampler_configs = sample_sampler_configs(microbatch);
        let mut flat_draft_distribution_indices = Vec::new();
        for req_index in 0..microbatch.num_reqs() {
            if !microbatch.is_decode_req(req_index) {
                continue;
            }
            let req_slot = microbatch.req_slots()[req_index];
            let num_spec_tokens = microbatch.num_spec_tokens(req_index) as usize;
            assert!(
                num_spec_tokens <= self.dspark_block_size,
                "Qwen3 speculative suffix exceeds DSpark capacity"
            );
            let q_end = microbatch.cu_tokens()[req_index + 1] as usize;
            for (spec_token_index, &draft_token) in microbatch.flat_token_ids()[q_end - num_spec_tokens..q_end]
                .iter()
                .enumerate()
            {
                self.spec_probs.assert_expected_draft_token(
                    req_slot,
                    spec_token_index,
                    draft_token
                        .try_into()
                        .expect("Qwen3 request contained a negative draft token ID"),
                );
                flat_draft_distribution_indices
                    .push(self.spec_probs.draft_distribution_index(req_slot, spec_token_index));
            }
        }
        let rejector = Rc::clone(
            self.rejection_sampling
                .as_ref()
                .expect("Qwen3 DSpark requires rejection sampling")
                .component()
                .rejector(),
        );
        let prepared = rejector.prepare_inputs(microbatch, &flat_draft_distribution_indices);
        let num_active_decode_reqs = prepared.num_active_decode_reqs();
        let num_active_draft_distributions = prepared.num_active_draft_distributions;
        let num_active_target_distributions = prepared.num_active_target_distributions();
        let num_decode_req_capacity = replay_bucket_capacity_usize(num_active_decode_reqs, self.config.max_requests);
        let max_draft_distributions = self
            .config
            .max_requests
            .checked_mul(self.dspark_block_size)
            .expect("Qwen3 rejection draft capacity must fit usize");
        let num_draft_distribution_capacity =
            replay_bucket_capacity_allow_zero(num_active_draft_distributions, max_draft_distributions);
        let max_target_distributions = self
            .config
            .max_requests
            .checked_mul(
                self.dspark_block_size
                    .checked_add(1)
                    .expect("Qwen3 Main rows per request must fit usize"),
            )
            .expect("Qwen3 rejection target capacity must fit usize")
            .min(self.config.max_tokens);
        let num_target_distribution_capacity =
            replay_bucket_capacity_usize(num_active_target_distributions, max_target_distributions);
        let target_shape = self
            .sampler
            .active_shape(&sampler_configs)
            .with_num_total_sampling_inputs(
                num_target_distribution_capacity
                    .try_into()
                    .expect("Qwen3 target-distribution capacity must fit u32"),
            );
        let rejection_input = RejectionSamplerInput {
            num_active_decode_reqs,
            num_decode_req_capacity,
            num_target_distribution_capacity,
            num_active_draft_distributions,
            num_draft_distribution_capacity,
            top_k: target_shape.top_k,
            target_token_ids: self.spec_probs.target_token_ids(),
            target_probs: self.spec_probs.target_probs(),
            draft_token_ids: self.spec_probs.draft_token_ids(),
            draft_probs: self.spec_probs.draft_probs(),
        };
        let input = RejectionSamplingInput {
            target_shape,
            logits: &self.unembed_logits,
            target_sparse: TopKSamplingWriteDistributionOutput {
                token_ids: self.spec_probs.target_token_ids(),
                probs: self.spec_probs.target_probs(),
                output_distribution_indices: &self.target_distribution_indices,
                max_k: self
                    .spec_probs
                    .max_k()
                    .try_into()
                    .expect("Qwen3 distribution width must fit u32"),
                num_output_distributions: self.spec_probs.num_target_distributions(),
            },
            rejection: rejection_input,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (rejection_key, _) = self
            .rejection_sampling
            .as_mut()
            .expect("Qwen3 DSpark requires rejection sampling")
            .record(&runtime, &input);
        self.sampler
            .set_configs(&sampler_configs, &sample_positions, SamplingDomain::Target);
        let mut runtime_params = Vec::with_capacity(num_active_decode_reqs);
        let mut sample_offset = 0usize;
        for &req_index in &prepared.decode_req_indices {
            let config = microbatch.sampler_configs()[req_index];
            runtime_params.push(SparseRejectionSamplingReqParams {
                seed: config.seed(),
                sample_position: sample_positions[sample_offset],
                top_k: self
                    .sampler_bounds
                    .active_top_k(&config)
                    .expect("Qwen3 rejection sampler config must fit bounds"),
            });
            sample_offset = sample_offset
                .checked_add(microbatch.num_spec_tokens(req_index) as usize + 1)
                .expect("Qwen3 rejection sample offset must fit usize");
        }
        assert_eq!(
            sample_offset,
            sample_positions.len(),
            "Qwen3 rejection positions must cover Main rows"
        );
        rejector.set_runtime_params(&runtime_params);
        let mut arguments = ReplayArguments::new();
        self.sampler.add_replay_arguments(target_shape, &mut arguments);
        rejector.add_replay_arguments(rejection_input, &mut arguments);
        recorder.rejection_key = Some(rejection_key);
        recorder.rejection_arguments = arguments;
        recorder.rejection_prepared = Some(prepared);
    }

    fn read_rejection_decisions(
        &self,
        recorder: &Qwen3ModelOpsRecorder,
        microbatch: &Qwen3Microbatch,
    ) -> Vec<Qwen3DecodeDecision> {
        let prepared = recorder
            .rejection_prepared
            .as_ref()
            .expect("Qwen3 rejection read requires prepared inputs");
        let rejector = self
            .rejection_sampling
            .as_ref()
            .expect("Qwen3 rejection read requires DSpark")
            .component()
            .rejector();
        let results = rejector.read_results(
            prepared.num_active_decode_reqs(),
            prepared.num_active_draft_distributions,
        );
        let mut flat_draft_index = 0usize;
        let mut decisions = Vec::with_capacity(prepared.num_active_decode_reqs());
        for decode_req_index in 0..prepared.num_active_decode_reqs() {
            let num_accepted = results.num_accepted_tokens(decode_req_index);
            decisions.push(Qwen3DecodeDecision {
                validated_tokens: results
                    .accepted_token_ids(flat_draft_index, num_accepted)
                    .iter()
                    .map(|&token_id| {
                        token_id
                            .try_into()
                            .expect("Qwen3 rejection returned a negative accepted token")
                    })
                    .collect(),
                validated_probs: results.accepted_probs(flat_draft_index, num_accepted).to_vec(),
                sampled_token: results
                    .sampled_token_id(decode_req_index)
                    .try_into()
                    .expect("Qwen3 rejection returned a negative sampled token"),
                sampled_prob: results.sampled_prob(decode_req_index),
                ..Qwen3DecodeDecision::default()
            });
            let req_index = prepared.decode_req_indices[decode_req_index];
            flat_draft_index = flat_draft_index
                .checked_add(microbatch.num_spec_tokens(req_index) as usize)
                .expect("Qwen3 rejection draft offset must fit usize");
        }
        assert_eq!(
            flat_draft_index, prepared.num_active_draft_distributions,
            "Qwen3 rejection results must cover all draft rows"
        );
        decisions
    }
}

fn replay_bucket_capacity_usize(active: usize, max_capacity: usize) -> usize {
    assert!(active > 0 && active <= max_capacity);
    active
        .checked_next_power_of_two()
        .map_or(max_capacity, |bucket| bucket.min(max_capacity))
}

fn replay_bucket_capacity_allow_zero(active: usize, max_capacity: usize) -> usize {
    if active == 0 {
        0
    } else {
        replay_bucket_capacity_usize(active, max_capacity)
    }
}

/// Main verification distributions live only in the current compact batch.
/// Draft distributions use request-slot identity in `SpecProbsStore`.
fn compact_target_distribution_indices(capacity: usize) -> Vec<u32> {
    assert!(capacity > 0, "Qwen3 target distribution capacity must be positive");
    (0..capacity)
        .map(|index| u32::try_from(index).expect("Qwen3 target distribution index must fit u32"))
        .collect()
}

#[cfg(test)]
mod sampling_tests {
    use super::compact_target_distribution_indices;

    #[test]
    fn test_target_distribution_indices_are_compact_and_ignore_request_slots() {
        assert_eq!(compact_target_distribution_indices(5), [0, 1, 2, 3, 4]);
    }
}
