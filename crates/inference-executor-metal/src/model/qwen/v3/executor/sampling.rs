impl Qwen3Executor {
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
        let sample_shape = self.sampling.component().prepare_shape(sampler_configs);
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
        let max_spec_tokens = self.speculator.dspark().execution.num_spec_tokens();
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
                num_spec_tokens <= max_spec_tokens,
                "Qwen3 speculative suffix exceeds DSpark capacity"
            );
            let q_end = microbatch.cu_tokens()[req_index + 1] as usize;
            for (spec_token_index, &draft_token) in microbatch.flat_token_ids()[q_end - num_spec_tokens..q_end]
                .iter()
                .enumerate()
            {
                self.speculator.dspark().spec_probs.assert_expected_draft_token(
                    req_slot,
                    spec_token_index,
                    draft_token
                        .try_into()
                        .expect("Qwen3 request contained a negative draft token ID"),
                );
                flat_draft_distribution_indices.push(
                    self.speculator
                        .dspark()
                        .spec_probs
                        .draft_distribution_index(req_slot, spec_token_index),
                );
            }
        }
        let rejector = Rc::clone(
            self.speculator
                .dspark()
                .rejection_sampling
                .component()
                .rejector(),
        );
        let prepared = rejector.prepare_inputs(microbatch, &flat_draft_distribution_indices);
        let num_active_decode_reqs = prepared.num_active_decode_reqs();
        let active_target_shape = self.sampler.active_shape(&sampler_configs);
        let rejection_shape = rejector.prepare_replay_shape(&prepared, active_target_shape.top_k);
        let target_shape = active_target_shape
            .with_num_total_sampling_inputs(rejection_shape.num_total_target_distributions);
        let Qwen3DSparkSpeculator {
            rejection_sampling,
            spec_probs,
            target_distribution_indices,
            ..
        } = self.speculator.dspark_mut();
        let rejection_input = RejectionSamplerInput {
            shape: rejection_shape,
            target_token_ids: spec_probs.target_token_ids(),
            target_probs: spec_probs.target_probs(),
            draft_token_ids: spec_probs.draft_token_ids(),
            draft_probs: spec_probs.draft_probs(),
        };
        let input = RejectionSamplingInput {
            target_shape,
            logits: &self.unembed_logits,
            target_sparse: TopKSamplingWriteDistributionOutput {
                token_ids: spec_probs.target_token_ids(),
                probs: spec_probs.target_probs(),
                output_distribution_indices: target_distribution_indices,
                max_k: spec_probs
                    .max_k()
                    .try_into()
                    .expect("Qwen3 distribution width must fit u32"),
                num_output_distributions: spec_probs.num_target_distributions(),
            },
            rejection: rejection_input,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (rejection_key, _) = rejection_sampling.record(&runtime, &input);
        self.sampler
            .set_configs(&sampler_configs, &sample_positions, SamplingDomain::Target);
        let mut runtime_params = Vec::with_capacity(num_active_decode_reqs);
        let mut sample_offset = 0usize;
        for &req_index in &prepared.decode_req_indices {
            let config = microbatch.sampler_configs()[req_index];
            let sample_position = sample_positions[sample_offset];
            let num_spec_tokens = microbatch.num_spec_tokens(req_index);
            assert!(
                sample_position <= u32::MAX - num_spec_tokens,
                "Qwen3 rejection sampling positions must fit u32"
            );
            runtime_params.push(SparseRejectionSamplingReqParams {
                seed: config.seed(),
                sample_position,
                top_k: self
                    .sampler_bounds
                    .active_top_k(&config)
                    .expect("Qwen3 rejection sampler config must fit bounds"),
            });
            sample_offset += num_spec_tokens as usize + 1;
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
            .speculator
            .dspark()
            .rejection_sampling
            .component()
            .rejector();
        let results = rejector.read_results(
            prepared.num_active_decode_reqs(),
            prepared.num_active_draft_distributions,
        );
        let mut flat_draft_index = 0usize;
        let mut decisions = Vec::with_capacity(prepared.num_active_decode_reqs());
        for decode_req_index in 0..prepared.num_active_decode_reqs() {
            let num_accepted_tokens = results.num_accepted_tokens(decode_req_index);
            decisions.push(Qwen3DecodeDecision {
                validated_tokens: results
                    .accepted_token_ids(flat_draft_index, num_accepted_tokens)
                    .iter()
                    .map(|&token_id| {
                        token_id
                            .try_into()
                            .expect("Qwen3 rejection returned a negative accepted token")
                    })
                    .collect(),
                validated_probs: results.accepted_probs(flat_draft_index, num_accepted_tokens).to_vec(),
                sampled_token: results
                    .sampled_token_id(decode_req_index)
                    .try_into()
                    .expect("Qwen3 rejection returned a negative sampled token"),
                sampled_prob: results.sampled_prob(decode_req_index),
                ..Qwen3DecodeDecision::default()
            });
            let req_index = prepared.decode_req_indices[decode_req_index];
            flat_draft_index += microbatch.num_spec_tokens(req_index) as usize;
        }
        assert_eq!(
            flat_draft_index, prepared.num_active_draft_distributions,
            "Qwen3 rejection results must cover all draft rows"
        );
        decisions
    }
}

/// Main verification distributions live only in the current compact batch.
/// Draft distributions use request-slot identity in `SpecProbsStore`.
fn compact_target_distribution_indices(capacity: usize) -> Vec<u32> {
    assert!(capacity > 0, "Qwen3 target distribution capacity must be positive");
    (0..capacity).map(|index| index as u32).collect()
}
