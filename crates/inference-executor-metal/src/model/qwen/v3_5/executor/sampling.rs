impl Qwen35Executor {
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
            "qwen3.5 sample runtime configs and positions must have equal lengths"
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

    fn prepare_draft_sample_replay(
        &mut self,
        sampler_configs: &[SamplerConfig],
        sample_positions: &[u32],
    ) -> (TopKSamplingReplayKey, ReplayArguments) {
        assert_eq!(
            sampler_configs.len(),
            sample_positions.len(),
            "qwen3.5 draft sample runtime configs and positions must have equal lengths"
        );
        let sample_shape = self.sample_replay_shape(sampler_configs);
        let input = DraftSamplingInput {
            shape: sample_shape,
            logits: &self.unembed_logits,
            output: self.sampler_output.as_output(),
            sparse: TopKSamplingSparseDistributionOutput {
                token_ids: self.spec_probs.draft_token_ids(),
                probs: self.spec_probs.draft_probs(),
                output_distribution_indices: &self.draft_distribution_indices,
                max_k: self
                    .spec_probs
                    .max_k()
                    .try_into()
                    .expect("qwen3.5 MTP distribution width must fit u32"),
                num_output_distributions: self.spec_probs.num_draft_distributions(),
            },
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (sample_key, _) = self.draft_sampling.record(&runtime, &input);
        self.sampler
            .set_configs(sampler_configs, sample_positions, SamplingDomain::Draft);
        let mut replay_arguments = ReplayArguments::new();
        self.sampler.add_replay_arguments(sample_shape, &mut replay_arguments);
        (sample_key, replay_arguments)
    }

    fn submit_main_sample_stage(
        &mut self,
        recorder: &mut Qwen35ModelRecorder,
        sampler_configs: &[SamplerConfig],
        sample_positions: &[u32],
    ) -> Duration {
        let (sample_key, sample_arguments) = self.prepare_sample_replay(sampler_configs, sample_positions);
        let sample_replay = self.sampling.replay(&sample_key);
        let elapsed = self.submit_main_decode_stage(recorder, sample_replay, &sample_arguments);
        trace::qwen35_state(|| {
            format!(
                "event=submit_main_sample_stage_done main_key={:?} sample_key={:?} elapsed_us={}",
                recorder.main_key,
                sample_key,
                elapsed.as_micros()
            )
        });
        elapsed
    }

    fn assert_expected_draft_tokens_fit(&self, num_tokens: usize) {
        assert!(num_tokens > 0, "qwen3.5 replay model requires at least one token");
        assert!(
            num_tokens <= self.layout.max_tokens as usize,
            "qwen3.5 replay model tokens={} exceed max_tokens={}",
            num_tokens,
            self.layout.max_tokens
        );
    }

    fn read_sampled_token_ids(&self, num_decode_reqs: usize) -> Qwen35SampledTokens {
        let output = &self.sampler_output;
        Qwen35SampledTokens::new(
            output.token_ids.read_typed::<i32>(0, num_decode_reqs),
            output.token_probs.read_typed::<f32>(0, num_decode_reqs),
        )
    }

    fn read_sample_decisions(&self, num_decode_reqs: usize) -> Vec<Qwen35DecodeDecision> {
        sample_decisions_from_sampled_tokens(&self.read_sampled_token_ids(num_decode_reqs))
    }

    fn target_rejection_sample(
        &mut self,
        recorder: &mut Qwen35ModelRecorder,
        microbatch: &Qwen35Microbatch,
    ) -> (Vec<Qwen35DecodeDecision>, ModelOutputTiming) {
        let mut timing = ModelOutputTiming::default();
        let sample_positions = sample_token_positions(microbatch);
        let num_target_hidden_states = num_target_hidden_states(microbatch);
        let sampler_configs = sample_sampler_configs(microbatch);
        let mut flat_draft_distribution_indices = Vec::new();
        let max_spec_tokens = self.num_speculative_tokens();
        assert!(
            max_spec_tokens > 0,
            "qwen3.5 target rejection sampling requires a speculator"
        );
        for req_index in 0..microbatch.num_reqs() {
            if !microbatch.is_decode_req(req_index) {
                continue;
            }
            let req_slot = microbatch.req_slots()[req_index];
            let num_spec_tokens = microbatch.num_spec_tokens(req_index) as usize;
            assert!(
                num_spec_tokens <= max_spec_tokens,
                "qwen3.5 replay rejection num_spec_tokens exceeds speculator capacity"
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
                        .expect("qwen3.5 request contained a negative draft token ID"),
                );
                flat_draft_distribution_indices.push(
                    self.spec_probs
                        .draft_distribution_index(req_slot, spec_token_index),
                );
            }
        }
        let prepared = self
            .rejection_sampling
            .component()
            .rejector()
            .prepare_inputs(microbatch, &flat_draft_distribution_indices);
        let num_active_decode_reqs = prepared.num_active_decode_reqs();
        let num_active_draft_distributions = prepared.num_active_draft_distributions;
        let num_active_target_distributions = prepared.num_active_target_distributions();
        let num_decode_req_capacity = replay_bucket_capacity_usize(num_active_decode_reqs, self.config.max_requests);
        let max_draft_distributions = self
            .config
            .max_requests
            .checked_mul(max_spec_tokens)
            .expect("qwen3.5 rejection draft-distribution capacity overflow");
        let num_draft_distribution_capacity =
            replay_bucket_capacity_allow_zero(num_active_draft_distributions, max_draft_distributions);
        let max_target_distributions = max_draft_distributions
            .checked_add(self.config.max_requests)
            .expect("qwen3.5 rejection target-distribution capacity overflow");
        let num_target_distribution_capacity =
            replay_bucket_capacity_usize(num_active_target_distributions, max_target_distributions);
        debug_assert_eq!(
            num_target_hidden_states, num_active_target_distributions,
            "qwen3.5 target hidden states must match target distributions"
        );
        let target_distribution_shape = self
            .sampler
            .active_shape(&sampler_configs)
            .with_num_total_sampling_inputs(
                num_target_distribution_capacity
                    .try_into()
                    .expect("qwen3.5 target-distribution capacity must fit u32"),
            );
        let top_k = target_distribution_shape.top_k;
        let rejection_key = Qwen35TargetRejectionReplayKey::new(
            num_decode_req_capacity,
            num_target_distribution_capacity,
            num_draft_distribution_capacity,
            top_k,
        );
        let rejection_input = Qwen35RejectionSamplingInput {
            num_active_decode_reqs,
            num_decode_req_capacity,
            num_target_distribution_capacity,
            num_active_draft_distributions,
            num_draft_distribution_capacity,
            top_k,
            target_token_ids: self.spec_probs.target_token_ids(),
            target_probs: self.spec_probs.target_probs(),
            draft_token_ids: self.spec_probs.draft_token_ids(),
            draft_probs: self.spec_probs.draft_probs(),
        };
        let component_input = RejectionSamplingInput {
            target_shape: target_distribution_shape,
            logits: &self.unembed_logits,
            target_sparse: TopKSamplingSparseDistributionOutput {
                token_ids: self.spec_probs.target_token_ids(),
                probs: self.spec_probs.target_probs(),
                output_distribution_indices: &self.target_distribution_indices,
                max_k: self
                    .spec_probs
                    .max_k()
                    .try_into()
                    .expect("qwen3.5 draft distribution width must fit u32"),
                num_output_distributions: self.spec_probs.num_target_distributions(),
            },
            rejection: rejection_input,
        };
        {
            let rejection_build_start = Instant::now();
            let runtime = MetalReplayRuntime::new(self.runtime.stream());
            let (recorded_key, rejection_cache_hit) = self.rejection_sampling.record(&runtime, &component_input);
            assert_eq!(
                recorded_key, rejection_key,
                "qwen3.5 rejection replay input must match its key"
            );
            if !rejection_cache_hit {
                timing.rejection_build_elapsed += rejection_build_start.elapsed();
            }
        }
        self.sampler
            .set_configs(&sampler_configs, &sample_positions, SamplingDomain::Target);
        let mut rejection_runtime_params = Vec::with_capacity(num_active_decode_reqs);
        let mut target_offset = 0usize;
        for &req_index in &prepared.decode_req_indices {
            let sampler_config = &microbatch.sampler_configs()[req_index];
            let sample_position = sample_positions[target_offset];
            let num_spec_tokens = microbatch.num_spec_tokens(req_index);
            sample_position
                .checked_add(num_spec_tokens)
                .expect("qwen3.5 rejection sampling position must fit u32");
            rejection_runtime_params.push(SparseRejectionSamplingReqParams {
                seed: microbatch.sampler_configs()[req_index].seed(),
                sample_position,
                top_k: self
                    .sampler_bounds
                    .active_top_k(sampler_config)
                    .expect("qwen3.5 rejection sampler config should fit bounds"),
            });
            target_offset = target_offset
                .checked_add(
                    usize::try_from(num_spec_tokens)
                        .expect("qwen3.5 speculative-token count must fit host usize")
                        .checked_add(1)
                        .expect("qwen3.5 target distribution count must fit usize"),
                )
                .expect("qwen3.5 cumulative target-distribution offset must fit usize");
        }
        assert_eq!(
            target_offset, num_active_target_distributions,
            "qwen3.5 rejection target distributions must cover sampled requests"
        );
        self.rejection_sampling
            .component()
            .rejector()
            .set_runtime_params(&rejection_runtime_params);
        let mut replay_arguments = ReplayArguments::new();
        self.sampler
            .add_replay_arguments(target_distribution_shape, &mut replay_arguments);
        self.rejection_sampling
            .component()
            .rejector()
            .add_replay_arguments(rejection_input, &mut replay_arguments);
        let rejection_replay = self.rejection_sampling.replay(&rejection_key);
        timing.main_output_replay_elapsed +=
            self.submit_main_decode_stage(recorder, rejection_replay, &replay_arguments);
        let rejection_read_start = Instant::now();
        let results = self
            .rejection_sampling
            .component()
            .rejector()
            .read_results(num_active_decode_reqs, num_active_draft_distributions);
        let mut decisions = Vec::with_capacity(num_active_decode_reqs);
        let mut flat_draft_index = 0usize;
        for (decode_req_index, &req_index) in prepared.decode_req_indices.iter().enumerate() {
            let num_accepted_tokens = results.num_accepted_tokens(decode_req_index);
            assert!(
                num_accepted_tokens <= microbatch.num_spec_tokens(req_index) as usize,
                "qwen3.5 replay rejection accepted more tokens than drafts"
            );
            let decision = Qwen35DecodeDecision {
                sampled_token: results
                    .sampled_token_id(decode_req_index)
                    .try_into()
                    .expect("qwen3.5 rejection sampler returned a negative token ID"),
                sampled_prob: results.sampled_prob(decode_req_index),
                validated_tokens: results
                    .accepted_token_ids(flat_draft_index, num_accepted_tokens)
                    .iter()
                    .map(|&token| {
                        token
                            .try_into()
                            .expect("qwen3.5 rejection sampler returned a negative accepted token ID")
                    })
                    .collect(),
                validated_probs: results.accepted_probs(flat_draft_index, num_accepted_tokens).to_vec(),
                ..Qwen35DecodeDecision::default()
            };
            decisions.push(decision);
            flat_draft_index += microbatch.num_spec_tokens(req_index) as usize;
        }
        timing.rejection_read_elapsed += rejection_read_start.elapsed();
        (decisions, timing)
    }

    fn num_speculative_tokens(&self) -> usize {
        usize::from(self.mtp.is_some())
    }

    fn sample(
        &mut self,
        recorder: &mut Qwen35ModelRecorder,
        model_batch_req: &Qwen35ModelBatchRequest,
        _model_batch_resp: &Qwen35ForwardOutput,
    ) -> Qwen35DecodeOutput {
        assert!(
            !model_batch_req.microbatch().has_spec_tokens(),
            "qwen3.5 replay sample requires rejection_sample for speculative inputs"
        );
        if self.spec_probs.is_enabled() {
            let (decisions, timing) = self.target_rejection_sample(recorder, model_batch_req.microbatch());
            return Qwen35DecodeOutput {
                decisions,
                read_sampling_output: false,
                timing,
            };
        }
        Qwen35DecodeOutput {
            decisions: Vec::new(),
            read_sampling_output: true,
            timing: ModelOutputTiming::default(),
        }
    }

    fn rejection_sample(
        &mut self,
        recorder: &mut Qwen35ModelRecorder,
        model_batch_req: &Qwen35ModelBatchRequest,
        _model_batch_resp: &Qwen35ForwardOutput,
    ) -> Qwen35DecodeOutput {
        assert!(
            model_batch_req.microbatch().has_spec_tokens(),
            "qwen3.5 replay rejection_sample requires speculative inputs"
        );
        assert!(
            self.spec_probs.is_enabled(),
            "qwen3.5 replay rejection_sample requires a speculator"
        );
        let (decisions, timing) = self.target_rejection_sample(recorder, model_batch_req.microbatch());
        Qwen35DecodeOutput {
            decisions,
            read_sampling_output: false,
            timing,
        }
    }
}
