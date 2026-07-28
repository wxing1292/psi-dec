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

}
