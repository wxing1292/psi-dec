impl Qwen35Executor {
    fn replay_runtime(&self) -> MetalReplayRuntime<'_> {
        MetalReplayRuntime::new(self.runtime.stream())
    }

    fn create_recorder(&self) -> ReplayRecorder {
        let runtime = self.replay_runtime();
        Runtime::create_recorder(&runtime)
    }

    fn submit_replay(&self, replay: &ReplayProgram) {
        self.runtime.submit_replay(replay).wait();
    }

    fn submit_replay_with_arguments(&self, replay: &ReplayProgram, arguments: &ReplayArguments) {
        self.runtime.submit_replay_with_arguments(replay, arguments).wait();
    }

    fn replay_main(&self, recorder: &Qwen35ModelRecorder) -> Duration {
        let main_embed_replay = self.main_embed.replay(&recorder.main_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let empty_arguments = ReplayArguments::new();
        let main_replay_start = Instant::now();
        self.replay_runtime()
            .submit_replay_sequence(&[
                ReplayExecution::new(main_embed_replay, &empty_arguments),
                ReplayExecution::new(main_replay, &empty_arguments),
            ])
            .wait();
        main_replay_start.elapsed()
    }

    fn submit_main_stage(&mut self, recorder: &mut Qwen35ModelRecorder) -> Duration {
        trace::qwen35_state(|| {
            format!(
                "event=submit_main_stage_start main_key={:?} submitted={}",
                recorder.main_key, recorder.main_stage_submitted
            )
        });
        assert!(
            !recorder.main_stage_submitted,
            "qwen3.5 replay main stage cannot be submitted twice"
        );
        let elapsed = self.replay_main(recorder);
        recorder.main_stage_submitted = true;
        trace::qwen35_state(|| {
            format!(
                "event=submit_main_stage_done main_key={:?} elapsed_us={}",
                recorder.main_key,
                elapsed.as_micros()
            )
        });
        elapsed
    }

    fn submit_gdn_state_restore(&mut self) -> Duration {
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let start = Instant::now();
        self.main_gdn_state.restore(&runtime, self.pages.buffer());
        let elapsed = start.elapsed();
        trace::qwen35_state(|| format!("event=gdn_restore elapsed_us={}", elapsed.as_micros()));
        elapsed
    }

    fn begin_ops_recording(&mut self, model_batch_request: &Qwen35ModelBatchRequest) -> Qwen35ModelRecorder {
        let main_embed_key = Qwen35MainEmbedReplayKey::new(
            model_batch_request
                .microbatch()
                .total_tokens()
                .try_into()
                .expect("qwen3.5 MainEmbed token count must fit u32"),
        );
        let main_key = Qwen35MainReplayKey::from_shapes(
            self.main_gqa_state.metadata().replay_shape(),
            self.main_gdn_state.metadata().replay_shape(),
        );
        trace::qwen35_state(|| {
            format!(
                "event=begin_ops_recording main_embed_key={:?} main_key={:?}",
                main_embed_key, main_key
            )
        });
        Qwen35ModelRecorder {
            compute_seq: model_batch_request.compute_seq(),
            main_embed_key,
            main_key,
            main_embed_cache_hit: false,
            main_cache_hit: false,
            gather_unembed_key: None,
            main_stage_submitted: false,
        }
    }

    fn finish_ops_recording(
        &mut self,
        recorder: Qwen35ModelRecorder,
        mut sampled_output: Qwen35DecodeOutput,
    ) -> Qwen35DecodeOutput {
        let mut recorder = recorder;
        if recorder.main_stage_submitted {
            return sampled_output;
        }
        if sampled_output.read_sampling_output {
            let (num_sample_tokens, sampler_configs, sample_positions) = {
                let microbatch = self.pending_transactions.pending_microbatch(recorder.compute_seq);
                debug_assert!(
                    !microbatch.has_spec_tokens(),
                    "qwen3.5 deferred sampling requires non-spec inputs"
                );
                (
                    u32::try_from(num_target_hidden_states(microbatch))
                        .expect("qwen3.5 target hidden-state count must fit u32"),
                    sample_sampler_configs(microbatch),
                    sample_token_positions(microbatch),
                )
            };
            sampled_output.timing.main_output_replay_elapsed +=
                self.submit_main_sample_stage(&mut recorder, &sampler_configs, &sample_positions);
            let sample_read_start = Instant::now();
            sampled_output.decisions = self.read_sample_decisions(num_sample_tokens as usize);
            trace_decisions("finish_sample_read", &sampled_output.decisions);
            sampled_output.timing.sample_read_elapsed += sample_read_start.elapsed();
        } else {
            sampled_output.timing.main_replay_elapsed += self.submit_main_stage(&mut recorder);
        }
        sampled_output
    }
}
