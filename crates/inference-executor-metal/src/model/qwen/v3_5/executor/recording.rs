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

    fn submit_main_recording(&self, recorder: &Qwen35ModelOpsRecorder) -> MetalReplaySubmission {
        let main_embed_replay = self.main_embed.replay(&recorder.main_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let empty_arguments = ReplayArguments::new();
        self.replay_runtime()
            .submit_replay_sequence(&[
                ReplayExecution::new(main_embed_replay, &empty_arguments),
                ReplayExecution::new(main_replay, &empty_arguments),
            ])
    }

    fn submit_main_sampling_recording(
        &self,
        recorder: &Qwen35ModelOpsRecorder,
    ) -> MetalReplaySubmission {
        trace::qwen35_state(|| {
            format!("event=submit_main_sequence_start main_key={:?}", recorder.main_key)
        });
        let main_embed_replay = self.main_embed.replay(&recorder.main_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let gather_unembed_key = recorder
            .gather_unembed_key
            .as_ref()
            .expect("qwen3.5 sampled output requires GatherUnembed replay");
        let gather_unembed_replay = self.gather_unembed.replay(gather_unembed_key);
        let (sampling_replay, sampling_arguments) = if self.spec_probs.is_enabled() {
            let rejection_key = recorder
                .rejection_key
                .as_ref()
                .expect("qwen3.5 sampled output requires RejectionSampling replay");
            (
                self.rejection_sampling.replay(rejection_key),
                &recorder.rejection_arguments,
            )
        } else {
            let sampling_key = recorder
                .sampling_key
                .as_ref()
                .expect("qwen3.5 sampled output requires Sampling replay");
            (self.sampling.replay(sampling_key), &recorder.sampling_arguments)
        };
        let empty_arguments = ReplayArguments::new();
        let submission = self
            .replay_runtime()
            .submit_replay_sequence(&[
                ReplayExecution::new(main_embed_replay, &empty_arguments),
                ReplayExecution::new(main_replay, &empty_arguments),
                ReplayExecution::new(gather_unembed_replay, &empty_arguments),
                ReplayExecution::new(sampling_replay, sampling_arguments),
            ]);
        trace::qwen35_state(|| {
            format!(
                "event=submit_main_sequence_submitted main_key={:?}",
                recorder.main_key
            )
        });
        submission
    }

    fn submit_gdn_state_restore(&mut self) -> Duration {
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let start = Instant::now();
        self.main_gdn_state.restore(&runtime, self.pages.buffer());
        let elapsed = start.elapsed();
        trace::qwen35_state(|| format!("event=gdn_restore elapsed_us={}", elapsed.as_micros()));
        elapsed
    }
}
