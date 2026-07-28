impl Qwen3Executor {
    fn replay_runtime(&self) -> MetalReplayRuntime<'_> {
        MetalReplayRuntime::new(self.runtime.stream())
    }

    fn submit_main_recording(&self, recorder: &Qwen3ModelOpsRecorder) -> MetalReplaySubmission {
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
        recorder: &Qwen3ModelOpsRecorder,
    ) -> MetalReplaySubmission {
        let main_embed_replay = self.main_embed.replay(&recorder.main_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let gather_unembed_key = recorder
            .gather_unembed_key
            .as_ref()
            .expect("qwen3 sampled output requires GatherUnembed replay");
        let gather_unembed_replay = self.gather_unembed.replay(gather_unembed_key);
        let sampling_key = recorder
            .sampling_key
            .as_ref()
            .expect("qwen3 sampled output requires Sampling replay");
        let sampling_replay = self.sampling.replay(sampling_key);
        let empty_arguments = ReplayArguments::new();
        self.replay_runtime()
            .submit_replay_sequence(&[
                ReplayExecution::new(main_embed_replay, &empty_arguments),
                ReplayExecution::new(main_replay, &empty_arguments),
                ReplayExecution::new(gather_unembed_replay, &empty_arguments),
                ReplayExecution::new(sampling_replay, &recorder.sampling_arguments),
            ])
    }
}
