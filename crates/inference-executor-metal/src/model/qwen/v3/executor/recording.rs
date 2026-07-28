impl Qwen3Executor {
    fn replay_runtime(&self) -> MetalReplayRuntime<'_> {
        MetalReplayRuntime::new(self.runtime.stream())
    }

    fn replay_main(&self, recorder: &Qwen3ModelOpsRecorder) -> Duration {
        let main_embed_replay = self.main_embed.replay(&recorder.main_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let empty_arguments = ReplayArguments::new();
        let start = Instant::now();
        self.replay_runtime()
            .submit_replay_sequence(&[
                ReplayExecution::new(main_embed_replay, &empty_arguments),
                ReplayExecution::new(main_replay, &empty_arguments),
            ])
            .wait();
        start.elapsed()
    }

    fn submit_main_stage(&mut self, recorder: &mut Qwen3ModelOpsRecorder) -> Duration {
        assert!(
            !recorder.main_stage_submitted,
            "qwen3 replay main stage cannot be submitted twice"
        );
        let elapsed = self.replay_main(recorder);
        recorder.main_stage_submitted = true;
        elapsed
    }
}
