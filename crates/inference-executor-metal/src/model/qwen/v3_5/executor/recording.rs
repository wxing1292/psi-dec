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

    fn replay_main(&self, recorder: &Qwen35ModelOpsRecorder) -> Duration {
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

    fn submit_main_stage(&mut self, recorder: &mut Qwen35ModelOpsRecorder) -> Duration {
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
}
