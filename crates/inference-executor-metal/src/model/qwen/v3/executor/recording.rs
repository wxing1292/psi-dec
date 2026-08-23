impl Qwen3Executor {
    fn replay_runtime(&self) -> MetalReplayRuntime<'_> {
        MetalReplayRuntime::new(self.runtime.stream())
    }

    fn submit_main_recording(&self, recorder: &Qwen3ModelOpsRecorder) -> MetalReplaySubmission {
        let main_embed_replay = self.main_embed.replay(&recorder.main_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let mut sequence = vec![
            ReplayExecution::new(main_embed_replay, &recorder.main_embed_arguments),
            ReplayExecution::new(main_replay, &recorder.main_arguments),
        ];
        if let Some(gather_unembed_key) = &recorder.gather_unembed_key {
            assert!(
                recorder.num_main_sample_rows > 0,
                "qwen3 GatherUnembed replay requires Main sampling rows"
            );
            sequence.push(ReplayExecution::new(
                self.gather_unembed.replay(gather_unembed_key),
                &recorder.gather_unembed_arguments,
            ));
            if let Some(rejection_key) = &recorder.rejection_key {
                assert!(
                    recorder.sampling_key.is_none(),
                    "qwen3 Main recording must select one sampling path"
                );
                sequence.push(ReplayExecution::new(
                    self.speculator
                        .dspark()
                        .rejection_sampling
                        .replay(rejection_key),
                    &recorder.rejection_arguments,
                ));
            } else {
                let sampling_key = recorder
                    .sampling_key
                    .as_ref()
                    .expect("qwen3 sampled output requires Sampling replay");
                sequence.push(ReplayExecution::new(
                    self.sampling.replay(sampling_key),
                    &recorder.sampling_arguments,
                ));
            }
        } else {
            assert_eq!(
                recorder.num_main_sample_rows, 0,
                "qwen3 Main sampling rows require GatherUnembed replay"
            );
            assert!(
                recorder.sampling_key.is_none() && recorder.rejection_key.is_none(),
                "qwen3 Main recording without output rows must not contain sampling"
            );
        }
        self.replay_runtime().submit_replay_sequence(&sequence)
    }
}
