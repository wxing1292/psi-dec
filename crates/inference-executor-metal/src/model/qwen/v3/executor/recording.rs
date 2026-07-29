impl Qwen3Executor {
    fn replay_runtime(&self) -> MetalReplayRuntime<'_> {
        MetalReplayRuntime::new(self.runtime.stream())
    }

    fn submit_main_recording(&self, recorder: &Qwen3ModelOpsRecorder) -> MetalReplaySubmission {
        let main_embed_replay = self.main_embed.replay(&recorder.main_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let empty_arguments = ReplayArguments::new();
        let mut sequence = vec![
            ReplayExecution::new(main_embed_replay, &empty_arguments),
            ReplayExecution::new(main_replay, &empty_arguments),
        ];
        if let Some(context_key) = &recorder.dspark_context_key {
            sequence.push(ReplayExecution::new(
                self.dspark_context
                    .as_ref()
                    .expect("Qwen3 DSpark context key requires its replay owner")
                    .replay(context_key),
                &empty_arguments,
            ));
        }
        if let Some(gather_unembed_key) = &recorder.gather_unembed_key {
            assert!(
                recorder.num_main_sample_rows > 0,
                "qwen3 GatherUnembed replay requires Main sampling rows"
            );
            sequence.push(ReplayExecution::new(
                self.gather_unembed.replay(gather_unembed_key),
                &empty_arguments,
            ));
            if let Some(rejection_key) = &recorder.rejection_key {
                assert!(
                    recorder.sampling_key.is_none(),
                    "qwen3 Main recording must select one sampling path"
                );
                sequence.push(ReplayExecution::new(
                    self.rejection_sampling
                        .as_ref()
                        .expect("Qwen3 rejection key requires its replay owner")
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

    fn submit_dspark_recording(&self, recorder: &Qwen3ModelOpsRecorder) -> MetalReplaySubmission {
        let empty_arguments = ReplayArguments::new();
        let embed = self
            .dspark_embed
            .as_ref()
            .expect("Qwen3 DSpark submission requires DSparkEmbed")
            .replay(
                recorder
                    .dspark_embed_key
                    .as_ref()
                    .expect("Qwen3 DSpark submission requires an embed key"),
            );
        let body = self
            .dspark
            .as_ref()
            .expect("Qwen3 DSpark submission requires its body")
            .replay(
                recorder
                    .dspark_key
                    .as_ref()
                    .expect("Qwen3 DSpark submission requires a body key"),
            );
        let gather_unembed = self
            .dspark_gather_unembed
            .as_ref()
            .expect("Qwen3 DSpark submission requires GatherUnembed")
            .replay(
                recorder
                    .dspark_gather_unembed_key
                    .as_ref()
                    .expect("Qwen3 DSpark submission requires a GatherUnembed key"),
            );
        let sampling = self
            .dspark_sampling
            .as_ref()
            .expect("Qwen3 DSpark submission requires Sampling")
            .replay(
                recorder
                    .dspark_sampling_key
                    .as_ref()
                    .expect("Qwen3 DSpark submission requires a Sampling key"),
            );
        self.replay_runtime().submit_replay_sequence(&[
            ReplayExecution::new(embed, &empty_arguments),
            ReplayExecution::new(body, &empty_arguments),
            ReplayExecution::new(gather_unembed, &empty_arguments),
            ReplayExecution::new(sampling, &recorder.dspark_sampling_arguments),
        ])
    }
}
