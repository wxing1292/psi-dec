impl Qwen3Executor {
    fn write_gather_flat_indices(&self, microbatch: &Qwen3Microbatch) -> Vec<u32> {
        let flat_indices = gather_flat_indices(microbatch);
        assert!(
            !flat_indices.is_empty(),
            "qwen3 replay unembed requires target hidden states"
        );
        assert!(
            flat_indices.iter().all(|&flat_index| {
                flat_index
                    < microbatch
                        .total_tokens()
                        .try_into()
                        .expect("qwen3 batch token count must fit u32")
            }),
            "qwen3 gather flat indices must select this batch's flat tokens"
        );
        self.gather_flat_indices.write_typed(0, &flat_indices);
        flat_indices
    }

    fn prepare_gather_unembed_replay(
        &mut self,
        microbatch: &Qwen3Microbatch,
        hidden_input: &Buffer,
    ) -> Qwen3GatherUnembedReplayKey {
        let gather_unembed_key = Qwen3GatherUnembedReplayKey::from_microbatch(microbatch);
        let num_target_hidden_states = self
            .write_gather_flat_indices(microbatch)
            .len()
            .try_into()
            .expect("qwen3 target hidden-state count must fit u32");
        assert_eq!(
            num_target_hidden_states,
            gather_unembed_key.num_target_hidden_states(),
            "qwen3 GatherUnembed replay key must match gathered hidden states"
        );
        let input = Qwen3GatherUnembedArgs {
            num_rows: num_target_hidden_states,
            hidden_input,
            row_indices: &self.gather_flat_indices,
            hidden_output: &self.unembed_hidden,
            logits: &self.unembed_logits,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, _) = self.gather_unembed.record(&runtime, &input);
        assert_eq!(
            recorded_key, gather_unembed_key,
            "qwen3 GatherUnembed replay input must match the prepared replay key"
        );
        recorded_key
    }

    fn submit_main_decode_stage(
        &self,
        recorder: &mut Qwen3ModelOpsRecorder,
        decision_replay: &ReplayProgram,
        decision_arguments: &ReplayArguments,
    ) -> Duration {
        assert!(
            !recorder.main_stage_submitted,
            "qwen3 replay main stage cannot be submitted twice"
        );
        let main_embed_replay = self.main_embed.replay(&recorder.main_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let gather_unembed_key = recorder
            .gather_unembed_key
            .as_ref()
            .expect("qwen3 sampled output requires GatherUnembed replay");
        let gather_unembed_replay = self.gather_unembed.replay(gather_unembed_key);
        let empty_arguments = ReplayArguments::new();
        let start = Instant::now();
        self.replay_runtime()
            .submit_replay_sequence(&[
                ReplayExecution::new(main_embed_replay, &empty_arguments),
                ReplayExecution::new(main_replay, &empty_arguments),
                ReplayExecution::new(gather_unembed_replay, &empty_arguments),
                ReplayExecution::new(decision_replay, decision_arguments),
            ])
            .wait();
        recorder.main_stage_submitted = true;
        start.elapsed()
    }
}
