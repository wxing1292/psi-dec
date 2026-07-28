impl Qwen35Executor {
    fn write_gather_flat_indices(&self, microbatch: &Qwen35Microbatch) -> Vec<u32> {
        // The mask selects source hidden states. Its compact indices are a
        // dynamic gather input, not batch state: [F, F, F, T, T, T] -> [3, 4, 5].
        let flat_indices = gather_flat_indices(microbatch);
        assert!(
            !flat_indices.is_empty(),
            "qwen3.5 replay unembed requires target hidden states"
        );
        assert!(
            flat_indices.iter().all(|&flat_index| {
                flat_index
                    < microbatch
                        .total_tokens()
                        .try_into()
                        .expect("qwen3.5 batch token count must fit u32")
            }),
            "qwen3.5 gather flat indices must select this batch's flat tokens"
        );
        self.gather_flat_indices.write_typed(0, &flat_indices);
        flat_indices
    }

    fn prepare_gather_unembed_replay(
        &mut self,
        microbatch: &Qwen35Microbatch,
        hidden_input: &Buffer,
    ) -> Qwen35GatherUnembedReplayKey {
        let gather_unembed_key = Qwen35GatherUnembedReplayKey::from_microbatch(microbatch);
        let num_target_hidden_states = self
            .write_gather_flat_indices(microbatch)
            .len()
            .try_into()
            .expect("qwen3.5 target hidden-state count must fit u32");
        assert_eq!(
            num_target_hidden_states,
            gather_unembed_key.num_target_hidden_states(),
            "qwen3.5 GatherUnembed replay key must match gathered hidden states"
        );
        let input = Qwen35GatherUnembedArgs {
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
            "qwen3.5 GatherUnembed replay input must match the prepared replay key"
        );
        recorded_key
    }

    fn submit_main_decode_stage(
        &self,
        recorder: &mut Qwen35ModelOpsRecorder,
        decision_replay: &ReplayProgram,
        decision_arguments: &ReplayArguments,
    ) -> Duration {
        assert!(
            !recorder.main_stage_submitted,
            "qwen3.5 replay main stage cannot be submitted twice"
        );
        let main_embed_replay = self.main_embed.replay(&recorder.main_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let gather_unembed_key = recorder
            .gather_unembed_key
            .as_ref()
            .expect("qwen3.5 sampled output requires GatherUnembed replay");
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
        let elapsed = start.elapsed();
        recorder.main_stage_submitted = true;
        elapsed
    }
}
