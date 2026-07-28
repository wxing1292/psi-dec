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

}
