impl Qwen3Executor {
    fn write_gather_flat_indices(&self, microbatch: &Qwen3Microbatch) -> Vec<u32> {
        let flat_indices = gather_flat_indices(microbatch);
        assert!(
            !flat_indices.is_empty(),
            "qwen3 replay unembed requires Main output rows"
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
        let num_main_output_rows = self
            .write_gather_flat_indices(microbatch)
            .len()
            .try_into()
            .expect("qwen3 Main output row count must fit u32");
        assert_eq!(
            num_main_output_rows,
            gather_unembed_key.num_main_output_rows(),
            "qwen3 GatherUnembed replay key must match gathered hidden states"
        );
        let input = Qwen3GatherUnembedArgs {
            num_rows: num_main_output_rows,
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

}
