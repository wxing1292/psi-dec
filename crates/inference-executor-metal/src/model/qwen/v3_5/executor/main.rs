impl Qwen35Executor {
    fn record_main(
        &mut self,
        recorder: &mut Qwen35ModelRecorder,
        model_batch_req: &Qwen35ModelBatchRequest,
        model_batch_hidden: Rc<Buffer>,
    ) -> Rc<Buffer> {
        let microbatch = model_batch_req.microbatch();
        assert!(
            Rc::ptr_eq(&model_batch_hidden, &self.token_hidden_input),
            "qwen3.5 Main must consume the MainEmbed hidden workspace"
        );
        let input = Qwen35MainArgs {
            num_tokens: microbatch
                .total_tokens()
                .try_into()
                .expect("qwen3.5 Main token count must fit u32"),
            hidden_input: &model_batch_hidden,
            hidden_output: &self.hidden_output,
            gqa: self.main_gqa_state.metadata(),
            gdn: self.main_gdn_state.metadata(),
            pages: self.pages.buffer(),
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, cache_hit) = self.main.record(&runtime, &input);
        assert_eq!(
            recorded_key, recorder.main_key,
            "qwen3.5 Main replay input must match the prepared replay key"
        );
        recorder.main_cache_hit = cache_hit;
        trace::qwen35_state(|| {
            format!(
                "event=main_replays main_embed_key={:?} main_key={:?} \
                 main_embed_cache_hit={} main_cache_hit={} cache_hit={}",
                recorder.main_embed_key,
                recorder.main_key,
                recorder.main_embed_cache_hit,
                recorder.main_cache_hit,
                recorder.main_replay_cache_hit(),
            )
        });
        self.pending_transactions
            .push(model_batch_req.compute_seq(), microbatch.clone());
        Rc::clone(&self.hidden_output)
    }

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
        recorder: &mut Qwen35ModelRecorder,
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

    fn embed(
        &mut self,
        recorder: &mut Qwen35ModelRecorder,
        model_batch_request: &Qwen35ModelBatchRequest,
    ) -> Rc<Buffer> {
        let input = Qwen35MainEmbedArgs {
            num_tokens: model_batch_request
                .microbatch()
                .total_tokens()
                .try_into()
                .expect("qwen3.5 MainEmbed token count must fit u32"),
            token_ids: &self.token_ids,
            hidden_output: &self.token_hidden_input,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, cache_hit) = self.main_embed.record(&runtime, &input);
        assert_eq!(
            recorded_key, recorder.main_embed_key,
            "qwen3.5 MainEmbed replay input must match the prepared replay key"
        );
        recorder.main_embed_cache_hit = cache_hit;
        Rc::clone(&self.token_hidden_input)
    }

    fn forward_main(
        &mut self,
        recorder: &mut Qwen35ModelRecorder,
        model_batch_req: &Qwen35ModelBatchRequest,
        model_batch_hidden: Rc<Buffer>,
    ) -> Rc<Buffer> {
        self.record_main(recorder, model_batch_req, model_batch_hidden)
    }

    fn unembed(
        &mut self,
        recorder: &mut Qwen35ModelRecorder,
        model_batch_req: &Qwen35ModelBatchRequest,
        model_batch_hidden: &Rc<Buffer>,
    ) -> Qwen35ForwardOutput {
        assert!(
            Rc::ptr_eq(model_batch_hidden, &self.hidden_output),
            "qwen3.5 Output must consume the executor final-norm hidden workspace"
        );
        recorder.gather_unembed_key =
            Some(self.prepare_gather_unembed_replay(model_batch_req.microbatch(), model_batch_hidden));
        Qwen35ForwardOutput
    }
}
