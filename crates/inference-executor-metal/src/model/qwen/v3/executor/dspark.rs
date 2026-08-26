impl Qwen3Executor {
    fn record_spec(&mut self, recorder: &mut Qwen3ModelOpsRecorder, microbatch: &Qwen3Microbatch) {
        assert!(recorder.dspark_spec_prefill.is_none() && recorder.dspark_spec_decode.is_none());
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let pages = self.pages.buffer();
        // Spec Prefill consumes every Main row. Main GQA metadata already owns
        // the expanded per-row request slots and token indices.
        let spec_prefill = self.main_gqa_state.metadata();
        recorder.dspark_spec_prefill = Some(
            self.speculator
                .dspark_mut()
                .execution
                .record_spec_prefill(&runtime, spec_prefill, pages),
        );
        let Some(prepared) = recorder.rejection_prepared.as_ref() else {
            return;
        };
        let decode_req_indices = &prepared.decode_req_indices;

        let mut req_slots = Vec::with_capacity(decode_req_indices.len());
        let mut anchor_indices = Vec::with_capacity(decode_req_indices.len());
        let mut num_spec_tokens_by_request = Vec::with_capacity(decode_req_indices.len());
        let mut sampler_configs = Vec::with_capacity(decode_req_indices.len());
        for &req_index in decode_req_indices {
            let num_spec_tokens = microbatch.num_spec_tokens(req_index);
            let anchor_index = microbatch.token_indices()[req_index]
                + microbatch.num_total_tokens(req_index)
                - num_spec_tokens;
            req_slots.push(microbatch.req_slots()[req_index]);
            anchor_indices.push(anchor_index);
            num_spec_tokens_by_request.push(num_spec_tokens);
            sampler_configs.push(microbatch.sampler_configs()[req_index]);
        }

        let token_ids = &self.token_ids;
        let dspark = self.speculator.dspark_mut();
        let rejection_sampling = dspark.rejection_sampling.component().rejector().output();
        let (decode_prepare, markov_replay_shape) = dspark.execution.record_decode_prepare(
            &runtime,
            token_ids,
            rejection_sampling,
            &req_slots,
            &anchor_indices,
            &num_spec_tokens_by_request,
            &sampler_configs,
            &dspark.spec_probs,
        );
        recorder.dspark_spec_decode = Some(dspark.execution.record_spec_decode(
            &runtime,
            token_ids,
            decode_prepare,
            markov_replay_shape,
            req_slots,
            pages,
            &dspark.spec_probs,
        ));
    }

    fn read_dspark_proposal(
        &mut self,
        recorder: &Qwen3ModelOpsRecorder,
        mut decisions: Vec<Qwen3DecodeDecision>,
    ) -> Vec<Qwen3DecodeDecision> {
        let dspark = self.speculator.dspark_mut();
        let proposal = dspark
            .execution
            .read_proposal(
                recorder
                    .dspark_spec_decode
                    .as_ref()
                    .expect("Qwen3 DSpark proposal requires a Spec Decode recording"),
                &mut dspark.spec_probs,
            );
        assert_eq!(
            proposal.token_ids.len(),
            decisions.len(),
            "Qwen3 DSpark proposal must match Main decisions"
        );
        assert_eq!(
            proposal.confidences.len(),
            decisions.len(),
            "Qwen3 DSpark confidence must match Main decisions"
        );
        for (((decision, token_ids), token_probs), confidences) in decisions
            .iter_mut()
            .zip(proposal.token_ids)
            .zip(proposal.token_probs)
            .zip(proposal.confidences)
        {
            assert_eq!(token_ids.len(), token_probs.len());
            assert_eq!(token_ids.len(), confidences.len());
            decision.spec_tokens = token_ids;
            decision.spec_probs = token_probs;
            decision.spec_confidences = confidences;
        }
        decisions
    }
}
