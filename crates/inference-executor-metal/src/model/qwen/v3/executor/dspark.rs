impl Qwen3Executor {
    fn record_dspark_embed(
        &mut self,
        recorder: &mut Qwen3ModelOpsRecorder,
        microbatch: &Qwen3Microbatch,
        decisions: &[Qwen3DecodeDecision],
    ) -> Rc<Buffer> {
        let mut req_slots = Vec::new();
        let mut anchor_token_ids = Vec::new();
        let mut anchor_positions = Vec::new();
        let mut sampler_configs = Vec::new();
        let mut decision_index = 0usize;
        for req_index in 0..microbatch.num_reqs() {
            if !microbatch.is_decode_req(req_index) {
                continue;
            }
            let decision = decisions
                .get(decision_index)
                .expect("Qwen3 DSpark requires one Main decision per decode request");
            let num_spec_tokens = microbatch.num_spec_tokens(req_index) as usize;
            assert!(
                decision.validated_tokens.len() <= num_spec_tokens,
                "Qwen3 DSpark accepted-token count exceeds the speculative suffix"
            );
            let q_end = microbatch.cu_tokens()[req_index + 1] as usize;
            let spec_start = q_end
                .checked_sub(num_spec_tokens)
                .expect("Qwen3 speculative suffix must fit the request");
            assert!(
                microbatch.flat_token_ids()[spec_start..spec_start + decision.validated_tokens.len()]
                    .iter()
                    .copied()
                    .eq(decision.validated_tokens.iter().map(|&token_id| {
                        i32::try_from(token_id).expect("Qwen3 accepted token ID must fit i32")
                    })),
                "Qwen3 accepted tokens must match the speculative input prefix"
            );
            let num_fixed_tokens = microbatch
                .q_len(req_index)
                .checked_sub(
                    num_spec_tokens
                        .try_into()
                        .expect("Qwen3 speculative-token count must fit u32"),
                )
                .expect("Qwen3 speculative suffix must fit q_len");
            let anchor_position = microbatch.token_indices()[req_index]
                .checked_add(num_fixed_tokens)
                .and_then(|position| {
                    position.checked_add(
                        decision
                            .validated_tokens
                            .len()
                            .try_into()
                            .expect("Qwen3 accepted-token count must fit u32"),
                    )
                })
                .expect("Qwen3 DSpark anchor position must fit u32");
            req_slots.push(microbatch.req_slots()[req_index]);
            anchor_token_ids.push(decision.sampled_token);
            anchor_positions.push(anchor_position);
            sampler_configs.push(microbatch.sampler_configs()[req_index]);
            decision_index += 1;
        }
        assert_eq!(
            decision_index,
            decisions.len(),
            "Qwen3 DSpark decisions must match decode requests"
        );
        assert!(!req_slots.is_empty(), "Qwen3 DSpark proposal requires decode requests");

        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let dspark = self.speculator.dspark_mut();
        dspark.execution.record_embed(
            &runtime,
            &self.token_ids,
            Qwen3xDSparkProposalInput::new(
                req_slots,
                &anchor_token_ids,
                &anchor_positions,
                &sampler_configs,
            ),
            &dspark.spec_probs,
            &mut recorder.dspark,
        )
    }

    fn record_dspark(
        &mut self,
        recorder: &mut Qwen3ModelOpsRecorder,
        hidden_input: Rc<Buffer>,
    ) -> Rc<Buffer> {
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let pages = self.pages.buffer();
        self.speculator
            .dspark_mut()
            .execution
            .record_body(&runtime, pages, &mut recorder.dspark, hidden_input)
    }

    fn record_dspark_gather_unembed(
        &mut self,
        recorder: &mut Qwen3ModelOpsRecorder,
        hidden_input: &Rc<Buffer>,
    ) {
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        self.speculator
            .dspark_mut()
            .execution
            .record_gather_unembed(&runtime, &mut recorder.dspark, hidden_input);
    }

    fn record_dspark_sampling(&mut self, recorder: &mut Qwen3ModelOpsRecorder) {
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let dspark = self.speculator.dspark_mut();
        dspark
            .execution
            .record_sampling(&runtime, &dspark.spec_probs, &mut recorder.dspark);
    }

    fn read_dspark_proposal(
        &mut self,
        recorder: &Qwen3ModelOpsRecorder,
        mut decisions: Vec<Qwen3DecodeDecision>,
    ) -> Vec<Qwen3DecodeDecision> {
        let dspark = self.speculator.dspark_mut();
        let proposal = dspark
            .execution
            .read_proposal(&recorder.dspark, &mut dspark.spec_probs);
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
