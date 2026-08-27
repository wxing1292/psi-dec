impl Qwen35Executor {
    fn record_dflash2_decode(
        &mut self,
        microbatch: &Qwen35Microbatch,
        decisions: &[Qwen35DecodeDecision],
    ) -> Qwen3xDFlash2DecodeRecording {
        let mut req_slots = Vec::new();
        let mut anchor_token_ids = Vec::new();
        let mut anchor_positions = Vec::new();
        let mut decision_index = 0usize;
        for req_index in 0..microbatch.num_reqs() {
            if !microbatch.is_decode_req(req_index) {
                continue;
            }
            let decision = decisions
                .get(decision_index)
                .expect("qwen3.5 DFlash2 requires one Main decision per decode request");
            let num_spec_tokens = microbatch.num_spec_tokens(req_index);
            assert!(
                decision.validated_tokens.len() <= num_spec_tokens as usize,
                "qwen3.5 DFlash2 accepted-token count exceeds the speculative suffix"
            );
            let q_end = microbatch.cu_tokens()[req_index + 1] as usize;
            let spec_start = q_end - num_spec_tokens as usize;
            assert!(
                microbatch.flat_token_ids()[spec_start..spec_start + decision.validated_tokens.len()]
                    .iter()
                    .copied()
                    .eq(decision.validated_tokens.iter().map(|&token_id| {
                        i32::try_from(token_id).expect("qwen3.5 accepted token ID must fit i32")
                    })),
                "qwen3.5 accepted tokens must match the speculative input prefix"
            );
            let num_fixed_tokens = microbatch.q_len(req_index) - num_spec_tokens;
            let anchor_position =
                microbatch.token_indices()[req_index] + num_fixed_tokens + decision.validated_tokens.len() as u32;
            req_slots.push(microbatch.req_slots()[req_index]);
            anchor_token_ids.push(decision.sampled_token);
            anchor_positions.push(anchor_position);
            decision_index += 1;
        }
        assert_eq!(
            decision_index,
            decisions.len(),
            "qwen3.5 DFlash2 decisions must match decode requests"
        );
        assert!(!req_slots.is_empty(), "qwen3.5 DFlash2 proposal requires decode requests");

        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let dflash2 = self.speculator.dflash2_mut();
        dflash2.execution.record_decode(
            &runtime,
            &self.token_ids,
            Qwen3xDFlash2ProposalInput::new(
                req_slots,
                &anchor_token_ids,
                &anchor_positions,
            ),
            self.pages.buffer(),
            &dflash2.common.spec_probs,
        )
    }

    fn read_dflash2_proposal(
        &mut self,
        recorder: &Qwen35ModelOpsRecorder,
        mut decisions: Vec<Qwen35DecodeDecision>,
    ) -> Vec<Qwen35DecodeDecision> {
        let dflash2 = self.speculator.dflash2_mut();
        let proposal = dflash2
            .execution
            .read_proposal(
                recorder
                    .dflash2_decode
                    .as_ref()
                    .expect("qwen3.5 DFlash2 proposal requires a Decode recording"),
                &mut dflash2.common.spec_probs,
            );
        assert_eq!(
            proposal.token_ids.len(),
            decisions.len(),
            "qwen3.5 DFlash2 proposal must match Main decisions"
        );
        for ((decision, token_ids), token_probs) in decisions
            .iter_mut()
            .zip(proposal.token_ids)
            .zip(proposal.token_probs)
        {
            assert_eq!(token_ids.len(), token_probs.len());
            decision.spec_confidences = vec![1.0; token_ids.len()];
            decision.spec_tokens = token_ids;
            decision.spec_probs = token_probs;
        }
        decisions
    }
}
