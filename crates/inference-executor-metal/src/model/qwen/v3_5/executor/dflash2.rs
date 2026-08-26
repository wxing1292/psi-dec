impl Qwen35Executor {
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
                    .dflash2_spec_decode
                    .as_ref()
                    .expect("qwen3.5 DFlash2 proposal requires a Spec Decode recording"),
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
