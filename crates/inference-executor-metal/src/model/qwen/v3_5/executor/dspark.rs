impl Qwen35Executor {
    fn read_dspark_proposal(
        &mut self,
        recorder: &Qwen35ModelOpsRecorder,
        mut decisions: Vec<Qwen35DecodeDecision>,
    ) -> Vec<Qwen35DecodeDecision> {
        let dspark = self.speculator.dspark_mut();
        let proposal = dspark
            .execution
            .read_proposal(
                recorder
                    .dspark_spec_decode
                    .as_ref()
                    .expect("qwen3.5 DSpark proposal requires a Spec Decode recording"),
                &mut dspark.common.spec_probs,
            );
        assert_eq!(
            proposal.token_ids.len(),
            decisions.len(),
            "qwen3.5 DSpark proposal must match Main decisions"
        );
        assert_eq!(
            proposal.confidences.len(),
            decisions.len(),
            "qwen3.5 DSpark confidence must match Main decisions"
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
