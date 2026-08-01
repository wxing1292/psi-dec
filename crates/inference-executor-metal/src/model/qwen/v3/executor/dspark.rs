impl Qwen3Executor {
    fn record_dspark_embed(
        &mut self,
        recorder: &mut Qwen3ModelOpsRecorder,
        microbatch: &Qwen3Microbatch,
        decisions: &[Qwen3DecodeDecision],
    ) -> Rc<Buffer> {
        assert!(self.dspark_block_size > 0, "Qwen3 DSpark embed requires DSpark");
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
            let num_base_tokens = microbatch
                .q_len(req_index)
                .checked_sub(
                    num_spec_tokens
                        .try_into()
                        .expect("Qwen3 speculative-token count must fit u32"),
                )
                .expect("Qwen3 speculative suffix must fit q_len");
            let anchor_position = microbatch.token_indices()[req_index]
                .checked_add(num_base_tokens)
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

        let block = DSparkBlockMetadata::new(&req_slots, &anchor_positions, self.dspark_block_size);
        self.dspark_gqa_state
            .as_ref()
            .expect("Qwen3 DSpark requires GQA state")
            .prepare_block(&block);
        let mask_token_id = self
            .dspark_mask_token_id
            .expect("Qwen3 DSpark requires a MASK token ID");
        let mut block_token_ids = Vec::with_capacity(block.num_tokens());
        for &anchor_token_id in &anchor_token_ids {
            block_token_ids.push(
                anchor_token_id
                    .try_into()
                    .expect("Qwen3 DSpark anchor token ID must fit i32"),
            );
            block_token_ids.extend(std::iter::repeat_n(mask_token_id, self.dspark_block_size - 1));
        }
        self.write_token_ids(&block_token_ids);
        let markov_replay_shape = self
            .dspark_markov
            .as_ref()
            .expect("Qwen3 DSpark requires Markov sampling")
            .prepare(
                &req_slots,
                &anchor_token_ids,
                &anchor_positions,
                &sampler_configs,
                &self.spec_probs,
            );
        self.dspark_gather_unembed
            .as_ref()
            .expect("Qwen3 DSpark requires GatherUnembed")
            .component()
            .prepare(req_slots.len());

        let hidden = Rc::clone(
            self.dspark_hidden_input
                .as_ref()
                .expect("Qwen3 DSpark requires an embed output"),
        );
        let input = Qwen3xDSparkEmbedArgs {
            num_tokens: block
                .num_tokens()
                .try_into()
                .expect("Qwen3 DSpark block token count must fit u32"),
            token_ids: &self.token_ids,
            hidden_output: &hidden,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (key, _) = self
            .dspark_embed
            .as_mut()
            .expect("Qwen3 DSpark requires DSparkEmbed")
            .record(&runtime, &input);
        recorder.dspark_embed_key = Some(key);
        recorder.dspark_markov_replay_shape = Some(markov_replay_shape);
        recorder.dspark_req_slots = req_slots;
        hidden
    }

    fn record_dspark(
        &mut self,
        recorder: &mut Qwen3ModelOpsRecorder,
        hidden_input: Rc<Buffer>,
    ) -> Rc<Buffer> {
        assert!(
            Rc::ptr_eq(
                &hidden_input,
                self.dspark_hidden_input
                    .as_ref()
                    .expect("Qwen3 DSpark requires its body-input workspace")
            ),
            "Qwen3 DSpark must consume the DSparkEmbed workspace"
        );
        let hidden_output = Rc::clone(
            self.dspark_hidden_output
                .as_ref()
                .expect("Qwen3 DSpark requires its body-output workspace"),
        );
        let metadata = self
            .dspark_gqa_state
            .as_ref()
            .expect("Qwen3 DSpark requires GQA state")
            .metadata();
        let input = Qwen3xDSparkBodyArgs {
            num_tokens: metadata.replay_shape().num_tokens,
            metadata,
            hidden_input: &hidden_input,
            hidden_output: &hidden_output,
            pages: self.pages.buffer(),
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (key, _) = self
            .dspark
            .as_mut()
            .expect("Qwen3 DSpark requires its body")
            .record(&runtime, &input);
        recorder.dspark_key = Some(key);
        hidden_output
    }

    fn record_dspark_gather_unembed(
        &mut self,
        recorder: &mut Qwen3ModelOpsRecorder,
        hidden_input: &Rc<Buffer>,
    ) {
        assert!(
            Rc::ptr_eq(
                hidden_input,
                self.dspark_hidden_output
                    .as_ref()
                    .expect("Qwen3 DSpark requires its body-output workspace")
            ),
            "Qwen3 DSpark GatherUnembed must consume the body output"
        );
        let input = Qwen3xDSparkGatherUnembedArgs {
            num_requests: recorder
                .dspark_req_slots
                .len()
                .try_into()
                .expect("Qwen3 DSpark request count must fit u32"),
            hidden_input,
            hidden_output: self
                .dspark_unembed_hidden
                .as_ref()
                .expect("Qwen3 DSpark requires GatherUnembed hidden scratch"),
            logits: self
                .dspark_logits
                .as_ref()
                .expect("Qwen3 DSpark requires draft logits"),
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (key, _) = self
            .dspark_gather_unembed
            .as_mut()
            .expect("Qwen3 DSpark requires GatherUnembed")
            .record(&runtime, &input);
        recorder.dspark_gather_unembed_key = Some(key);
    }

    fn record_dspark_sampling(&mut self, recorder: &mut Qwen3ModelOpsRecorder) {
        let shape = recorder
            .dspark_markov_replay_shape
            .expect("Qwen3 DSpark sampling requires a prepared Markov shape");
        let input = Qwen3xDSparkSamplingArgs {
            shape,
            logits: self
                .dspark_logits
                .as_ref()
                .expect("Qwen3 DSpark sampling requires draft logits"),
            hidden: self
                .dspark_unembed_hidden
                .as_ref()
                .expect("Qwen3 DSpark sampling requires gathered hidden states"),
            distribution_store: &self.spec_probs,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (key, _) = self
            .dspark_sampling
            .as_mut()
            .expect("Qwen3 DSpark requires DraftSampling")
            .record(&runtime, &input);
        let mut arguments = ReplayArguments::new();
        self.dspark_markov
            .as_ref()
            .expect("Qwen3 DSpark requires Markov sampling")
            .add_replay_arguments(shape, &mut arguments);
        recorder.dspark_sampling_key = Some(key);
        recorder.dspark_sampling_arguments = arguments;
    }

    fn read_dspark_proposal(
        &mut self,
        recorder: &Qwen3ModelOpsRecorder,
        mut decisions: Vec<Qwen3DecodeDecision>,
    ) -> Vec<Qwen3DecodeDecision> {
        let proposal = self
            .dspark_markov
            .as_ref()
            .expect("Qwen3 DSpark requires Markov sampling")
            .read_proposal(&recorder.dspark_req_slots, &mut self.spec_probs);
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
