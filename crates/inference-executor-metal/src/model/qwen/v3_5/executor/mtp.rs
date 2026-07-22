impl Qwen35Executor {
    fn mtp_requests(&self, microbatch: &Qwen35Microbatch, decisions: &[Qwen35DecodeDecision]) -> Vec<Qwen35MTPRequest> {
        let mut requests = Vec::with_capacity(microbatch.num_reqs());
        let mut decision_index = 0usize;
        for req_index in 0..microbatch.num_reqs() {
            let flat_start = microbatch.cu_tokens()[req_index] as usize;
            let flat_end = microbatch.cu_tokens()[req_index + 1] as usize;
            if microbatch.is_decode_req(req_index) {
                let decision = &decisions[decision_index];
                let num_spec_tokens = microbatch.num_spec_tokens(req_index) as usize;
                assert!(
                    num_spec_tokens <= self.config.num_mtp_modules,
                    "qwen3.5 MTP proposal num_spec_tokens exceeds configured MTP modules"
                );
                let num_base_tokens = (flat_end - flat_start)
                    .checked_sub(num_spec_tokens)
                    .expect("qwen3.5 MTP proposal requires num_spec_tokens <= q_len");
                assert!(
                    num_base_tokens > 0,
                    "qwen3.5 MTP proposal requires a non-spec anchor token"
                );
                assert!(
                    decision.validated_tokens.len() <= num_spec_tokens,
                    "qwen3.5 MTP proposal accepted tokens exceed speculative suffix"
                );
                let accepted_start = flat_start + num_base_tokens;
                let accepted_end = accepted_start + decision.validated_tokens.len();
                assert!(
                    microbatch.flat_token_ids()[accepted_start..accepted_end]
                        .iter()
                        .copied()
                        .eq(decision.validated_tokens.iter().map(|&token| {
                            i32::try_from(token).expect("qwen3.5 validated token must fit the model i32 token domain")
                        })),
                    "qwen3.5 MTP accepted tokens must be the speculative input prefix"
                );
                let num_tokens = num_base_tokens + decision.validated_tokens.len();
                requests.push(Qwen35MTPRequest {
                    num_tokens,
                    current_token_ids: microbatch.flat_token_ids()[flat_start..flat_start + num_tokens].to_vec(),
                    next_token_id: Some(
                        decision
                            .sampled_token
                            .try_into()
                            .expect("qwen3.5 sampled token must fit the model i32 token domain"),
                    ),
                    decision_index: Some(decision_index),
                });
                decision_index += 1;
            } else {
                requests.push(Qwen35MTPRequest {
                    num_tokens: flat_end - flat_start,
                    current_token_ids: Vec::new(),
                    next_token_id: None,
                    decision_index: None,
                });
            }
        }
        assert_eq!(
            decision_index,
            decisions.len(),
            "qwen3.5 MTP decisions must match sampled requests"
        );
        assert!(
            requests.iter().map(|request| request.num_tokens).sum::<usize>() <= self.config.max_tokens,
            "qwen3.5 MTP flat tokens exceed max_tokens"
        );
        requests
    }

    fn mtp_batch(&self, microbatch: &Qwen35Microbatch, requests: &mut [Qwen35MTPRequest]) -> Qwen35MTPModuleBatch {
        let mut flat_token_ids = Vec::new();
        let mut flat_sample_mask = Vec::new();
        let mut cu_tokens = Vec::with_capacity(requests.len() + 1);
        let mut input_gather_flat_indices = Vec::new();
        let mut draft_distribution_indices = Vec::new();
        let mut sampler_configs = Vec::new();
        let mut sample_positions = Vec::new();
        cu_tokens.push(0);
        for (req_index, request) in requests.iter_mut().enumerate() {
            let flat_start = flat_token_ids.len();
            if let Some(next_token_id) = request.next_token_id {
                assert_eq!(
                    request.current_token_ids.len(),
                    request.num_tokens,
                    "qwen3.5 decode MTP tokens must match request flat tokens"
                );
                flat_token_ids.extend_from_slice(&request.current_token_ids[1..]);
                flat_token_ids.push(next_token_id);
                request.current_token_ids = flat_token_ids[flat_start..flat_start + request.num_tokens].to_vec();
            } else {
                let lane_tokens = microbatch.token_ids_for_lane(req_index, 1);
                assert_eq!(
                    lane_tokens.len(),
                    request.num_tokens,
                    "qwen3.5 prefill MTP lane must preserve the main request width"
                );
                flat_token_ids.extend_from_slice(lane_tokens);
            }
            flat_sample_mask.extend(std::iter::repeat_n(false, request.num_tokens));
            if request.decision_index.is_some() {
                *flat_sample_mask
                    .last_mut()
                    .expect("qwen3.5 MTP sampled request requires token") = true;
            }
            let input_start = microbatch.cu_tokens()[req_index] as usize;
            input_gather_flat_indices.extend((0..request.num_tokens).map(|offset| {
                input_start
                    .checked_add(offset)
                    .and_then(|index| u32::try_from(index).ok())
                    .expect("qwen3.5 MTP gather index must fit u32")
            }));
            if request.decision_index.is_some() {
                let req_slot = microbatch.req_slots()[req_index];
                draft_distribution_indices.push(self.spec_probs.draft_distribution_index(req_slot, 0));
                sampler_configs.push(microbatch.sampler_configs()[req_index]);
                sample_positions.push(mtp_proposal_sample_position(
                    microbatch.token_indices()[req_index],
                    request.num_tokens,
                ));
            }
            cu_tokens.push(
                flat_token_ids
                    .len()
                    .try_into()
                    .expect("qwen3.5 MTP cumulative token count must fit u32"),
            );
        }
        let gdn_state_txns = requests
            .iter()
            .enumerate()
            .map(|(req_index, request)| {
                GDNStateTxn::new(
                    microbatch.token_indices()[req_index],
                    request
                        .num_tokens
                        .try_into()
                        .expect("qwen3.5 MTP request token count must fit u32"),
                    0,
                )
            })
            .collect();
        Qwen35MTPModuleBatch {
            microbatch: Qwen35Microbatch::new(
                microbatch.req_slots().to_vec(),
                microbatch.block_indices().to_vec(),
                microbatch.token_indices().to_vec(),
                flat_token_ids,
                cu_tokens,
                gdn_state_txns,
                vec![Vec::new(); requests.len()],
                microbatch.sampler_configs().to_vec(),
                flat_sample_mask,
            ),
            input_gather_flat_indices,
            draft_distribution_indices,
            sampler_configs,
            sample_positions,
        }
    }

    fn forward_mtp_batch(
        &mut self,
        microbatch: &Qwen35Microbatch,
        main_hidden: Rc<Buffer>,
        decisions: &mut [Qwen35DecodeDecision],
    ) -> ModelOutputTiming {
        let mut timing = ModelOutputTiming::default();
        let decode_req_indices = (0..microbatch.num_reqs())
            .filter(|&req_index| microbatch.is_decode_req(req_index))
            .collect::<Vec<_>>();
        assert_eq!(
            decisions.len(),
            decode_req_indices.len(),
            "qwen3.5 MTP proposal requires one decision per decode request"
        );
        let mut requests = self.mtp_requests(microbatch, decisions);
        if self.mtp.is_some() {
            let module_batch = self.mtp_batch(microbatch, &mut requests);
            let num_tokens = module_batch.microbatch.total_tokens();
            let num_target_hidden_states = module_batch.sampler_configs.len();
            self.write_token_ids(module_batch.microbatch.flat_token_ids());
            let mtp_gqa_state = self.mtp_gqa_state.as_ref().expect("qwen3.5 MTP requires GQA state");
            let mtp_gqa_shape = mtp_gqa_state.prepare_metadata(&module_batch.microbatch);
            self.mtp_input_gather_flat_indices
                .write_typed(0, &module_batch.input_gather_flat_indices);
            if num_target_hidden_states > 0 {
                self.draft_distribution_indices
                    .write_typed(0, &module_batch.draft_distribution_indices);
            }
            self.mtp
                .as_ref()
                .expect("qwen3.5 MTP batch requires the MTP module")
                .component()
                .validate_batch(&module_batch.microbatch);
            let mtp_key = Qwen35MTPReplayKey::new(0, num_tokens, mtp_gqa_shape);
            let mtp_embed_key = Qwen35MTPEmbedReplayKey::new(0, num_tokens);
            let mtp_hidden_input = Rc::clone(
                self.mtp_hidden_input
                    .as_ref()
                    .expect("qwen3.5 MTP requires its body-input workspace"),
            );
            let mtp_embed_build_start = Instant::now();
            let input = Qwen35MTPEmbedArgs {
                num_tokens: num_tokens.try_into().expect("qwen3.5 MTP token count must fit u32"),
                prev_hidden_source: &main_hidden,
                prev_hidden_indices: &self.mtp_input_gather_flat_indices,
                prev_hidden_input: &self.mtp_previous_hidden,
                token_ids: &self.token_ids,
                token_hidden_input: &self.token_hidden_input,
                hidden_output: &mtp_hidden_input,
            };
            let runtime = MetalReplayRuntime::new(self.runtime.stream());
            let (recorded_key, mtp_embed_cache_hit) = self
                .mtp_embed
                .as_mut()
                .expect("qwen3.5 MTPEmbed replay build requires the MTP module")
                .record(&runtime, &input);
            assert_eq!(
                recorded_key, mtp_embed_key,
                "qwen3.5 MTPEmbed replay input must match its key"
            );
            if !mtp_embed_cache_hit {
                timing.mtp_build_elapsed += mtp_embed_build_start.elapsed();
            }
            let mtp_build_start = Instant::now();
            let input = Qwen35MTPArgs {
                num_tokens: num_tokens.try_into().expect("qwen3.5 MTP token count must fit u32"),
                hidden_input: &mtp_hidden_input,
                hidden_output: &self.hidden_output,
                gqa: mtp_gqa_state.metadata(),
                pages: self.pages.buffer(),
            };
            let runtime = MetalReplayRuntime::new(self.runtime.stream());
            let (recorded_key, mtp_cache_hit) = self
                .mtp
                .as_mut()
                .expect("qwen3.5 MTP replay build requires the MTP module")
                .record(&runtime, &input);
            assert_eq!(recorded_key, mtp_key, "qwen3.5 MTP replay input must match its key");
            if !mtp_cache_hit {
                timing.mtp_build_elapsed += mtp_build_start.elapsed();
            }
            let mtp_replay_start = Instant::now();
            let empty_arguments = ReplayArguments::new();
            if num_target_hidden_states > 0 {
                let hidden_output = Rc::clone(&self.hidden_output);
                let gather_unembed_key =
                    self.prepare_gather_unembed_replay(&module_batch.microbatch, &hidden_output);
                let (draft_sample_key, draft_sample_arguments) =
                    self.prepare_draft_sample_replay(&module_batch.sampler_configs, &module_batch.sample_positions);
                let mtp_embed_replay = self
                    .mtp_embed
                    .as_ref()
                    .expect("qwen3.5 MTPEmbed replay requires the MTP module")
                    .replay(&mtp_embed_key);
                let mtp_replay = self
                    .mtp
                    .as_ref()
                    .expect("qwen3.5 MTP replay requires the MTP module")
                    .replay(&mtp_key);
                let gather_unembed_replay = self.gather_unembed.replay(&gather_unembed_key);
                let draft_sample_replay = self.draft_sampling.replay(&draft_sample_key);
                self.replay_runtime()
                    .submit_replay_sequence(&[
                        ReplayExecution::new(mtp_embed_replay, &empty_arguments),
                        ReplayExecution::new(mtp_replay, &empty_arguments),
                        ReplayExecution::new(gather_unembed_replay, &empty_arguments),
                        ReplayExecution::new(draft_sample_replay, &draft_sample_arguments),
                    ])
                    .wait();
            } else {
                let mtp_embed_replay = self
                    .mtp_embed
                    .as_ref()
                    .expect("qwen3.5 MTPEmbed replay requires the MTP module")
                    .replay(&mtp_embed_key);
                let mtp_replay = self
                    .mtp
                    .as_ref()
                    .expect("qwen3.5 MTP replay requires the MTP module")
                    .replay(&mtp_key);
                self.replay_runtime()
                    .submit_replay_sequence(&[
                        ReplayExecution::new(mtp_embed_replay, &empty_arguments),
                        ReplayExecution::new(mtp_replay, &empty_arguments),
                    ])
                    .wait();
            }
            timing.mtp_replay_elapsed += mtp_replay_start.elapsed();
            timing.mtp_modules += 1;
            if num_target_hidden_states > 0 {
                let mtp_read_start = Instant::now();
                let draft_token_ids = self
                    .sampler_output
                    .token_ids
                    .read_typed::<i32>(0, num_target_hidden_states);
                let draft_probs = self
                    .sampler_output
                    .token_probs
                    .read_typed::<f32>(0, num_target_hidden_states);
                timing.mtp_read_elapsed += mtp_read_start.elapsed();
                for (sample_index, &req_index) in decode_req_indices.iter().enumerate() {
                    let request = &mut requests[req_index];
                    let decision_index = request
                        .decision_index
                        .expect("qwen3.5 MTP decode request requires a decision");
                    let draft_token = draft_token_ids[sample_index]
                        .try_into()
                        .expect("qwen3.5 sampler returned a negative draft token ID");
                    self.spec_probs
                        .set_expected_draft_token(microbatch.req_slots()[req_index], 0, draft_token);
                    decisions[decision_index].spec_tokens.push(draft_token);
                    decisions[decision_index].spec_probs.push(draft_probs[sample_index]);
                    request.next_token_id = Some(
                        draft_token
                            .try_into()
                            .expect("qwen3.5 draft token ID must fit the model i32 token domain"),
                    );
                }
            }
            trace::qwen35_state(|| {
                format!(
                    "event=mtp_forward_module mtp_module_index={} num_tokens={} num_reqs={} \
                     num_target_hidden_states={} mtp_embed_cache_hit={} cache_hit={} mtp_embed_key={:?} mtp_key={:?}",
                    0,
                    num_tokens,
                    requests.len(),
                    num_target_hidden_states,
                    mtp_embed_cache_hit,
                    mtp_cache_hit,
                    mtp_embed_key,
                    mtp_key,
                )
            });
        }
        trace_decisions("mtp_propose_done", decisions);
        timing
    }

    fn forward_mtp(
        &mut self,
        recorder: &mut Qwen35ModelRecorder,
        model_batch_req: &Qwen35ModelBatchRequest,
        model_batch_hidden: &Rc<Buffer>,
        mut sampled_output: Qwen35DecodeOutput,
    ) -> Qwen35DecodeOutput {
        if self.mtp.is_none() {
            return sampled_output;
        }
        assert!(
            Rc::ptr_eq(model_batch_hidden, &self.hidden_output),
            "qwen3.5 speculator must consume the executor final-norm hidden workspace"
        );
        let microbatch = model_batch_req.microbatch();
        let num_decode_reqs = (0..microbatch.num_reqs())
            .filter(|&req_index| microbatch.is_decode_req(req_index))
            .count();
        assert_eq!(
            sampled_output.decisions.len(),
            num_decode_reqs,
            "qwen3.5 speculator requires one decision per decode request"
        );
        if !recorder.main_stage_submitted {
            sampled_output.timing.main_replay_elapsed += self.submit_main_stage(recorder);
        }
        let timing =
            self.forward_mtp_batch(microbatch, Rc::clone(model_batch_hidden), &mut sampled_output.decisions);
        sampled_output.timing.add_assign(timing);
        sampled_output
    }
}
