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

    fn mtp_batch(&self, microbatch: &Qwen35Microbatch, requests: &mut [Qwen35MTPRequest]) -> Qwen35MTPBatch {
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
        Qwen35MTPBatch {
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

    fn record_mtp_embed(
        &mut self,
        recorder: &mut Qwen35ModelOpsRecorder,
        microbatch: &Qwen35Microbatch,
        main_hidden: Rc<Buffer>,
        decisions: &[Qwen35DecodeDecision],
    ) -> Rc<Buffer> {
        let decode_req_indices = (0..microbatch.num_reqs())
            .filter(|&req_index| microbatch.is_decode_req(req_index))
            .collect::<Vec<_>>();
        assert_eq!(
            decisions.len(),
            decode_req_indices.len(),
            "qwen3.5 MTP proposal requires one decision per decode request"
        );
        assert!(self.mtp.is_some(), "qwen3.5 MTP recording requires the MTP module");
        let mut requests = self.mtp_requests(microbatch, decisions);
        let module_batch = self.mtp_batch(microbatch, &mut requests);
        let num_tokens = module_batch.microbatch.total_tokens();
        let num_mtp_sample_rows = module_batch.sampler_configs.len();
        self.write_token_ids(module_batch.microbatch.flat_token_ids());
        let mtp_gqa_state = self.mtp_gqa_state.as_ref().expect("qwen3.5 MTP requires GQA state");
        let mtp_gqa_shape = mtp_gqa_state.prepare_metadata(
            module_batch.microbatch.req_slots(),
            module_batch.microbatch.token_indices(),
            module_batch.microbatch.cu_tokens(),
        );
        self.mtp_input_gather_flat_indices
            .write_typed(0, &module_batch.input_gather_flat_indices);
        if num_mtp_sample_rows > 0 {
            self.draft_distribution_indices
                .write_typed(0, &module_batch.draft_distribution_indices);
        }
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
            recorder.mtp_build_elapsed += mtp_embed_build_start.elapsed();
        }
        recorder.mtp_sample_req_slots = decode_req_indices
            .iter()
            .map(|&req_index| microbatch.req_slots()[req_index])
            .collect();
        recorder.mtp_sample_decision_indices = decode_req_indices
            .iter()
            .map(|&req_index| {
                requests[req_index]
                    .decision_index
                    .expect("qwen3.5 MTP decode request requires a decision")
            })
            .collect();
        trace::qwen35_state(|| {
            format!(
                "event=mtp_embed_record mtp_module_index={} num_tokens={} num_reqs={} num_sample_rows={} cache_hit={} \
                 key={:?}",
                0,
                num_tokens,
                requests.len(),
                num_mtp_sample_rows,
                mtp_embed_cache_hit,
                mtp_embed_key,
            )
        });
        recorder.mtp_embed_key = Some(mtp_embed_key);
        recorder.mtp_key = Some(mtp_key);
        recorder.mtp_microbatch = Some(module_batch.microbatch);
        recorder.mtp_sampler_configs = module_batch.sampler_configs;
        recorder.mtp_sample_positions = module_batch.sample_positions;
        recorder.mtp_embed_cache_hit = mtp_embed_cache_hit;
        mtp_hidden_input
    }

    fn record_mtp(&mut self, recorder: &mut Qwen35ModelOpsRecorder, mtp_hidden_input: Rc<Buffer>) -> Rc<Buffer> {
        assert!(
            Rc::ptr_eq(
                &mtp_hidden_input,
                self.mtp_hidden_input
                    .as_ref()
                    .expect("qwen3.5 MTP requires its body-input workspace")
            ),
            "qwen3.5 MTP must consume the MTPEmbed hidden workspace"
        );
        let microbatch = recorder
            .mtp_microbatch
            .as_ref()
            .expect("qwen3.5 MTP recording requires the MTP microbatch");
        self.mtp
            .as_ref()
            .expect("qwen3.5 MTP batch requires the MTP module")
            .component()
            .validate_batch(microbatch);
        let mtp_key = recorder
            .mtp_key
            .as_ref()
            .expect("qwen3.5 MTP recording requires its replay key");
        let num_tokens = microbatch.total_tokens();
        let mtp_gqa_state = self.mtp_gqa_state.as_ref().expect("qwen3.5 MTP requires GQA state");
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
        assert_eq!(&recorded_key, mtp_key, "qwen3.5 MTP replay input must match its key");
        if !mtp_cache_hit {
            recorder.mtp_build_elapsed += mtp_build_start.elapsed();
        }
        trace::qwen35_state(|| {
            format!(
                "event=mtp_record mtp_module_index={} num_tokens={} num_reqs={} num_sample_rows={} \
                 mtp_embed_cache_hit={} cache_hit={} key={:?}",
                0,
                num_tokens,
                microbatch.num_reqs(),
                recorder.num_mtp_sample_rows(),
                recorder.mtp_embed_cache_hit,
                mtp_cache_hit,
                mtp_key,
            )
        });
        Rc::clone(&self.hidden_output)
    }

    fn record_mtp_gather_unembed(&mut self, recorder: &mut Qwen35ModelOpsRecorder, mtp_hidden_output: &Rc<Buffer>) {
        assert!(
            Rc::ptr_eq(mtp_hidden_output, &self.hidden_output),
            "qwen3.5 MTP GatherUnembed must consume the MTP hidden workspace"
        );
        if recorder.num_mtp_sample_rows() == 0 {
            return;
        }
        let gather_unembed_key = self.prepare_gather_unembed_replay(
            recorder
                .mtp_microbatch
                .as_ref()
                .expect("qwen3.5 MTP GatherUnembed requires the MTP microbatch"),
            mtp_hidden_output,
        );
        recorder.mtp_gather_unembed_key = Some(gather_unembed_key);
    }

    fn record_mtp_sampling(&mut self, recorder: &mut Qwen35ModelOpsRecorder) {
        if recorder.num_mtp_sample_rows() == 0 {
            return;
        }
        let (mtp_sampling_key, mtp_sampling_arguments) =
            self.prepare_mtp_sampling_replay(&recorder.mtp_sampler_configs, &recorder.mtp_sample_positions);
        recorder.mtp_sampling_key = Some(mtp_sampling_key);
        recorder.mtp_sampling_arguments = mtp_sampling_arguments;
    }

    fn submit_mtp_recording(&self, recorder: &Qwen35ModelOpsRecorder) -> MetalReplaySubmission {
        let mtp_embed_key = recorder
            .mtp_embed_key
            .as_ref()
            .expect("qwen3.5 MTP submission requires MTPEmbed replay");
        let mtp_key = recorder
            .mtp_key
            .as_ref()
            .expect("qwen3.5 MTP submission requires MTP replay");
        let mtp_embed_replay = self
            .mtp_embed
            .as_ref()
            .expect("qwen3.5 MTPEmbed replay requires the MTP module")
            .replay(mtp_embed_key);
        let mtp_replay = self
            .mtp
            .as_ref()
            .expect("qwen3.5 MTP replay requires the MTP module")
            .replay(mtp_key);
        let empty_arguments = ReplayArguments::new();
        if recorder.num_mtp_sample_rows() == 0 {
            return self.replay_runtime().submit_replay_sequence(&[
                ReplayExecution::new(mtp_embed_replay, &empty_arguments),
                ReplayExecution::new(mtp_replay, &empty_arguments),
            ]);
        }
        let gather_unembed_key = recorder
            .mtp_gather_unembed_key
            .as_ref()
            .expect("qwen3.5 MTP sampled output requires GatherUnembed replay");
        let mtp_sampling_key = recorder
            .mtp_sampling_key
            .as_ref()
            .expect("qwen3.5 MTP sampled output requires Sampling replay");
        self.replay_runtime().submit_replay_sequence(&[
            ReplayExecution::new(mtp_embed_replay, &empty_arguments),
            ReplayExecution::new(mtp_replay, &empty_arguments),
            ReplayExecution::new(self.gather_unembed.replay(gather_unembed_key), &empty_arguments),
            ReplayExecution::new(
                self.mtp_sampling.replay(mtp_sampling_key),
                &recorder.mtp_sampling_arguments,
            ),
        ])
    }

    fn read_mtp_proposal(
        &mut self,
        recorder: &Qwen35ModelOpsRecorder,
        decisions: &mut [Qwen35DecodeDecision],
        replay_elapsed: Duration,
    ) -> ModelOutputTiming {
        let mut timing = ModelOutputTiming {
            spec_build_elapsed: recorder.mtp_build_elapsed,
            spec_replay_elapsed: replay_elapsed,
            spec_passes: 1,
            ..ModelOutputTiming::default()
        };
        let num_mtp_sample_rows = recorder.num_mtp_sample_rows();
        if num_mtp_sample_rows > 0 {
            assert_eq!(
                recorder.mtp_sample_decision_indices.len(),
                num_mtp_sample_rows,
                "qwen3.5 MTP decisions must match draft sampling rows"
            );
            let mtp_read_start = Instant::now();
            let draft_token_ids = self.sampler_output.token_ids.read_typed::<i32>(0, num_mtp_sample_rows);
            let draft_probs = self
                .sampler_output
                .token_probs
                .read_typed::<f32>(0, num_mtp_sample_rows);
            timing.spec_read_elapsed += mtp_read_start.elapsed();
            for sample_index in 0..num_mtp_sample_rows {
                let draft_token = draft_token_ids[sample_index]
                    .try_into()
                    .expect("qwen3.5 sampler returned a negative draft token ID");
                self.spec_probs
                    .set_expected_draft_token(recorder.mtp_sample_req_slots[sample_index], 0, draft_token);
                let decision = &mut decisions[recorder.mtp_sample_decision_indices[sample_index]];
                decision.spec_tokens.push(draft_token);
                decision.spec_probs.push(draft_probs[sample_index]);
            }
        }
        trace_decisions("mtp_propose_done", decisions);
        timing
    }
}
