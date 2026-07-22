impl Qwen35Executor {
    fn validate_input(&self, core_batch_req: &BatchDeviceRequest) {
        assert!(
            core_batch_req.dev_reqs.len() <= self.config.max_requests,
            "qwen3.5 replay executor supports at most {} requests per batch, got {}",
            self.config.max_requests,
            core_batch_req.dev_reqs.len()
        );
        assert!(
            core_batch_req.token_cost() <= self.config.max_tokens,
            "qwen3.5 replay executor supports at most {} tokens per batch, got {}",
            self.config.max_tokens,
            core_batch_req.token_cost()
        );
        for request in &core_batch_req.dev_reqs {
            assert!(
                request.decoder_query_tokens.token_consumption() <= self.config.max_tokens_per_request,
                "qwen3.5 replay executor request tokens={} exceed scheduler max_tokens_per_request={}",
                request.decoder_query_tokens.token_consumption(),
                self.config.max_tokens_per_request
            );
        }
        let max_context_tokens = core_batch_req
            .dev_reqs
            .iter()
            .map(|request| {
                request
                    .decoder_query_tokens
                    .token_index()
                    .checked_add(request.decoder_query_tokens.token_consumption())
                    .expect("qwen3.5 GQA request context length overflow")
            })
            .max()
            .expect("qwen3.5 replay executor requires at least one request");
        let num_physical_pages =
            max_context_tokens.div_ceil(self.main_gqa_state.backend().num_tokens_per_page() as usize);
        let page_capacity = self.gqa_page_table_layout.num_physical_pages_per_request();
        assert!(
            num_physical_pages <= page_capacity,
            "qwen3.5 GQA request context needs {} physical pages but capacity is {}",
            num_physical_pages,
            page_capacity
        );
    }

    pub fn num_main_gqa_page_ids_per_block(&self) -> usize {
        let layout = self.gqa_page_table_layout;
        let num_page_ids = u64::from(layout.num_gqa_layers)
            .checked_mul(u64::from(layout.num_page_ids_per_block))
            .expect("qwen3.5 main GQA page IDs per block overflow");
        usize::try_from(num_page_ids).expect("qwen3.5 main GQA page IDs per block must fit usize")
    }

    pub fn num_mtp_gqa_page_ids_per_block(&self) -> Vec<usize> {
        self.mtp_gqa_state
            .iter()
            .map(|state| {
                let layout = state.request_page_table().layout();
                let num_page_ids = u64::from(layout.num_gqa_layers)
                    .checked_mul(u64::from(layout.num_page_ids_per_block))
                    .expect("qwen3.5 MTP GQA page IDs per block overflow");
                usize::try_from(num_page_ids).expect("qwen3.5 MTP GQA page IDs per block must fit usize")
            })
            .collect()
    }

    pub fn num_gdn_state_page_ids_per_block(&self) -> usize {
        self.main_gdn_state.num_pages_per_state_slot()
    }

    fn commit(&mut self, compute_seq: RawComputeSlotSeq, decisions: &[Qwen35DecodeDecision]) {
        let verified_state_versions = self.pending_transactions.commit(compute_seq, decisions);
        trace_decisions("model_commit_decisions", decisions);
        trace::qwen35_state(|| {
            format!(
                "event=model_commit verified_state_versions={:?}",
                verified_state_versions
            )
        });
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        self.main_gdn_state
            .commit(&runtime, self.pages.buffer(), &verified_state_versions);
        // Publish is submitted asynchronously here and overlaps returning the
        // response to runtime core. The next prepare/reset waits before reusing
        // the shared GDN page-I/O staging and live-state resources.
    }

    fn finish_cache_publish(&mut self) {
        let start = Instant::now();
        self.main_gdn_state.finish_publish();
        trace::qwen35_state(|| format!("event=cache_publish_wait elapsed_us={}", start.elapsed().as_micros()));
    }

    fn write_token_ids(&self, token_ids: &[i32]) {
        self.assert_expected_draft_tokens_fit(token_ids.len());
        self.token_ids.write_typed(0, token_ids);
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        self.finish_cache_publish();
        self.request_sampling.reset(request_slots);
        self.main_gqa_state.reset_req_slots(request_slots);
        if let Some(mtp_gqa_state) = &self.mtp_gqa_state {
            mtp_gqa_state.reset_req_slots(request_slots);
        }
        self.spec_probs.reset_req_slots(request_slots);
        self.main_gdn_state.reset_req_slots(request_slots);
    }

    fn prepare_batch(&mut self, core_batch_req: &BatchDeviceRequest) -> Qwen35ModelBatchRequest {
        self.finish_cache_publish();
        let batch_seq = core_batch_req.seq;
        self.validate_input(core_batch_req);
        let sampler_configs = core_batch_req
            .dev_reqs
            .iter()
            .map(|request| {
                let seed = self
                    .request_sampling
                    .resolve(request.req_slot, request.sampling_config.seed);
                SamplerConfig::from_runtime(&request.sampling_config, seed)
            })
            .collect();
        let model_batch_request =
            Qwen35ModelBatchRequest::from_core_batch(core_batch_req, usize::from(self.mtp.is_some()), sampler_configs);
        let microbatch = model_batch_request.microbatch();
        trace::qwen35_state(|| {
            format!(
                "event=batch_from_core seq={} req_slots={:?} token_indices={:?} num_spec_tokens={:?} seeds={:?} \
                 total_tokens={} num_reqs={}",
                batch_seq,
                microbatch.req_slots(),
                microbatch.token_indices(),
                microbatch
                    .gdn_state_txns()
                    .iter()
                    .map(|txn| txn.num_spec_tokens)
                    .collect::<Vec<_>>(),
                microbatch
                    .sampler_configs()
                    .iter()
                    .map(SamplerConfig::seed)
                    .collect::<Vec<_>>(),
                microbatch.total_tokens(),
                microbatch.num_reqs()
            )
        });
        self.write_token_ids(microbatch.flat_token_ids());
        let prepare_start = Instant::now();
        let gqa_start = Instant::now();
        self.main_gqa_state.prepare_pages(core_batch_req);
        let gqa_shape = self.main_gqa_state.prepare_metadata(microbatch);
        let gqa_elapsed = gqa_start.elapsed();
        debug_assert_eq!(gqa_shape.num_tokens as usize, microbatch.total_tokens());
        let gdn_states_start = Instant::now();
        let gdn_prepared = self.main_gdn_state.prepare_states(microbatch);
        let gdn_states_elapsed = gdn_states_start.elapsed();
        let gdn_metadata_start = Instant::now();
        let gdn_shape = self.main_gdn_state.prepare_metadata(microbatch, &gdn_prepared);
        let gdn_metadata_elapsed = gdn_metadata_start.elapsed();
        debug_assert_eq!(gdn_shape.num_tokens as usize, microbatch.total_tokens());
        debug_assert_eq!(gdn_shape.num_reqs as usize, microbatch.num_reqs());
        if let Some(mtp_gqa_state) = &self.mtp_gqa_state {
            mtp_gqa_state.prepare_pages(core_batch_req);
        }
        let prepare_elapsed = prepare_start.elapsed();
        trace::qwen35_state(|| {
            format!(
                "event=prepare_sync seq={} gqa_us={} gdn_states_us={} gdn_metadata_us={} wall_us={}",
                batch_seq,
                gqa_elapsed.as_micros(),
                gdn_states_elapsed.as_micros(),
                gdn_metadata_elapsed.as_micros(),
                prepare_elapsed.as_micros()
            )
        });
        let restore_elapsed = self.submit_gdn_state_restore();
        trace::qwen35_state(|| {
            format!(
                "event=prepare_batch_done seq={} gdn_restore_us={}",
                batch_seq,
                restore_elapsed.as_micros()
            )
        });
        model_batch_request
    }

    fn commit_batch(
        &mut self,
        core_batch_req: BatchDeviceRequest,
        sampled_output: Qwen35DecodeOutput,
    ) -> BatchDeviceResponse {
        self.commit(core_batch_req.seq, &sampled_output.decisions);
        to_core_batch_resp(core_batch_req, sampled_output.decisions)
    }
}
