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
            let num_spec_tokens = request.decoder_query_tokens.num_spec_tokens();
            let max_spec_tokens = self.num_speculative_tokens();
            if max_spec_tokens == 0 {
                assert_eq!(
                    num_spec_tokens, 0,
                    "qwen3.5 executor without a speculator does not accept speculative input tokens"
                );
            } else {
                assert!(
                    num_spec_tokens <= max_spec_tokens,
                    "qwen3.5 speculative-token count exceeds speculator capacity"
                );
            }
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
        if self.speculator.is_dspark() {
            let dspark = &self.speculator.dspark().execution;
            let num_dspark_pages = max_context_tokens.div_ceil(dspark.num_tokens_per_page());
            let dspark_page_capacity = dspark.num_physical_pages_per_request();
            assert!(
                num_dspark_pages <= dspark_page_capacity,
                "qwen3.5 DSpark request context needs {} physical pages but capacity is {}",
                num_dspark_pages,
                dspark_page_capacity
            );
        }
    }

    pub fn num_main_gqa_page_ids_per_block(&self) -> usize {
        let layout = self.gqa_page_table_layout;
        let num_page_ids = u64::from(layout.num_gqa_layers)
            .checked_mul(u64::from(layout.num_page_ids_per_block))
            .expect("qwen3.5 main GQA page IDs per block overflow");
        usize::try_from(num_page_ids).expect("qwen3.5 main GQA page IDs per block must fit usize")
    }

    pub fn num_main_lane_gqa_page_ids_per_block(&self) -> usize {
        self.num_runtime_page_ids_per_main_block
    }

    pub fn num_mtp_gqa_page_ids_per_block(&self) -> Vec<usize> {
        match &self.speculator {
            Qwen35Speculator::Vanilla | Qwen35Speculator::DSpark(_) => Vec::new(),
            Qwen35Speculator::Mtp(mtp) => {
                let layout = mtp.gqa_state.request_page_table().layout();
                let num_page_ids = u64::from(layout.num_gqa_layers)
                    .checked_mul(u64::from(layout.num_page_ids_per_block))
                    .expect("qwen3.5 MTP GQA page IDs per block overflow");
                vec![usize::try_from(num_page_ids).expect("qwen3.5 MTP GQA page IDs per block must fit usize")]
            },
        }
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
}
