impl Qwen3Executor {
    fn validate_input(&self, core_batch_req: &BatchDeviceRequest) {
        assert!(
            core_batch_req.dev_reqs.len() <= self.config.max_requests,
            "qwen3 replay executor supports at most {} requests per batch, got {}",
            self.config.max_requests,
            core_batch_req.dev_reqs.len()
        );
        assert!(
            core_batch_req.token_cost() <= self.config.max_tokens,
            "qwen3 replay executor supports at most {} tokens per batch, got {}",
            self.config.max_tokens,
            core_batch_req.token_cost()
        );
        let max_spec_tokens = self.speculator.num_spec_tokens();
        for request in &core_batch_req.dev_reqs {
            assert!(
                request.decoder_query_tokens.token_consumption() <= self.config.max_tokens_per_request,
                "qwen3 replay executor request tokens={} exceed scheduler max_tokens_per_request={}",
                request.decoder_query_tokens.token_consumption(),
                self.config.max_tokens_per_request
            );
            let num_spec_tokens = request.decoder_query_tokens.num_spec_tokens();
            if max_spec_tokens == 0 {
                assert_eq!(
                    num_spec_tokens, 0,
                    "qwen3 executor without DSpark does not accept speculative input tokens"
                );
            } else {
                assert!(
                    num_spec_tokens <= max_spec_tokens,
                    "qwen3 speculative-token count exceeds the configured DSpark capacity"
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
                    .expect("qwen3 GQA request context length overflow")
            })
            .max()
            .expect("qwen3 replay executor requires at least one request");
        let num_physical_pages = max_context_tokens.div_ceil(self.main_gqa_state.num_tokens_per_page());
        let page_capacity = self.gqa_page_table_layout.num_physical_pages_per_request();
        assert!(
            num_physical_pages <= page_capacity,
            "qwen3 GQA request context needs {} physical pages but capacity is {}",
            num_physical_pages,
            page_capacity
        );
        if self.speculator.is_dspark() {
            let dspark = &self.speculator.dspark().execution;
            let num_dspark_pages = max_context_tokens.div_ceil(dspark.num_tokens_per_page());
            let dspark_page_capacity = dspark.num_physical_pages_per_request();
            assert!(
                num_dspark_pages <= dspark_page_capacity,
                "qwen3 DSpark request context needs {} physical pages but capacity is {}",
                num_dspark_pages,
                dspark_page_capacity
            );
        }
    }

    pub fn model_config(&self) -> &Qwen3ModelConfig {
        &self.model_config
    }

    pub fn num_spec_tokens(&self) -> usize {
        self.speculator.num_spec_tokens()
    }

    pub fn num_main_gqa_page_ids_per_block(&self) -> usize {
        let layout = self.gqa_page_table_layout;
        let num_page_ids = (layout.num_gqa_layers as u64)
            .checked_mul(layout.num_page_ids_per_block as u64)
            .expect("qwen3 main GQA page IDs per block overflow");
        usize::try_from(num_page_ids).expect("qwen3 main GQA page IDs per block must fit usize")
    }

    pub fn num_kv_page_ids_per_block(&self) -> usize {
        self.num_gqa_page_ids_per_main_lane_block
    }

    fn prepare_gqa_page_ids(&self, batch: &BatchDeviceRequest) {
        let num_main_gqa_page_ids = self.num_main_gqa_page_ids_per_block();
        let num_speculator_gqa_page_ids = self.speculator.num_gqa_page_ids_per_main_lane_block();
        debug_assert_eq!(
            num_main_gqa_page_ids
                .checked_add(num_speculator_gqa_page_ids)
                .expect("qwen3 Main cache-lane page-ID count must fit usize"),
            self.num_gqa_page_ids_per_main_lane_block
        );
        for request in &batch.dev_reqs {
            let page_ids_by_block = request
                .decoder_sync_blocks
                .kv_page_ids()
                .first()
                .expect("qwen3 Main GQA request requires runtime cache lane 0");
            for (block_offset, page_ids) in page_ids_by_block.iter().enumerate() {
                let block_index = request
                    .decoder_sync_blocks
                    .block_index()
                    .checked_add(block_offset)
                    .expect("qwen3 Main cache-block index must fit usize");
                let (main_page_ids, speculator_page_ids) =
                    split_main_lane_page_ids(page_ids, num_main_gqa_page_ids, num_speculator_gqa_page_ids);
                self.main_gqa_state
                    .write_page_ids(request.req_slot, block_index, main_page_ids);
                self.speculator
                    .write_page_ids(request.req_slot, block_index, speculator_page_ids);
            }
        }
    }

    fn write_token_ids(&self, token_ids: &[i32]) {
        assert!(!token_ids.is_empty(), "qwen3 replay model requires at least one token");
        assert!(
            token_ids.len() <= self.layout.max_tokens as usize,
            "qwen3 replay model tokens={} exceed max_tokens={}",
            token_ids.len(),
            self.layout.max_tokens
        );
        self.token_ids.write_typed(0, token_ids);
    }
}
