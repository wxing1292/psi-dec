use std::cmp::min;
use std::iter;
use std::sync::Arc;

use inference_runtime_macro::sanity_check;
use smallvec::SmallVec;

use crate::compute::QueryTokens;
use crate::compute::SampledTokens;
use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::ResourceSegment;
use crate::runtime::decoder::trie_cache::AllocateMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::BlockMetadata;
use crate::runtime::decoder::trie_cache::CommitMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::CommitMultiLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::DecoderBlock;
use crate::runtime::decoder::trie_cache::DecoderBlocks;
use crate::runtime::decoder::trie_cache::InitBlockOnceResult;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::ReserveMultiLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::TrieDecoderBlocks;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::UninitBlockOnceResult;
use crate::runtime::decoder::trie_cache::blocks::cache_tokens;
use crate::runtime::decoder::trie_cache::blocks::pop_front_queued_tokens;
use crate::runtime::decoder::trie_cache::blocks::push_front_queued_tokens;
use crate::runtime::decoder::trie_cache::blocks::push_tokens;
use crate::runtime::decoder::trie_cache::blocks::schedule_tokens;
use crate::runtime::decoder::trie_cache::blocks::unschedule_tokens;
use crate::runtime::resource::ResourceID;

impl<const N: usize, const P: usize, const L: usize, BC> DecoderBlocks for TrieDecoderBlocks<N, P, L, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    fn ready_token_slots(&self) -> usize {
        self.semi_immutable_blocks
            .iter()
            .map(|block_vec| block_vec[0].ready_token_slots())
            .sum::<usize>()
            + self
                .mutable_blocks
                .iter()
                .map(|block_vec| block_vec[0].ready_token_slots())
                .sum::<usize>()
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn init_block_once(&mut self) -> InitBlockOnceResult {
        let num_cachable_tokens = self.num_queued_tokens().saturating_sub(L - 1);
        debug_assert!(
            num_cachable_tokens == 0
                || self
                    .mutable_blocks
                    .iter()
                    .all(|block_vec| N == block_vec[0].total_tokens().len())
        );

        let block_index = self.immutable_blocks.len() + self.semi_immutable_blocks.len() + self.mutable_blocks.len();
        let annotation_vec = self.block_annotation_vec(block_index);
        let resource_ids = annotation_resource_ids(&annotation_vec);
        let missing_resource_ids = resource_ids
            .iter()
            .copied()
            .filter(|resource_id| {
                self.resources
                    .get(resource_id)
                    .expect("cache block annotation must reference a resource owned by its decoder blocks")
                    .is_symbolic()
            })
            .collect::<Vec<_>>();

        let num_tokens = min(num_cachable_tokens, N);
        let tokens = if num_tokens == 0 {
            vec![]
        } else {
            let tokens = pop_front_queued_tokens::<L>(&mut self.queued_tokens, num_tokens);
            debug_assert_eq!(num_tokens + L - 1, tokens.len());
            tokens
        };
        let use_cache = num_tokens == N
            && !self.queued_tokens.is_empty()
            && self.mutable_blocks.is_empty()
            && self.semi_immutable_blocks.is_empty();
        if !use_cache {
            if !missing_resource_ids.is_empty() {
                push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                return InitBlockOnceResult::ResourceNotFound {
                    ready_token_slots: self.ready_token_slots(),
                    resource_ids: missing_resource_ids,
                };
            }
            match self.block_cache.alloc_mutable_block::<N>() {
                AllocateMultiLaneMutableBlockResult::Mutable { mut block_vec } => {
                    for (block, annotations) in block_vec.iter_mut().zip(annotation_vec) {
                        block.insert_annotations(annotations);
                    }
                    push_tokens::<N, L>(&mut block_vec, &tokens);
                    self.mutable_blocks.push_back(block_vec);
                    InitBlockOnceResult::Success {
                        ready_token_slots: self.ready_token_slots(),
                    }
                },
                AllocateMultiLaneMutableBlockResult::ResourceLimitExceeded => {
                    push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                    InitBlockOnceResult::ResourceLimitExceeded
                },
            }
        } else {
            debug_assert_eq!(N, num_tokens);
            debug_assert!(self.mutable_blocks.is_empty());
            debug_assert!(self.semi_immutable_blocks.is_empty());

            let parent_trie_node_key_vec = self.parent_trie_node_key_vec(block_index);
            let block_metadata_vec: [BlockMetadata<N>; L] = parent_trie_node_key_vec
                .into_iter()
                .zip(annotation_vec)
                .zip(tokens.windows(num_tokens))
                .map(|((parent_trie_node_key, annotations), tokens)| {
                    BlockMetadata::new(parent_trie_node_key, annotations, tokens.to_vec().into())
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            match self.block_cache.reserve_semi_immutable_block(block_metadata_vec) {
                ReserveMultiLaneSemiImmutableBlockResult::Mutable { mut block_vec } => {
                    if !missing_resource_ids.is_empty() {
                        push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                        self.block_cache.free_mutable_block(block_vec);
                        return InitBlockOnceResult::ResourceNotFound {
                            ready_token_slots: self.ready_token_slots(),
                            resource_ids: missing_resource_ids,
                        };
                    }
                    push_tokens::<N, L>(&mut block_vec, &tokens);
                    self.mutable_blocks.push_back(block_vec);
                    InitBlockOnceResult::Success {
                        ready_token_slots: self.ready_token_slots(),
                    }
                },
                ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { block_vec } => {
                    if !missing_resource_ids.is_empty() {
                        push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                        self.block_cache.free_semi_immutable_block(block_vec);
                        return InitBlockOnceResult::ResourceNotFound {
                            ready_token_slots: self.ready_token_slots(),
                            resource_ids: missing_resource_ids,
                        };
                    }
                    self.semi_immutable_blocks.push_back(block_vec);
                    InitBlockOnceResult::Success {
                        ready_token_slots: self.ready_token_slots(),
                    }
                },
                ReserveMultiLaneSemiImmutableBlockResult::Immutable { block_vec } => {
                    self.immutable_blocks.push(block_vec);
                    InitBlockOnceResult::Success {
                        ready_token_slots: self.ready_token_slots(),
                    }
                },
                ReserveMultiLaneSemiImmutableBlockResult::Wait { wait, .. } => {
                    push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                    InitBlockOnceResult::Await { wait }
                },
                ReserveMultiLaneSemiImmutableBlockResult::ResourceLimitExceeded { .. } => {
                    push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                    InitBlockOnceResult::ResourceLimitExceeded
                },
            }
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn uninit_block_once(&mut self) -> UninitBlockOnceResult {
        if let Some(block_vec) = self.mutable_blocks.pop_back() {
            push_front_queued_tokens::<1>(&mut self.queued_tokens, block_vec[0].total_tokens().iter().copied());
            self.block_cache.free_mutable_block(block_vec);
        } else if let Some(block_vec) = self.semi_immutable_blocks.pop_back() {
            push_front_queued_tokens::<1>(&mut self.queued_tokens, block_vec[0].total_tokens().iter().copied());
            self.block_cache.free_semi_immutable_block(block_vec);
        } else if let Some(block_vec) = self.immutable_blocks.pop() {
            push_front_queued_tokens::<1>(&mut self.queued_tokens, block_vec[0].total_tokens().iter().copied());
        }

        UninitBlockOnceResult::Success {
            cached_token_slots: self.num_cached_tokens(),
        }
    }

    // NOTE:
    // * main module    -> cache lane == 0
    // * MTP 0 module   -> cache lane == 1
    // ...
    // * MTP L-2 module -> cache lane == L-1
    //
    // prefill:
    //  there must be >= L tokens (ready tokens + queued tokens) for main model
    //  prefill request must have token len T >= L
    //  * main model using tokens[0..len-L+1]
    //  * MTP 0 using tokens[1..len-L+2]
    //  * MTP L-2 using tokens[L..len]
    //
    //  there are **len-L+1** tokens applicable to each cache line, each model forward
    //  increase cache index by **len-L+1**
    // decode:
    //  there must be >= 1 tokens (ready tokens + queued tokens) for main model
    //  decode request does not necessary increase cache index
    //  input:
    //  * tokens with len T
    //  * spec tokens with len S
    //  output:
    //  * validated tokens with len V
    //  * sampled token with len 1
    //
    //  there are **min(T+V, T+V+1-L+1)** tokens applicable to each cache line, each model forward
    //  increase cache index by **min(T+V, T+V+1-L+1)**

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn prepare(&mut self, token_budget: usize) -> Option<QueryTokens> {
        debug_assert!(0 < token_budget);
        // caller is expected to init enough ready token slots via init_block_once
        debug_assert!(token_budget <= self.ready_token_slots());

        let num_ready_tokens = self.num_ready_tokens();
        let num_queued_tokens = self.num_queued_tokens();
        let num_validated_tokens = num_ready_tokens + num_queued_tokens;
        debug_assert!(num_validated_tokens != 0);

        let mut token_index = None;
        let mut block_index = self.immutable_blocks.len();
        let mut tokens = Vec::with_capacity(token_budget);
        'schedule_loop: {
            for block_vec in self.semi_immutable_blocks.iter_mut() {
                let main_block = &mut block_vec[0];
                let num_ready_tokens = main_block.ready_tokens().len();
                if num_ready_tokens == 0 {
                    block_index += 1;
                    continue;
                }

                let _ = token_index.get_or_insert(
                    block_index * N + main_block.cached_tokens().len() + main_block.scheduled_tokens().len(),
                );
                tokens.extend_from_slice(schedule_tokens::<N, L, _>(
                    block_vec,
                    min(num_ready_tokens, token_budget - tokens.len()),
                ));
                if tokens.len() >= token_budget {
                    debug_assert_eq!(token_budget, tokens.len());
                    break 'schedule_loop;
                }
                block_index += 1;
            }

            for block_vec in self.mutable_blocks.iter_mut() {
                let main_block = &mut block_vec[0];
                let num_ready_tokens = main_block.ready_tokens().len();
                if num_ready_tokens == 0 {
                    block_index += 1;
                    continue;
                }

                let _ = token_index.get_or_insert(
                    block_index * N + main_block.cached_tokens().len() + main_block.scheduled_tokens().len(),
                );
                tokens.extend_from_slice(schedule_tokens::<N, L, _>(
                    block_vec,
                    min(num_ready_tokens, token_budget - tokens.len()),
                ));
                if tokens.len() >= token_budget {
                    debug_assert_eq!(token_budget, tokens.len());
                    break 'schedule_loop;
                }
                block_index += 1;
            }
        }

        if token_budget < num_validated_tokens {
            debug_assert_eq!(tokens.len(), token_budget);
            debug_assert!(token_index.is_some());
            debug_assert!(L - 1 <= self.num_queued_tokens());
            tokens.extend(self.queued_tokens.iter().take(L - 1).copied());
            Some(QueryTokens::Prefill {
                epoch: self.epoch,
                token_index: token_index.unwrap(),
                tokens,
                window: token_budget,
            })
        } else {
            debug_assert!(tokens.len() <= token_budget);
            debug_assert!(self.num_queued_tokens() < L);
            tokens.extend(&self.queued_tokens);
            debug_assert!(tokens.len() <= token_budget);
            let num_selected_spec_tokens = token_budget - tokens.len();
            debug_assert!(num_selected_spec_tokens <= self.spec_tokens.len());
            self.spec_tokens.truncate(num_selected_spec_tokens);
            self.spec_probs.truncate(num_selected_spec_tokens);
            self.spec_confidences.truncate(num_selected_spec_tokens);
            let spec_tokens = self.spec_tokens.clone();

            Some(QueryTokens::Decode {
                epoch: self.epoch,
                token_index: token_index.unwrap_or(self.num_cached_tokens() + self.num_scheduled_tokens()),
                tokens,
                spec_tokens,
            })
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn cancel(&mut self, query_tokens: QueryTokens) {
        let tokens = match query_tokens {
            QueryTokens::Prefill {
                epoch,
                token_index,
                tokens,
                window,
            } => {
                debug_assert_eq!(self.epoch, epoch);
                debug_assert!(token_index < self.num_total_tokens());
                debug_assert!(!tokens.is_empty());
                debug_assert!(1 <= window);
                debug_assert_eq!(window, tokens.len() - (L - 1));

                tokens
            },
            QueryTokens::Decode {
                epoch,
                token_index,
                tokens,
                spec_tokens,
            } => {
                debug_assert_eq!(self.epoch, epoch);
                debug_assert!(token_index < self.num_total_tokens());
                debug_assert!(!tokens.is_empty());
                debug_assert_eq!(self.spec_tokens, spec_tokens);

                tokens
            },
        };

        let mut index_end = tokens.len().saturating_sub(L - 1);
        let mut index_start = index_end;
        'unschedule_loop: {
            for block_vec in self.mutable_blocks.iter_mut().rev() {
                index_start = index_end - min(index_end, block_vec[0].scheduled_tokens().len());
                unschedule_tokens::<N, L, _>(block_vec, &tokens, index_start, index_end);
                index_end = index_start;
                if index_end == 0 {
                    break 'unschedule_loop;
                }
            }

            for block_vec in self.semi_immutable_blocks.iter_mut().rev() {
                index_start = index_end - min(index_end, block_vec[0].scheduled_tokens().len());
                unschedule_tokens::<N, L, _>(block_vec, &tokens, index_start, index_end);
                index_end = index_start;
                if index_end == 0 {
                    break 'unschedule_loop;
                }
            }
        }

        debug_assert_eq!(0, index_start);
        debug_assert_eq!(0, index_end);
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn commit(&mut self, query_tokens: QueryTokens, sampled_tokens: SampledTokens) {
        let (
            ready_to_cached_tokens,
            ready_to_cached_token_window,
            queued_to_cached_tokens,
            queued_to_cached_token_window,
        ) = match (query_tokens, sampled_tokens) {
            (
                QueryTokens::Prefill {
                    epoch: input_epoch,
                    token_index: input_token_index,
                    tokens: input_tokens,
                    ..
                },
                SampledTokens::Prefill { epoch: output_epoch },
            ) => {
                debug_assert_eq!(self.epoch, input_epoch);
                debug_assert_eq!(self.epoch, output_epoch);
                debug_assert!(input_token_index < self.num_total_tokens());
                debug_assert!(!input_tokens.is_empty());

                let ready_to_cached_tokens = input_tokens;
                let ready_to_cached_token_window = ready_to_cached_tokens.len().saturating_sub(L - 1);
                let queued_to_cached_tokens: Vec<_> = ready_to_cached_tokens
                    .iter()
                    .copied()
                    .skip(ready_to_cached_tokens.len().saturating_sub(L - 1))
                    .collect();
                let queued_to_cached_token_window = 0;
                (
                    ready_to_cached_tokens,
                    ready_to_cached_token_window,
                    queued_to_cached_tokens,
                    queued_to_cached_token_window,
                )
            },
            (
                QueryTokens::Decode {
                    epoch: input_epoch,
                    token_index: input_token_index,
                    tokens: input_tokens,
                    spec_tokens: input_spec_tokens,
                },
                SampledTokens::Decode {
                    epoch: output_epoch,
                    validated_tokens: output_validated_tokens,
                    validated_probs: output_validated_probs,
                    sampled_token: output_sampled_token,
                    sampled_prob: output_sampled_prob,
                    spec_tokens: output_spec_tokens,
                    spec_probs: output_spec_probs,
                    spec_confidences: output_spec_confidences,
                },
            ) => {
                debug_assert_eq!(self.epoch, input_epoch);
                debug_assert_eq!(self.epoch, output_epoch);
                debug_assert!(input_token_index < self.num_total_tokens());
                debug_assert!(!input_tokens.is_empty());
                debug_assert!(
                    input_tokens.ends_with(self.queued_tokens.iter().copied().collect::<Vec<_>>().as_slice())
                );
                assert!(
                    input_spec_tokens.starts_with(&output_validated_tokens),
                    "validated tokens must equal a prefix of the speculative input suffix"
                );
                debug_assert_eq!(output_validated_tokens.len(), output_validated_probs.len());
                debug_assert_eq!(output_spec_tokens.len(), output_spec_probs.len());
                debug_assert_eq!(output_spec_tokens.len(), output_spec_confidences.len());

                let ready_to_cached_token_window = input_tokens.len().saturating_sub(L - 1);
                let queued_to_cached_token_window = min(
                    input_tokens.len() + output_validated_tokens.len(),
                    (input_tokens.len() + output_validated_tokens.len() + 1).saturating_sub(L - 1),
                ) - ready_to_cached_token_window;
                self.queued_tokens.extend(output_validated_tokens.iter().copied());
                self.queued_tokens.push_back(output_sampled_token);
                self.queued_tokens.drain(..queued_to_cached_token_window);
                self.spec_tokens = output_spec_tokens;
                self.spec_probs = output_spec_probs;
                self.spec_confidences = output_spec_confidences;

                let ready_to_cached_tokens = input_tokens;
                let queued_to_cached_tokens = ready_to_cached_tokens
                    .iter()
                    .copied()
                    .skip(ready_to_cached_tokens.len().saturating_sub(L - 1))
                    .chain(output_validated_tokens)
                    .chain(iter::once(output_sampled_token))
                    .collect();
                (
                    ready_to_cached_tokens,
                    ready_to_cached_token_window,
                    queued_to_cached_tokens,
                    queued_to_cached_token_window,
                )
            },
            _ => unreachable!(),
        };

        let mut index_start = 0;
        let mut index_end = 0;
        'commit_loop: {
            while let Some(mut block_vec) = self.semi_immutable_blocks.pop_front() {
                index_end = index_start
                    + min(
                        ready_to_cached_token_window - index_start,
                        block_vec[0].scheduled_tokens().len(),
                    );
                cache_tokens::<N, L, _>(&mut block_vec, &ready_to_cached_tokens, index_start, index_end);
                index_start = index_end;

                if block_vec[0].cached_tokens().len() != N {
                    self.semi_immutable_blocks.push_front(block_vec);
                    break 'commit_loop;
                } else {
                    match self.block_cache.commit_semi_immutable_block(block_vec) {
                        CommitMultiLaneSemiImmutableBlockResult::Immutable { block_vec } => {
                            self.immutable_blocks.push(block_vec);
                        },
                        CommitMultiLaneSemiImmutableBlockResult::ImmutableCollision { block_vec } => {
                            self.immutable_blocks.push(block_vec);
                            self.num_in_sync_blocks = 0;
                        },
                    }
                }
            }

            while let Some(mut block_vec) = self.mutable_blocks.pop_front() {
                index_end = index_start
                    + min(
                        ready_to_cached_token_window - index_start,
                        block_vec[0].scheduled_tokens().len(),
                    );
                cache_tokens::<N, L, _>(&mut block_vec, &ready_to_cached_tokens, index_start, index_end);
                index_start = index_end;

                if block_vec[0].cached_tokens().len() != N {
                    self.mutable_blocks.push_front(block_vec);
                    break 'commit_loop;
                } else {
                    debug_assert!(self.semi_immutable_blocks.is_empty());

                    let parent_trie_node_key_vec = self.parent_trie_node_key_vec(self.immutable_blocks.len());
                    match self
                        .block_cache
                        .commit_mutable_block(parent_trie_node_key_vec, block_vec)
                    {
                        CommitMultiLaneMutableBlockResult::Immutable { block_vec } => {
                            debug_assert!(self.semi_immutable_blocks.is_empty());
                            self.immutable_blocks.push(block_vec);
                        },
                        CommitMultiLaneMutableBlockResult::ImmutableCollision { block_vec } => {
                            debug_assert!(self.semi_immutable_blocks.is_empty());
                            self.immutable_blocks.push(block_vec);
                            self.num_in_sync_blocks = 0;
                        },
                    }
                }
            }
        }

        let mut additional_index_start = 0;
        let mut additional_index_end = 0;
        'schedule_commit_loop: {
            while let Some(mut block_vec) = self.mutable_blocks.pop_front() {
                let num_token = min(
                    queued_to_cached_token_window - additional_index_start,
                    N - block_vec[0].total_tokens().len(),
                );
                additional_index_end = additional_index_start + num_token;
                if additional_index_start != additional_index_end {
                    push_tokens::<N, L>(
                        &mut block_vec,
                        &queued_to_cached_tokens[additional_index_start..additional_index_end + L - 1],
                    );
                    let _ = schedule_tokens::<N, L, _>(&mut block_vec, num_token);
                    cache_tokens::<N, L, _>(
                        &mut block_vec,
                        &queued_to_cached_tokens,
                        additional_index_start,
                        additional_index_end,
                    );
                }
                additional_index_start = additional_index_end;

                if block_vec[0].cached_tokens().len() != N {
                    self.mutable_blocks.push_front(block_vec);
                    break 'schedule_commit_loop;
                } else {
                    debug_assert!(self.semi_immutable_blocks.is_empty());

                    let parent_trie_node_key_vec = self.parent_trie_node_key_vec(self.immutable_blocks.len());
                    match self
                        .block_cache
                        .commit_mutable_block(parent_trie_node_key_vec, block_vec)
                    {
                        CommitMultiLaneMutableBlockResult::Immutable { block_vec } => {
                            debug_assert!(self.semi_immutable_blocks.is_empty());
                            self.immutable_blocks.push(block_vec);
                        },
                        CommitMultiLaneMutableBlockResult::ImmutableCollision { block_vec } => {
                            debug_assert!(self.semi_immutable_blocks.is_empty());
                            self.immutable_blocks.push(block_vec);
                            self.num_in_sync_blocks = 0;
                        },
                    }
                }
            }
        }
        self.try_mark_ready();
        self.unload_cached_resources();

        debug_assert_eq!(ready_to_cached_token_window, index_start);
        debug_assert_eq!(ready_to_cached_token_window, index_end);
        debug_assert_eq!(queued_to_cached_token_window, additional_index_start);
        debug_assert!(queued_to_cached_token_window <= additional_index_end);
    }
}

fn annotation_resource_ids<const L: usize>(annotation_vec: &[SmallVec<[BlockAnnotation; 1]>; L]) -> Vec<ResourceID> {
    let mut resource_ids = Vec::new();
    for annotation in annotation_vec.iter().flatten() {
        let BlockAnnotation::Resource(segment) = annotation else {
            continue;
        };
        if !resource_ids.contains(&segment.resource_id()) {
            resource_ids.push(segment.resource_id());
        }
    }
    resource_ids
}

impl<const N: usize, const P: usize, const L: usize, BC> TrieDecoderBlocks<N, P, L, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    fn try_mark_ready(&mut self) {
        let mut num_cachable_tokens = self.num_queued_tokens().saturating_sub(L - 1);

        for block_vec in self.mutable_blocks.iter_mut() {
            if num_cachable_tokens == 0 {
                break;
            }
            if N == block_vec[0].total_tokens().len() {
                continue;
            }

            let num_tokens = min(num_cachable_tokens, N - block_vec[0].total_tokens().len());
            num_cachable_tokens -= num_tokens;
            let tokens = pop_front_queued_tokens::<L>(&mut self.queued_tokens, num_tokens);
            debug_assert_eq!(num_tokens + L - 1, tokens.len());
            push_tokens::<N, L>(block_vec, &tokens);
            // TODO when mutable block is full, maybe turn it into semi immutable
        }
    }

    fn parent_trie_node_key_vec(&self, block_index: usize) -> [Option<TrieNodeKey>; L] {
        if block_index == 0 {
            std::array::from_fn(|_| None)
        } else {
            debug_assert!(
                block_index <= self.immutable_blocks.len(),
                "parent_trie_node_key_vec: block_index={block_index} requires previous immutable block"
            );
            let parent_block_vec = &self.immutable_blocks[block_index - 1];
            std::array::from_fn(|lane| Some(parent_block_vec[lane].trie_node_key()))
        }
    }

    fn block_annotation_vec(&self, block_index: usize) -> [SmallVec<[BlockAnnotation; 1]>; L] {
        std::array::from_fn(|lane| {
            let mut annotations: SmallVec<[BlockAnnotation; 1]> = SmallVec::new();
            if block_index == 0 && lane > 0 {
                // MTP root lane identity, L = 4:
                //
                // verified tokens  t0  t1  t2  ...
                // lane 0 root      t0  t1  t2  ...
                // lane 1 root          t1  t2  ...  prefix: [t0]
                // lane 2 root              t2  ...  prefix: [t0, t1]
                // lane 3 root                  ...  prefix: [t0, t1, t2]
                //
                // TODO: Allow a short MTP root to defer its lane-specific PrefixTokens annotation until verified
                // history contains that lane's complete prefix.
                let prefix_tokens: Arc<[Token]> = self.total_tokens().take(lane).collect::<Vec<_>>().into();
                debug_assert_eq!(lane, prefix_tokens.len());
                annotations.push(BlockAnnotation::prefix_tokens(prefix_tokens));
            }

            let block_token_start = block_index * N + lane;
            let block_token_end = block_token_start + N;
            for placement in &self.resource_placements {
                for &(placement_token_start, resource_index, len) in placement.placements() {
                    let placement_token_end = placement_token_start + len;
                    let active_token_start = block_token_start.max(placement_token_start);
                    let active_token_end = block_token_end.min(placement_token_end);
                    if active_token_start >= active_token_end {
                        continue;
                    }

                    let local_token_index = (active_token_start - block_token_start) as u16;
                    let active_resource_index =
                        u32::try_from(resource_index + active_token_start - placement_token_start)
                            .expect("resource index must fit the cache annotation u32 domain");
                    let active_len = (active_token_end - active_token_start) as u16;
                    annotations.push(BlockAnnotation::resource(ResourceSegment::new(
                        placement.resource_id(),
                        local_token_index,
                        active_resource_index,
                        active_len,
                    )));
                }
            }
            annotations.sort_unstable();
            annotations
        })
    }
}

#[cfg(test)]
#[path = "./api_test_vanilla.rs"]
mod api_test_vanilla;

#[cfg(test)]
#[path = "./api_test_w_mtp.rs"]
mod api_test_w_mtp;

#[cfg(test)]
#[path = "./api_test_w_dspark.rs"]
mod api_test_w_dspark;

#[cfg(test)]
#[path = "./api_test_resource.rs"]
mod api_test_resource;

#[cfg(test)]
#[path = "./api_test_support.rs"]
mod api_test_support;
