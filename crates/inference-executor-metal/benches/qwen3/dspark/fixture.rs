use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use inference_executor_core::model::ReplayableModel;
use inference_executor_core::model::qwen::v3_x::dspark::init_qwen3x_dspark_config;
use inference_executor_metal::model::qwen::v3::executor::Qwen3Executor;
use inference_executor_metal::model::qwen::v3::executor::Qwen3ExecutorConfig;
use inference_executor_metal::model::qwen::v3::executor::init_qwen_3_model;
use inference_executor_metal::model::qwen::v3::executor::init_qwen_3_model_with_dspark;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::SampledTokens;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::Token;

use crate::Case;

pub const NUM_TOKENS_PER_BLOCK: usize = 1024;

pub struct Fixture {
    model: Qwen3Executor,
    case: Case,
    num_spec_tokens: usize,
    num_cache_pages: usize,
    num_kv_page_ids_per_block: usize,
    next_sequence: u64,
    requests: Vec<RequestState>,
}

struct RequestState {
    req_id: usize,
    req_slot: u32,
    token_index: usize,
    token: Token,
    spec_tokens: Vec<Token>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutionTiming {
    pub wall: Duration,
    pub prepare: Duration,
    pub main_record: Duration,
    pub main_submit: Duration,
    pub main_read: Duration,
    pub spec_record: Duration,
    pub spec_submit: Duration,
    pub spec_read: Duration,
    pub commit: Duration,
}

impl ExecutionTiming {
    pub fn add_assign(&mut self, other: Self) {
        self.wall += other.wall;
        self.prepare += other.prepare;
        self.main_record += other.main_record;
        self.main_submit += other.main_submit;
        self.main_read += other.main_read;
        self.spec_record += other.spec_record;
        self.spec_submit += other.spec_submit;
        self.spec_read += other.spec_read;
        self.commit += other.commit;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Trajectory {
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub generated_proposals: usize,
    pub sampled_tokens: usize,
}

impl Trajectory {
    pub fn add_assign(&mut self, other: Self) {
        self.proposed_tokens += other.proposed_tokens;
        self.accepted_tokens += other.accepted_tokens;
        self.generated_proposals += other.generated_proposals;
        self.sampled_tokens += other.sampled_tokens;
    }
}

impl Fixture {
    pub fn new(
        case: Case,
        model_dir: &Path,
        dspark_model_dir: Option<&Path>,
        num_requests: usize,
        start_context: usize,
        num_cache_pages: usize,
    ) -> Self {
        let num_spec_tokens = match case {
            Case::Main => 0,
            Case::DSpark => {
                let config = init_qwen3x_dspark_config(
                    dspark_model_dir.expect("Qwen3 DSpark benchmark case requires a DSpark model directory"),
                )
                .expect("unable to load Qwen3 DSpark benchmark config");
                config.num_spec_tokens().get()
            },
        };
        let max_tokens = num_requests
            .checked_mul(
                num_spec_tokens
                    .checked_add(1)
                    .expect("Qwen3 DSpark benchmark tokens per request must fit usize"),
            )
            .expect("Qwen3 DSpark benchmark token capacity must fit usize");
        let config = Qwen3ExecutorConfig {
            max_requests: num_requests,
            max_tokens,
            max_tokens_per_request: NUM_TOKENS_PER_BLOCK,
            num_cache_pages,
            num_tokens_per_block: NUM_TOKENS_PER_BLOCK,
        };
        let model = match case {
            Case::Main => init_qwen_3_model(model_dir, config),
            Case::DSpark => {
                init_qwen_3_model_with_dspark(
                    model_dir,
                    dspark_model_dir.expect("Qwen3 DSpark benchmark requires a DSpark model directory"),
                    config,
                )
            },
        }
        .unwrap_or_else(|error| panic!("unable to initialize Qwen3 {} benchmark: {error}", case.key()));
        assert_eq!(model.num_spec_tokens(), num_spec_tokens);
        let num_kv_page_ids_per_block = model.num_kv_page_ids_per_block();
        let required_cache_pages = num_requests
            .checked_mul(num_kv_page_ids_per_block)
            .expect("Qwen3 DSpark benchmark cache-page requirement must fit usize");
        assert!(
            required_cache_pages <= num_cache_pages,
            "Qwen3 DSpark benchmark requires {required_cache_pages} cache pages for {num_requests} requests, \
             configured {num_cache_pages}"
        );
        let requests = (0..num_requests)
            .map(|req_index| {
                RequestState {
                    req_id: req_index,
                    req_slot: req_index
                        .try_into()
                        .expect("Qwen3 DSpark benchmark request slot must fit u32"),
                    token_index: start_context,
                    token: Token::new(11),
                    spec_tokens: Vec::new(),
                }
            })
            .collect();
        Self {
            model,
            case,
            num_spec_tokens,
            num_cache_pages,
            num_kv_page_ids_per_block,
            next_sequence: 0,
            requests,
        }
    }

    pub fn num_spec_tokens(&self) -> usize {
        self.num_spec_tokens
    }

    pub fn run(&mut self) -> (ExecutionTiming, Trajectory) {
        let batch_request = self.batch_request();
        let proposed_tokens = batch_request
            .dev_reqs
            .iter()
            .map(|request| request.decoder_query_tokens.num_spec_tokens())
            .sum();
        let (timing, response) = self.execute(batch_request);
        let trajectory = self.advance(response, proposed_tokens);
        (timing, trajectory)
    }

    fn execute(&mut self, core_batch_request: BatchDeviceRequest) -> (ExecutionTiming, BatchDeviceResponse) {
        let wall_start = Instant::now();
        let prepare_start = Instant::now();
        let model_batch_request = self.model.prepare_batch(&core_batch_request);
        let prepare = prepare_start.elapsed();

        let main_record_start = Instant::now();
        let mut recorder = self.model.begin_ops_recording(&model_batch_request);
        let main_hidden = self.model.embed_main(&mut recorder, &model_batch_request);
        let main_hidden = self
            .model
            .forward_main(&mut recorder, &model_batch_request, main_hidden);
        let main_output = self
            .model
            .unembed_main(&mut recorder, &model_batch_request, &main_hidden);
        self.model
            .sample_main(&mut recorder, &model_batch_request, &main_output);
        let main_record = main_record_start.elapsed();

        let main_submit_start = Instant::now();
        let main_submission = self.model.submit_main(&recorder);
        main_submission.wait();
        let main_submit = main_submit_start.elapsed();
        let main_read_start = Instant::now();
        let gpu_timestamp_durations = main_submission.gpu_timestamp_durations();
        drop(main_submission);
        let mut sampled_output = self.model.read_main(
            &recorder,
            &model_batch_request,
            main_submit,
            gpu_timestamp_durations.as_deref(),
        );
        let main_read = main_read_start.elapsed();

        let mut spec_record = Duration::ZERO;
        let mut spec_submit = Duration::ZERO;
        let mut spec_read = Duration::ZERO;
        let run_spec_prefill = self.model.run_spec_prefill(&model_batch_request);
        let run_spec_decode = self.model.run_spec_decode(&model_batch_request, &sampled_output);
        if run_spec_prefill || run_spec_decode {
            let spec_record_start = Instant::now();
            if run_spec_prefill {
                self.model
                    .prefill_spec(&mut recorder, &model_batch_request, &sampled_output);
            }
            if run_spec_decode {
                self.model
                    .decode_spec(&mut recorder, &model_batch_request, &sampled_output);
            }
            spec_record = spec_record_start.elapsed();

            let spec_submit_start = Instant::now();
            self.model.submit_spec(&recorder).wait();
            spec_submit = spec_submit_start.elapsed();
            if run_spec_decode {
                let spec_read_start = Instant::now();
                sampled_output = self
                    .model
                    .read_spec(&recorder, &model_batch_request, sampled_output, spec_submit);
                spec_read = spec_read_start.elapsed();
            }
        }
        drop(recorder);

        let commit_start = Instant::now();
        let response = self.model.commit_batch(core_batch_request, sampled_output);
        let commit = commit_start.elapsed();
        (
            ExecutionTiming {
                wall: wall_start.elapsed(),
                prepare,
                main_record,
                main_submit,
                main_read,
                spec_record,
                spec_submit,
                spec_read,
                commit,
            },
            response,
        )
    }

    fn batch_request(&self) -> BatchDeviceRequest {
        let requests = self
            .requests
            .iter()
            .map(|request| {
                DeviceRequest::new(
                    request.req_id,
                    request.req_slot,
                    QueryTokens::Decode {
                        epoch: 0,
                        token_index: request.token_index,
                        tokens: vec![request.token],
                        spec_tokens: request.spec_tokens.clone(),
                    },
                    DecoderSyncBlocks::new(0, vec![vec![self.kv_page_ids(request.req_slot)]], Vec::new()),
                    vec![],
                    SamplingConfig {
                        max_sampled_tokens: usize::MAX,
                        temperature: 0.0,
                        top_k: 1,
                        top_p: 1.0,
                        seed: Some(request.req_slot),
                        stop_sequences: Vec::new(),
                    },
                )
            })
            .collect::<Vec<_>>();
        BatchDeviceRequest::new(self.next_sequence, requests)
    }

    fn kv_page_ids(&self, req_slot: u32) -> Vec<u32> {
        let first = usize::try_from(req_slot)
            .expect("Qwen3 DSpark benchmark request slot must fit usize")
            .checked_mul(self.num_kv_page_ids_per_block)
            .expect("Qwen3 DSpark benchmark page-ID offset must fit usize");
        let end = first
            .checked_add(self.num_kv_page_ids_per_block)
            .expect("Qwen3 DSpark benchmark page-ID end must fit usize");
        assert!(end <= self.num_cache_pages);
        (first..end)
            .map(|page_id| page_id.try_into().expect("Qwen3 DSpark benchmark page ID must fit u32"))
            .collect()
    }

    fn advance(&mut self, response: BatchDeviceResponse, proposed_tokens: usize) -> Trajectory {
        assert_eq!(
            response.dev_resps.len(),
            self.requests.len(),
            "Qwen3 DSpark benchmark requires one response per request"
        );
        let mut trajectory = Trajectory {
            proposed_tokens,
            ..Trajectory::default()
        };
        for (request, response) in self.requests.iter_mut().zip(response.dev_resps) {
            assert_eq!(
                response.req_id, request.req_id,
                "Qwen3 DSpark benchmark response order must match requests"
            );
            let SampledTokens::Decode {
                validated_tokens,
                sampled_token,
                spec_tokens,
                ..
            } = response.sampled_tokens
            else {
                panic!("Qwen3 DSpark benchmark requires decode output");
            };
            trajectory.accepted_tokens += validated_tokens.len();
            trajectory.generated_proposals += spec_tokens.len();
            trajectory.sampled_tokens += 1;
            request.token_index = request
                .token_index
                .checked_add(validated_tokens.len())
                .and_then(|position| position.checked_add(1))
                .expect("Qwen3 DSpark benchmark token index must fit usize");
            assert!(
                request.token_index <= NUM_TOKENS_PER_BLOCK,
                "Qwen3 DSpark benchmark currently supports one cache block"
            );
            request.token = sampled_token;
            request.spec_tokens = match self.case {
                Case::Main => {
                    assert!(
                        spec_tokens.is_empty(),
                        "Qwen3 Main benchmark must not produce Spec tokens"
                    );
                    Vec::new()
                },
                Case::DSpark => {
                    assert_eq!(
                        spec_tokens.len(),
                        self.num_spec_tokens,
                        "Qwen3 DSpark benchmark must produce one fixed proposal block"
                    );
                    spec_tokens
                },
            };
        }
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .expect("Qwen3 DSpark benchmark sequence must fit u64");
        trajectory
    }
}
