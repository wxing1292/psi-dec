use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Select;
use crossbeam_channel::Sender;
use inference_runtime_core::channel::DedupNotifier;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::DeviceResponse;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::ReplayableModelBatchExecutor;
use inference_runtime_core::compute::SampledTokens;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::Token;
use tonic::Status;

use crate::perf_metrics::ExecutorBatchPerfMetrics;
use crate::perf_metrics::emit_executor_batch_perf_metrics;
use crate::perf_metrics::summarize_batch_device_request;
use crate::perf_metrics::summarize_batch_device_response;
use crate::profiling;

pub struct ReplayableModelExecutorLoop<M> {
    batch_dev_req_rx: Receiver<BatchDeviceRequest>,
    batch_dev_resp_tx: Sender<BatchDeviceResponse>,
    req_slot_reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
    req_slot_reset_rx: Receiver<()>,
    shutdown: Shutdown,
    model: M,
    debug_logging: bool,
}

impl<M> ReplayableModelExecutorLoop<M>
where
    M: ReplayableModelBatchExecutor,
{
    pub fn new(
        batch_dev_req_rx: Receiver<BatchDeviceRequest>,
        batch_dev_resp_tx: Sender<BatchDeviceResponse>,
        req_slot_reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
        req_slot_reset_rx: Receiver<()>,
        shutdown: Shutdown,
        model: M,
    ) -> Self {
        Self {
            batch_dev_req_rx,
            batch_dev_resp_tx,
            req_slot_reset_notifier,
            req_slot_reset_rx,
            shutdown,
            model,
            debug_logging: false,
        }
    }

    pub fn with_debug_logging(mut self, enabled: bool) -> Self {
        self.debug_logging = enabled;
        self
    }

    pub fn event_loop(mut self) -> Result<(), Status> {
        let span = tracing::info_span!("replayable executor loop");
        let _enter = span.enter();
        tracing::info!("started");

        let shutdown_rx = self.shutdown.sync_rx().clone();
        'event_loop: while !self.shutdown.is_shutdown() {
            let mut select = Select::new();
            let op_shutdown = select.recv(&shutdown_rx);
            let op_recv_req_slot_reset = select.recv(&self.req_slot_reset_rx);
            let op_recv_batch_dev_req = select.recv(&self.batch_dev_req_rx);

            let op = select.select();
            let op_index = op.index();
            match op_index {
                _ if op_index == op_shutdown => {
                    let _ = op.recv(&shutdown_rx);
                    tracing::info!("received shutdown signal, stopping");
                    break 'event_loop;
                },
                _ if op_index == op_recv_req_slot_reset => {
                    match op.recv(&self.req_slot_reset_rx) {
                        Ok(()) => self.reset_req_slots(),
                        Err(_) => {
                            tracing::info!("request slot reset channel closed, stopping");
                            break 'event_loop;
                        },
                    }
                },
                _ if op_index == op_recv_batch_dev_req => {
                    match op.recv(&self.batch_dev_req_rx) {
                        Ok(batch_dev_req) => {
                            self.reset_req_slots();
                            let batch_dev_resp = self.execute(batch_dev_req)?;
                            if let Err(err) = self.batch_dev_resp_tx.send(batch_dev_resp) {
                                tracing::info!("unable to send batch device response, err: {err}, stopping");
                                break 'event_loop;
                            }
                        },
                        Err(_) => {
                            tracing::info!("batch device request channel closed, stopping");
                            break 'event_loop;
                        },
                    }
                },
                _ => unreachable!(),
            }
        }

        self.shutdown.shutdown();
        tracing::info!("stopped");
        Ok(())
    }

    fn reset_req_slots(&mut self) {
        self.req_slot_reset_rx.try_iter().for_each(drop);
        let req_slots = self.req_slot_reset_notifier.recv_many();
        if !req_slots.is_empty() {
            let req_slots = req_slots.into_iter().collect::<Vec<_>>();
            self.model.reset_req_slots(&req_slots);
        }
    }

    fn execute(&mut self, batch_req: BatchDeviceRequest) -> Result<BatchDeviceResponse, Status> {
        let total_start = Instant::now();
        let batch_seq = batch_req.seq;
        let batch_summary = summarize_batch_device_request(&batch_req);
        tracing::debug!(
            target: "inference-runtime-service::executor",
            phase = "executor.batch.request",
            model = self.model.model_name(),
            batch_seq,
            num_reqs = batch_req.dev_reqs.len(),
            requests = %batch_req
                .dev_reqs
                .iter()
                .map(summarize_device_request)
                .collect::<Vec<_>>()
                .join(" | "),
            "executor batch request"
        );

        let _executor_batch_span = profiling::span("executor.batch");

        if batch_req.dev_reqs.is_empty() {
            return Ok(BatchDeviceResponse::new(batch_seq, Vec::new()));
        }

        let prepare_batch_start = Instant::now();
        let model_batch_req = {
            let _span = profiling::span("prepare_batch");
            self.model.prepare_batch(&batch_req)
        };
        let prepare_batch_elapsed = prepare_batch_start.elapsed();

        let mut recorder = self.model.begin_ops_recording(&model_batch_req);

        let input_start = Instant::now();
        let model_batch_hidden_req = {
            let _span = profiling::span("model.input");
            if self.model.first_pp_stage(&model_batch_req) {
                self.model.embed(&mut recorder, &model_batch_req)
            } else {
                todo!("pipeline stages after the first must read hidden states from the batch request")
            }
        };
        let input_elapsed = input_start.elapsed();

        let model_start = Instant::now();
        let model_batch_hidden_resp = {
            let _span = profiling::span("model.forward");
            self.model
                .forward_main(&mut recorder, &model_batch_req, model_batch_hidden_req)
        };
        let model_elapsed = model_start.elapsed();

        let last_pp_stage = self.model.last_pp_stage(&model_batch_req);
        let do_sample = last_pp_stage
            && batch_req
                .dev_reqs
                .iter()
                .any(|dev_req| matches!(dev_req.decoder_query_tokens, QueryTokens::Decode { .. }));
        let do_rejection_sample = batch_req.dev_reqs.iter().any(|dev_req| {
            matches!(
                &dev_req.decoder_query_tokens,
                QueryTokens::Decode { spec_tokens, .. } if !spec_tokens.is_empty()
            )
        });

        let output_start = Instant::now();
        let sampled_output = {
            let _span = profiling::span("model.output");
            let sampled_output = if do_sample {
                let model_batch_resp = self
                    .model
                    .unembed(&mut recorder, &model_batch_req, &model_batch_hidden_resp);
                if do_rejection_sample {
                    self.model
                        .rejection_sample(&mut recorder, &model_batch_req, &model_batch_resp)
                } else {
                    self.model.sample(&mut recorder, &model_batch_req, &model_batch_resp)
                }
            } else {
                self.model.empty_sampled_output()
            };
            if last_pp_stage {
                self.model.forward_mtp(
                    &mut recorder,
                    &model_batch_req,
                    &model_batch_hidden_resp,
                    sampled_output,
                )
            } else {
                sampled_output
            }
        };
        let sampled_output = self.model.finish_ops_recording(recorder, sampled_output);
        let model_output_timing = self.model.sampled_output_timing(&sampled_output);
        let sampled_rows_count = self.model.sampled_output_len(&sampled_output);
        let output_elapsed = output_start.elapsed();

        let commit_batch_start = Instant::now();
        let batch_resp = {
            let _span = profiling::span("commit_batch");
            self.model.commit_batch(batch_req, sampled_output)
        };
        let commit_batch_elapsed = commit_batch_start.elapsed();
        let response_summary = summarize_batch_device_response(&batch_resp);
        drop(_executor_batch_span);

        profiling::maybe_emit_tree_profile_summary("executor.batch", batch_seq);
        emit_executor_batch_perf_metrics(
            self.debug_logging,
            self.model.model_name(),
            batch_seq,
            batch_summary,
            response_summary,
            ExecutorBatchPerfMetrics {
                sampled_rows: sampled_rows_count,
                do_sample,
                model_output_timing,
                total_elapsed: total_start.elapsed(),
                prepare_batch_elapsed,
                input_elapsed,
                model_elapsed,
                output_elapsed,
                commit_batch_elapsed,
            },
        );
        tracing::debug!(
            target: "inference-runtime-service::executor",
            phase = "executor.batch.response",
            model = self.model.model_name(),
            batch_seq,
            num_responses = batch_resp.dev_resps.len(),
            responses = %batch_resp
                .dev_resps
                .iter()
                .map(summarize_device_response)
                .collect::<Vec<_>>()
                .join(" | "),
            "executor batch response"
        );

        Ok(batch_resp)
    }
}

fn summarize_device_request(dev_req: &DeviceRequest) -> String {
    let lane_kv_page_ids = dev_req.decoder_sync_blocks.kv_page_ids();
    let lane0 = lane_kv_page_ids.first().map(Vec::as_slice).unwrap_or(&[]);
    let pages_per_kv_block = lane0.first().map(Vec::len).unwrap_or(0);
    let query_tokens = summarize_query_tokens(&dev_req.decoder_query_tokens);

    format!(
        "req_id={} req_slot={} kind={} epoch={} token_index={} {} kv_block_index={} kv_blocks={} pages_per_kv_block={}",
        dev_req.req_id,
        dev_req.req_slot,
        query_kind(&dev_req.decoder_query_tokens),
        dev_req.decoder_query_tokens.epoch(),
        dev_req.decoder_query_tokens.token_index(),
        query_tokens,
        dev_req.decoder_sync_blocks.block_index(),
        lane0.len(),
        pages_per_kv_block
    )
}

fn summarize_device_response(dev_resp: &DeviceResponse) -> String {
    match &dev_resp.sampled_tokens {
        SampledTokens::Prefill { epoch } => format!("req_id={} prefill epoch={epoch}", dev_resp.req_id),
        SampledTokens::Decode {
            epoch,
            validated_tokens,
            sampled_token,
            sampled_prob,
            spec_tokens,
            spec_probs,
            ..
        } => {
            format!(
                "req_id={} decode epoch={} validated={} sampled={} prob={:.6} spec_out={} spec_probs={}",
                dev_resp.req_id,
                epoch,
                summarize_tokens(validated_tokens),
                sampled_token.value(),
                sampled_prob.into_inner(),
                summarize_tokens(spec_tokens),
                summarize_f32_slice(&spec_probs.iter().map(|prob| prob.into_inner()).collect::<Vec<_>>())
            )
        },
    }
}

fn summarize_query_tokens(query_tokens: &QueryTokens) -> String {
    match query_tokens {
        QueryTokens::Prefill { tokens, window, .. } => {
            format!("tokens={} window={window}", summarize_tokens(tokens))
        },
        QueryTokens::Decode {
            tokens, spec_tokens, ..
        } => {
            format!(
                "tokens={} spec={}",
                summarize_tokens(tokens),
                summarize_tokens(spec_tokens)
            )
        },
    }
}

fn summarize_tokens(tokens: &[Token]) -> String {
    summarize_u32_slice(&tokens.iter().map(|token| token.value()).collect::<Vec<_>>())
}

fn query_kind(query_tokens: &QueryTokens) -> &'static str {
    match query_tokens {
        QueryTokens::Prefill { .. } => "prefill",
        QueryTokens::Decode { spec_tokens, .. } if spec_tokens.is_empty() => "decode",
        QueryTokens::Decode { .. } => "spec-decode",
    }
}

fn summarize_u32_slice(values: &[u32]) -> String {
    const MAX_VALUES: usize = 8;
    if values.len() <= MAX_VALUES {
        return format!("{values:?}");
    }
    format!("{:?}..(+{})", &values[..MAX_VALUES], values.len() - MAX_VALUES)
}

fn summarize_f32_slice(values: &[f32]) -> String {
    const MAX_VALUES: usize = 8;
    if values.len() <= MAX_VALUES {
        return format!(
            "[{}]",
            values
                .iter()
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    format!(
        "[{}]..(+{})",
        values[..MAX_VALUES]
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(", "),
        values.len() - MAX_VALUES
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crossbeam_channel::Sender;
    use crossbeam_channel::bounded;

    use super::*;

    struct ResetOnlyModel {
        reset_tx: Sender<Vec<RawRequestSlot>>,
    }

    impl ReplayableModelBatchExecutor for ResetOnlyModel {
        type ModelBatchReq = ();
        type ModelBatchHidden = ();
        type ModelBatchResp = ();
        type SampledOutput = ();
        type ModelOpsRecorder = ();

        fn model_name(&self) -> &str {
            "reset-only"
        }

        fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
            self.reset_tx.send(request_slots.to_vec()).unwrap();
        }

        fn prepare_batch(&mut self, _core_batch_req: &BatchDeviceRequest) -> Self::ModelBatchReq {
            panic!("reset-only model must not execute a batch")
        }

        fn commit_batch(
            &mut self,
            _core_batch_req: BatchDeviceRequest,
            _sampled_output: Self::SampledOutput,
        ) -> BatchDeviceResponse {
            panic!("reset-only model must not commit a batch")
        }

        fn begin_ops_recording(&mut self, _batch_req: &Self::ModelBatchReq) -> Self::ModelOpsRecorder {}

        fn embed(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _batch_req: &Self::ModelBatchReq,
        ) -> Self::ModelBatchHidden {
        }

        fn unembed(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchReq,
            _model_batch_hidden: &Self::ModelBatchHidden,
        ) -> Self::ModelBatchResp {
        }

        fn forward_main(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchReq,
            _model_batch_hidden: Self::ModelBatchHidden,
        ) -> Self::ModelBatchHidden {
        }

        fn sample(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchReq,
            _model_batch_resp: &Self::ModelBatchResp,
        ) -> Self::SampledOutput {
        }

        fn empty_sampled_output(&self) -> Self::SampledOutput {}

        fn sampled_output_len(&self, _sampled_output: &Self::SampledOutput) -> usize {
            0
        }
    }

    #[test]
    fn executor_drains_request_slot_resets_without_a_device_batch() {
        let (_batch_dev_req_tx, batch_dev_req_rx) = bounded(1);
        let (batch_dev_resp_tx, _batch_dev_resp_rx) = bounded(1);
        let (req_slot_reset_notifier, req_slot_reset_rx) = DedupNotifier::new();
        let (seen_reset_tx, seen_reset_rx) = bounded(1);
        let shutdown = Shutdown::new();
        let executor = ReplayableModelExecutorLoop::new(
            batch_dev_req_rx,
            batch_dev_resp_tx,
            req_slot_reset_notifier.clone(),
            req_slot_reset_rx,
            shutdown.clone(),
            ResetOnlyModel {
                reset_tx: seen_reset_tx,
            },
        );
        let executor_thread = std::thread::spawn(move || executor.event_loop());

        req_slot_reset_notifier.send_one(3);
        assert_eq!(seen_reset_rx.recv_timeout(Duration::from_secs(1)).unwrap(), vec![3]);
        req_slot_reset_notifier.send_one(7);
        assert_eq!(seen_reset_rx.recv_timeout(Duration::from_secs(1)).unwrap(), vec![7]);

        shutdown.shutdown();
        executor_thread.join().unwrap().unwrap();
    }
}
