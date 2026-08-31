use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DeviceResponse;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::ReplayableModelExecutorRequest;
use inference_runtime_core::compute::ReplayableModelExecutorResponse;
use inference_runtime_core::compute::SampledTokens;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT;
use inference_runtime_core::config::ExecutorHibernationMode;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::resource::processor::ResourceProcessors;
use inference_runtime_proto::inference_runtime_service::CompletionReason as ProtoCompletionReason;
use inference_runtime_proto::inference_runtime_service::DecodeRequest as ProtoDecodeRequest;
use inference_runtime_proto::inference_runtime_service::decode_response::Event;
use inference_runtime_proto::inference_runtime_service::inference_runtime_client::InferenceRuntimeClient;
use inference_runtime_proto::inference_runtime_service::inference_runtime_server::InferenceRuntimeServer;
use ordered_float::NotNan;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::Code;
use tonic::Request;
use tonic::transport::Server;

use super::GRPCServer;
use crate::api::Inference;
use crate::runtime::InferenceRuntime;

type TestRuntime = InferenceRuntime<1024, 1, 4>;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_decode_stream_lifecycle() {
    let runtime = new_runtime();
    let inference = Arc::new(Inference::new(runtime.clone(), Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let listen_addr = listener.local_addr().unwrap();
    let (server_shutdown_tx, server_shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(
        Server::builder()
            .add_service(InferenceRuntimeServer::new(GRPCServer::new(inference)))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                let _ = server_shutdown_rx.await;
            }),
    );
    let mut client = InferenceRuntimeClient::connect(format!("http://{listen_addr}"))
        .await
        .unwrap();
    let empty_stream = tokio_stream::empty::<ProtoDecodeRequest>();
    let status = client.decode_stream(Request::new(empty_stream)).await.unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(status.message().contains("at least one request"));

    let (input_closed_tx, input_closed_rx) = std::sync::mpsc::channel();
    let driver_runtime = runtime.clone();
    let driver = tokio::task::spawn_blocking(move || drive_turns(driver_runtime, input_closed_rx));

    let (input_tx, mut input_rx) = mpsc::channel(3);
    input_tx.send(fixture_request(1)).await.unwrap();
    input_tx.send(fixture_request(2)).await.unwrap();
    input_tx
        .send(ProtoDecodeRequest {
            tokens: Vec::new(),
            ..fixture_request(3)
        })
        .await
        .unwrap();
    drop(input_tx);
    let input = async_stream::stream! {
        while let Some(request) = input_rx.recv().await {
            yield request;
        }
        input_closed_tx.send(()).unwrap();
    };
    let mut output = client.decode_stream(Request::new(input)).await.unwrap().into_inner();

    let first_chunk = output.message().await.unwrap().unwrap();
    let first_chunk_id = first_chunk.request_id;
    let Some(Event::Chunk(first_chunk)) = first_chunk.event else {
        panic!("first turn must start with a chunk")
    };
    assert_eq!(first_chunk.tokens, vec![101]);
    assert_eq!(first_chunk.probs, vec![1.0]);

    let first_completion = output.message().await.unwrap().unwrap();
    let first_completion_id = first_completion.request_id;
    let Some(Event::Completion(first_completion)) = first_completion.event else {
        panic!("first turn must end with one completion")
    };
    assert_eq!(first_completion.reason, ProtoCompletionReason::LengthLimit as i32);
    assert_eq!(first_completion.num_output_tokens, 1);

    let second_chunk = output.message().await.unwrap().unwrap();
    let second_chunk_id = second_chunk.request_id;
    let Some(Event::Chunk(second_chunk)) = second_chunk.event else {
        panic!("second turn must start with a chunk")
    };
    assert_eq!(second_chunk.tokens, vec![202]);
    assert_eq!(second_chunk.probs, vec![1.0]);

    let second_completion = output.message().await.unwrap().unwrap();
    let second_completion_id = second_completion.request_id;
    let Some(Event::Completion(second_completion)) = second_completion.event else {
        panic!("second turn must end with one completion")
    };
    assert_eq!(second_completion.reason, ProtoCompletionReason::LengthLimit as i32);
    assert_eq!(second_completion.num_output_tokens, 1);

    assert_ne!(first_chunk_id, 0);
    assert_eq!(first_chunk_id, first_completion_id);
    assert_eq!(first_chunk_id, second_chunk_id);
    assert_eq!(second_chunk_id, second_completion_id);

    let status = output.message().await.unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(status.message().contains("at least one token"));
    assert!(output.message().await.unwrap().is_none());

    driver.await.unwrap();
    drop(output);
    drop(client);
    server_shutdown_tx.send(()).unwrap();
    server.await.unwrap().unwrap();
    runtime.shutdown();
}

fn new_runtime() -> Arc<TestRuntime> {
    let shutdown = Shutdown::new();
    let async_task_handle = tokio::runtime::Handle::current();
    Arc::new(TestRuntime::new(
        RuntimeConfig {
            max_queued_requests: 1,
            max_running_requests: 1,
            executor_hibernation_timeout: DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT,
            executor_hibernation_mode: ExecutorHibernationMode::Selected,
            context_window: 4096,
            num_tokens_per_cache_block: 1024,
            num_kv_heads: 1,
            kv_head_dim: 1,
            kv_dtype_bytes: 1,
            num_pages: 64,
            page_bytes: 32,
            cache_lanes: vec![CacheLaneRuntimeConfig {
                num_pages_per_kv_block: 64,
                num_pages_per_state_block: 0,
                block_cache_capacity: 1,
            }],
        },
        SchedulerConfig {
            max_requests: 1,
            max_tokens: 1024,
            max_tokens_per_request: 1024,
            max_compute_slots: 1,
        },
        0,
        shutdown,
        &async_task_handle,
        Arc::new(ResourceProcessors::new()),
    ))
}

fn drive_turns(runtime: Arc<TestRuntime>, input_closed_rx: Receiver<()>) {
    let request_rx = runtime.model_executor_request_rx();
    let response_tx = runtime.model_executor_response_tx();
    let mut session_identity = None;
    for (turn_index, sampled_token) in [101, 202].into_iter().enumerate() {
        let ReplayableModelExecutorRequest::Batch(mut batch) = request_rx.recv_timeout(Duration::from_secs(5)).unwrap()
        else {
            panic!("decode turn must submit one executor batch")
        };
        assert_eq!(batch.dev_reqs.len(), 1);
        let request = batch.dev_reqs.pop().unwrap();
        match session_identity {
            Some(identity) => assert_eq!((request.req_id, request.req_slot), identity),
            None => session_identity = Some((request.req_id, request.req_slot)),
        }
        let query_tokens = request.decoder_query_tokens;
        let QueryTokens::Decode {
            token_index,
            tokens,
            spec_tokens,
            ..
        } = &query_tokens
        else {
            panic!("decode turn must submit Decode query tokens")
        };
        assert!(spec_tokens.is_empty());
        if turn_index == 0 {
            assert_eq!((*token_index, tokens.as_slice()), (0, &[Token::new(1)][..]));
        } else {
            assert_eq!(
                (*token_index, tokens.as_slice()),
                (1, &[Token::new(101), Token::new(2)][..])
            );
        }
        if turn_index == 0 {
            input_closed_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        let probability = NotNan::new(1.0).unwrap();
        let response = DeviceResponse {
            req_id: request.req_id,
            sampled_tokens: SampledTokens::Decode {
                epoch: query_tokens.epoch(),
                validated_tokens: Vec::new(),
                validated_probs: Vec::new(),
                sampled_token: Token::new(sampled_token),
                sampled_prob: probability,
                spec_tokens: Vec::new(),
                spec_probs: Vec::new(),
                spec_confidences: Vec::new(),
            },
            query_tokens,
        };
        response_tx
            .send(ReplayableModelExecutorResponse::Batch(BatchDeviceResponse::new(
                batch.seq,
                [response],
            )))
            .unwrap();
    }
}

fn fixture_request(token: u32) -> ProtoDecodeRequest {
    ProtoDecodeRequest {
        tokens: vec![token],
        max_sampled_tokens: 1,
        stop_sequences: Vec::new(),
        temperature: None,
        top_k: None,
        top_p: None,
        seed: Some(token),
    }
}
