use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use inference_executor_metal::model::qwen::v3_5::executor::Qwen35ExecutorConfig;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_dspark;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_mtp;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::ReplayableModel;
use inference_runtime_core::runtime::Token;

const MODEL_27B_DIR_ENV: &str = "PSI_DEC_MODEL_RESIDENCY_TEST_27B_MODEL_DIR";
const MTP_27B_DIR_ENV: &str = "PSI_DEC_MODEL_RESIDENCY_TEST_27B_MTP_MODEL_DIR";
const DSPARK_27B_DIR_ENV: &str = "PSI_DEC_MODEL_RESIDENCY_TEST_27B_DSPARK_MODEL_DIR";
const MODEL_35B_DIR_ENV: &str = "PSI_DEC_MODEL_RESIDENCY_TEST_35B_MODEL_DIR";
const MTP_35B_DIR_ENV: &str = "PSI_DEC_MODEL_RESIDENCY_TEST_35B_MTP_MODEL_DIR";
const DSPARK_35B_DIR_ENV: &str = "PSI_DEC_MODEL_RESIDENCY_TEST_35B_DSPARK_MODEL_DIR";
const NUM_CACHE_PAGES: usize = 1024;

#[test]
#[ignore = "requires Qwen3.6 27B Main and MTP checkpoints and substantial unified memory"]
fn model_residency_27b_mtp_round_trip() {
    let model = init_qwen_3_5_model_with_mtp(
        model_dir(MODEL_27B_DIR_ENV),
        model_dir(MTP_27B_DIR_ENV),
        num_spec_tokens(),
        executor_config(),
    )
    .expect("Qwen3.6 27B with MTP must initialize");
    run_residency_round_trip(model);
}

#[test]
#[ignore = "requires Qwen3.6 27B Main and DSpark checkpoints and substantial unified memory"]
fn model_residency_27b_dspark_round_trip() {
    let model = init_qwen_3_5_model_with_dspark(
        model_dir(MODEL_27B_DIR_ENV),
        model_dir(DSPARK_27B_DIR_ENV),
        Some(num_spec_tokens()),
        executor_config(),
    )
    .expect("Qwen3.6 27B with DSpark must initialize");
    run_residency_round_trip(model);
}

#[test]
#[ignore = "requires Qwen3.6 35B Main and MTP checkpoints and substantial unified memory"]
fn model_residency_35b_mtp_round_trip() {
    let model = init_qwen_3_5_model_with_mtp(
        model_dir(MODEL_35B_DIR_ENV),
        model_dir(MTP_35B_DIR_ENV),
        num_spec_tokens(),
        executor_config(),
    )
    .expect("Qwen3.6 35B with MTP must initialize");
    run_residency_round_trip(model);
}

#[test]
#[ignore = "requires Qwen3.6 35B Main and DSpark checkpoints and substantial unified memory"]
fn model_residency_35b_dspark_round_trip() {
    let model = init_qwen_3_5_model_with_dspark(
        model_dir(MODEL_35B_DIR_ENV),
        model_dir(DSPARK_35B_DIR_ENV),
        Some(num_spec_tokens()),
        executor_config(),
    )
    .expect("Qwen3.6 35B with DSpark must initialize");
    run_residency_round_trip(model);
}

fn model_dir(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{variable} must name a Qwen3.6 model directory"))
}

fn num_spec_tokens() -> NonZeroUsize {
    NonZeroUsize::new(1).unwrap()
}

fn executor_config() -> Qwen35ExecutorConfig {
    Qwen35ExecutorConfig {
        max_requests: 1,
        max_tokens: 1,
        max_tokens_per_request: 128,
        num_cache_pages: NUM_CACHE_PAGES,
        num_tokens_per_block: 128,
    }
}

fn run_residency_round_trip(mut model: inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor) {
    run_one_decode(&mut model);
    let temp_dir = std::env::temp_dir().join(format!(
        "psi-dec-model-residency-round-trip-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must not precede the Unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir).expect("test snapshot directory must be created");
    let lifecycle_snapshot = temp_dir.join("lifecycle.state");
    let digest_snapshot = temp_dir.join("digest.state");

    let before = model
        .residency_digest(&digest_snapshot)
        .expect("initial model residency must be hashed");
    model.clear_replay_cache();
    model
        .unload_state(&lifecycle_snapshot)
        .expect("model state must unload");
    model.unload_weights();
    model.load_weights().expect("model weights must reload");
    model.load_state(&lifecycle_snapshot).expect("model state must reload");
    let after = model
        .residency_digest(&digest_snapshot)
        .expect("restored model residency must be hashed");

    assert_eq!(
        before, after,
        "model weights and state changed across the residency round trip"
    );
    std::fs::remove_dir_all(&temp_dir).expect("test snapshot directory must be removed");
}

fn run_one_decode(model: &mut inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor) {
    let page_ids_per_lane = std::iter::once(model.num_main_lane_gqa_page_ids_per_block())
        .chain(model.num_mtp_gqa_page_ids_per_block())
        .collect::<Vec<_>>();
    let num_page_ids = page_ids_per_lane.iter().sum::<usize>();
    assert!(
        num_page_ids <= NUM_CACHE_PAGES,
        "test cache must fit one complete Qwen3.6 runtime block"
    );
    let mut next_page_id = 0_u32;
    let kv_page_ids = page_ids_per_lane
        .into_iter()
        .map(|num_lane_page_ids| {
            let begin = next_page_id;
            next_page_id = next_page_id
                .checked_add(u32::try_from(num_lane_page_ids).unwrap())
                .expect("test page ID must fit u32");
            vec![(begin..next_page_id).collect()]
        })
        .collect();
    let core_batch = BatchDeviceRequest::new(
        0,
        [DeviceRequest::new(
            0,
            0,
            QueryTokens::Decode {
                epoch: 0,
                token_index: 0,
                tokens: vec![Token::new(11)],
                spec_tokens: Vec::new(),
            },
            DecoderSyncBlocks::new(0, kv_page_ids, Vec::new()),
            Default::default(),
        )],
    );
    let model_batch = model.prepare_batch(&core_batch);
    let mut recorder = model.begin_ops_recording(&model_batch);
    let hidden = model.embed_main(&mut recorder, &model_batch);
    let hidden = model.forward_main(&mut recorder, &model_batch, hidden);
    let output = model.unembed_main(&mut recorder, &model_batch, &hidden);
    model.sample_main(&mut recorder, &model_batch, &output);
    model.submit_main(&recorder).wait();
    let mut sampled = model.read_main(&recorder, &model_batch, Duration::ZERO);
    if model.run_spec(&model_batch, &sampled) {
        let hidden = model.embed_spec(&mut recorder, &model_batch, &hidden, &sampled);
        let hidden = model.forward_spec(&mut recorder, &model_batch, hidden);
        let output = model.unembed_spec(&mut recorder, &model_batch, &hidden);
        model.sample_spec(&mut recorder, &model_batch, &output);
        model.submit_spec(&recorder).wait();
        sampled = model.read_spec(&recorder, &model_batch, sampled, Duration::ZERO);
    }
    drop(recorder);
    model.commit_batch(core_batch, sampled);
}
