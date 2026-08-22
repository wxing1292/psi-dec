use half::bf16;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::Stream;

#[test]
fn test_scores_and_probabilistic_walk_match_reference_and_preserve_inactive_rows() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        rank: 4,
        top_k: 3,
        embedding_dtype: Dtype::Bfloat16,
    };
    let shape = Shape {
        num_total_requests: 2,
        num_steps: 2,
    };
    let candidates = vec![10, 20, 30, 40, 50, 60, 110, 120, 130, 140, 150, 160];
    let anchors = vec![7, 107];
    let expected_predecessors = vec![7, 7, 7, 10, 20, 30, 107, 107, 107, 110, 120, 130];
    let predecessor_embeddings = embedding_rows(&expected_predecessors, config.rank as usize, 0.013);
    let successor_embeddings = embedding_rows(&candidates, config.rank as usize, -0.009);
    let projected_hidden = (0..shape.proposal_count() * config.rank as usize)
        .map(|index| bf16::from_f32((index as f32 % 9.0 - 4.0) / 7.0).to_bits())
        .collect::<Vec<_>>();
    let candidate_logits = (0..config.candidate_count(shape))
        .map(|index| (index as f32 % 7.0 - 3.0) / 5.0)
        .collect::<Vec<_>>();
    let anchor_buffer = Buffer::from_slice(&device, &anchors);
    let candidate_buffer = Buffer::from_slice(&device, &candidates);
    let predecessor_id_buffer = Buffer::from_slice(&device, &vec![i32::MIN; candidates.len()]);
    let predecessor_embedding_buffer = Buffer::from_slice(&device, &predecessor_embeddings);
    let successor_embedding_buffer = Buffer::from_slice(&device, &successor_embeddings);
    let hidden_buffer = Buffer::from_slice(&device, &projected_hidden);
    let logit_buffer = Buffer::from_slice(&device, &candidate_logits);
    let score_canary = -777.0f32;
    let score_buffer = Buffer::from_slice(&device, &vec![score_canary; config.score_count(shape)]);
    let runtime_params = Buffer::new_zeroed(&device, shape.num_total_requests as usize * 4 * size_of::<u32>());
    for request in 0..shape.num_total_requests as usize {
        runtime_params.write_typed(request * 4, &[0.8f32]);
        runtime_params.write_typed(request * 4 + 1, &[17u32, 29, 0xD1A5_0001]);
    }
    let distribution_indices = Buffer::from_slice(&device, &[2u32, 3, 4, 5]);
    let proposal_token_ids = Buffer::from_slice(&device, &[i32::MIN; 4]);
    let proposal_probs = Buffer::from_slice(&device, &[-1.0f32; 4]);
    let distribution_token_ids = Buffer::from_slice(&device, &[i32::MIN; 6 * 4]);
    let distribution_probs = Buffer::from_slice(&device, &[-1.0f32; 6 * 4]);
    let compute = Compute::new(&device, config);
    let active = ReplayU32::Parameter(NUM_ACTIVE_REQUESTS_KEY);
    let mut builder = stream.create_replay_program();
    builder.record(compute.invoke_predecessor_ids(
        shape,
        active,
        PredecessorIdBuffers {
            anchor_token_ids: &anchor_buffer,
            candidate_token_ids: &candidate_buffer,
            predecessor_token_ids: &predecessor_id_buffer,
        },
    ));
    builder.record_with_barrier_before(compute.invoke_scores(
        shape,
        active,
        ScoreBuffers {
            candidate_logits: &logit_buffer,
            projected_hidden: &hidden_buffer,
            predecessor_embeddings: &predecessor_embedding_buffer,
            successor_embeddings: &successor_embedding_buffer,
            scores: &score_buffer,
        },
    ));
    builder.record_with_barrier_before(compute.invoke_walk(
        shape,
        active,
        WalkBuffers {
            candidate_token_ids: &candidate_buffer,
            scores: &score_buffer,
            runtime_params: &runtime_params,
            output_distribution_indices: &distribution_indices,
            proposal_token_ids: &proposal_token_ids,
            proposal_probs: &proposal_probs,
            distribution_token_ids: &distribution_token_ids,
            distribution_probs: &distribution_probs,
            max_distribution_k: 4,
            num_output_distributions: 6,
        },
    ));
    let replay = builder.build();
    let mut arguments = ReplayArguments::new();
    compute.add_replay_arguments(shape, 1, &mut arguments);
    stream.submit_replay_with_arguments(&replay, &arguments).wait();

    assert_eq!(
        predecessor_id_buffer.read_typed::<i32>(0, candidates.len()),
        [expected_predecessors[..6].to_vec(), vec![i32::MIN; 6]].concat()
    );
    let expected_scores = score_reference(
        config,
        shape,
        &candidate_logits,
        &projected_hidden,
        &predecessor_embeddings,
        &successor_embeddings,
    );
    let actual_scores = score_buffer.read_typed::<f32>(0, config.score_count(shape));
    for (actual, expected) in actual_scores[..18].iter().zip(&expected_scores[..18]) {
        assert!(
            (actual - expected).abs() < 2.0e-4,
            "actual={actual} expected={expected}"
        );
    }
    assert_eq!(&actual_scores[18..], &vec![score_canary; 18]);

    let (expected_tokens, expected_probs, expected_distributions) =
        walk_reference(config, &candidates[..6], &actual_scores[..18], 0.8, 17, 29, 0xD1A5_0001);
    assert_eq!(proposal_token_ids.read_typed::<i32>(0, 2), expected_tokens);
    assert_eq!(proposal_token_ids.read_typed::<i32>(2, 2), vec![i32::MIN; 2]);
    for (actual, expected) in proposal_probs.read_typed::<f32>(0, 2).iter().zip(expected_probs) {
        assert!((actual - expected).abs() < 1.0e-6);
    }
    for step in 0..2 {
        let distribution = 2 + step;
        let ids = distribution_token_ids.read_typed::<i32>(distribution * 4, 4);
        let probs = distribution_probs.read_typed::<f32>(distribution * 4, 4);
        assert_eq!(&ids[..3], &candidates[step * 3..step * 3 + 3]);
        assert_eq!(ids[3], -1);
        for candidate in 0..3 {
            assert!((probs[candidate] - expected_distributions[step][candidate]).abs() < 1.0e-6);
        }
        assert_eq!(probs[3], 0.0);
    }
}

fn embedding_rows(token_ids: &[i32], rank: usize, scale: f32) -> Vec<u16> {
    token_ids
        .iter()
        .flat_map(|&token| {
            (0..rank).map(move |index| bf16::from_f32((token as f32 + index as f32 + 1.0) * scale).to_bits())
        })
        .collect()
}

fn score_reference(
    config: Config,
    shape: Shape,
    unary: &[f32],
    hidden: &[u16],
    predecessors: &[u16],
    successors: &[u16],
) -> Vec<f32> {
    let rank = config.rank as usize;
    let top_k = config.top_k as usize;
    let mut scores = vec![0.0; config.score_count(shape)];
    for request in 0..shape.num_total_requests as usize {
        for step in 0..shape.num_steps as usize {
            let candidate_base = (request * shape.num_steps as usize + step) * top_k;
            let hidden_base = (request * shape.num_steps as usize + step) * rank;
            for predecessor in 0..top_k {
                for successor in 0..top_k {
                    let mut score = unary[candidate_base + successor];
                    for index in 0..rank {
                        score += bf16::from_bits(predecessors[(candidate_base + predecessor) * rank + index]).to_f32()
                            * bf16::from_bits(hidden[hidden_base + index]).to_f32()
                            * bf16::from_bits(successors[(candidate_base + successor) * rank + index]).to_f32();
                    }
                    let edge = ((candidate_base + predecessor) * top_k) + successor;
                    scores[edge] = score;
                }
            }
        }
    }
    scores
}

fn walk_reference(
    config: Config,
    candidates: &[i32],
    scores: &[f32],
    temperature: f32,
    seed: u32,
    sample_position: u32,
    domain: u32,
) -> (Vec<i32>, Vec<f32>, Vec<Vec<f32>>) {
    let top_k = config.top_k as usize;
    let mut previous = 0;
    let mut tokens = Vec::new();
    let mut selected_probs = Vec::new();
    let mut distributions = Vec::new();
    for step in 0..2 {
        let score_base = (step * top_k + previous) * top_k;
        let maximum = scores[score_base..score_base + top_k]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = scores[score_base..score_base + top_k]
            .iter()
            .map(|&score| ((score - maximum) / temperature).exp())
            .collect::<Vec<_>>();
        let total = probabilities.iter().sum::<f32>();
        for probability in &mut probabilities {
            *probability /= total;
        }
        let draw = uniform01(sampling_random(seed, sample_position + step as u32, domain));
        let mut cumulative = 0.0;
        let mut index = top_k - 1;
        for (candidate, &probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if cumulative >= draw {
                index = candidate;
                break;
            }
        }
        tokens.push(candidates[step * top_k + index]);
        selected_probs.push(probabilities[index]);
        distributions.push(probabilities);
        previous = index;
    }
    (tokens, selected_probs, distributions)
}

fn sampling_random(seed: u32, sample_position: u32, domain: u32) -> u32 {
    mix(seed ^ mix(sample_position.wrapping_add(0x9e37_79b9)) ^ domain)
}

fn mix(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn uniform01(random: u32) -> f32 {
    ((random & 0x00ff_ffff) as f32 + 0.5) / 16_777_216.0
}
