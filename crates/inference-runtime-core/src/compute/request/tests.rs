use super::*;
use crate::runtime::Token;

#[test]
fn test_batch_construction_groups_query_phases() {
    let requests = [(1, false, 1, 2), (2, true, 1, 0), (3, false, 4, 0), (4, true, 2, 0)]
        .into_iter()
        .map(|(req_id, prefill, num_tokens, num_spec_tokens)| {
            let tokens = vec![Token::new(req_id as u32); num_tokens];
            let query = if prefill {
                QueryTokens::Prefill {
                    epoch: req_id,
                    token_index: req_id * 10,
                    tokens,
                    window: num_tokens,
                }
            } else {
                QueryTokens::Decode {
                    epoch: req_id,
                    token_index: req_id * 10,
                    tokens,
                    spec_tokens: vec![Token::new(100); num_spec_tokens],
                }
            };
            DeviceRequest::new(
                req_id,
                req_id as u32,
                query,
                DecoderSyncBlocks::new(req_id, Vec::new(), Vec::new()),
                None,
                Vec::new(),
                SamplingConfig::default(),
            )
        })
        .collect::<Vec<_>>();
    let batch = BatchDeviceRequest::from_parts(7, requests);
    assert_eq!(
        batch.dev_reqs.iter().map(DevReq::id).collect::<Vec<_>>(),
        vec![2, 4, 1, 3]
    );
}
