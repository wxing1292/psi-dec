use super::sanity_check_cache_lane_token_windows;
use crate::runtime::Token;

#[test]
fn test_matching_token_windows() {
    let cache_lane_total_tokens = fixture_cached_token_windows();
    let total_tokens = [0, 1, 2, 3, 4, 5].map(Token::new);
    // Prefill: all shifted token identities are canonical.
    sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens, &[], 2, 3);
    // Decode: Main anchor 3 is canonical; 4, 5, and 6 are drafts.
    let spec_tokens = [4, 5, 6].map(Token::new);
    sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens[..4], &spec_tokens, 1, 3);
}

#[test]
fn test_no_blocks() {
    let cache_lane_total_tokens = [vec![], vec![], vec![], vec![]];
    // The owner requires at least L - 1 tokens at initialization.
    let total_tokens = [0, 1, 2].map(Token::new);
    sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens, &[], 0, 0);
}

#[test]
fn test_replaced_speculative_tail() {
    let cache_lane_total_tokens = fixture_cached_token_windows();
    let total_tokens = [0, 1, 2, 3, 50, 51, 52].map(Token::new);
    // New canonical tokens replace drafts 4 and 5. Cached MTP rows still
    // contain the old IDs until commit; the anchor 3 must remain unchanged.
    sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens, &[], 1, 3);
}

#[test]
fn test_uncached_placeholders() {
    let mut cache_lane_total_tokens = fixture_cached_token_windows();
    cache_lane_total_tokens[0].push(Token::new(3));
    for tokens in &mut cache_lane_total_tokens[1..] {
        tokens.push(Token::default());
    }
    let total_tokens = [0, 1, 2, 3].map(Token::new);
    // Only draft 4 is submitted. Pending MTP slots can remain placeholders,
    // whether or not the current source contains their token IDs.
    sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens, &[Token::new(4)], 1, 3);
}

#[test]
#[should_panic(expected = "cached anchor changed")]
fn test_changed_anchor() {
    let total_tokens = [0, 1, 2, 30, 50, 51, 52].map(Token::new);
    sanity_check_cache_lane_token_windows(&fixture_cached_token_windows(), &total_tokens, &[], 1, 3);
}

#[test]
#[should_panic(expected = "cached lane=2 shifted window")]
fn test_inconsistent_cached_window() {
    let mut cache_lane_total_tokens = fixture_cached_token_windows();
    cache_lane_total_tokens[2][1] = Token::new(99);
    let total_tokens = [0, 1, 2, 3, 50, 51, 52].map(Token::new);
    sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens, &[], 1, 3);
}

#[test]
#[should_panic(expected = "cached lane=2 index=1 is a placeholder")]
fn test_cached_placeholder() {
    let mut cache_lane_total_tokens = fixture_cached_token_windows();
    cache_lane_total_tokens[2][1] = Token::default();
    let total_tokens = [0, 1, 2, 3, 50, 51, 52].map(Token::new);
    sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens, &[], 1, 3);
}

fn fixture_cached_token_windows() -> [Vec<Token>; 4] {
    // Cache-local indices 0..3 are cached in every lane.
    // +-------+---+---+---+
    // | Index | 0 | 1 | 2 |
    // | Main  | 0 | 1 | 2 |
    // | MTP0  | 1 | 2 | 3 |
    // | MTP1  | 2 | 3 | 4 |
    // | MTP2  | 3 | 4 | 5 |
    // +-------+---+---+---+
    [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]].map(|tokens| tokens.map(Token::new).to_vec())
}
