use super::test_utils::*;
use crate::runtime::decoder::trie_cache::TryEvictByKeyResult;

#[test]
fn test_peek_by_key_wo_parent() {
    let trie = new_trie();

    let annotations = new_annotations(1);
    let tokens = new_tokens(&[1, 2]);
    let trie_node_key = trie.alloc_trie_node(
        None,
        annotations.clone(),
        tokens.clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );

    let trie_node = trie.peek_by_key(trie_node_key);
    assert_eq!(trie_node_key, trie_node.key());
    assert_eq!(&annotations, trie_node.annotations());
    assert_eq!(&tokens, trie_node.tokens());
}

#[test]
fn test_peek_by_key_w_parent() {
    let trie = new_trie();

    let parent_annotations = new_annotations(1);
    let parent_tokens = new_tokens(&[1, 2]);
    let parent_trie_node_key = trie.alloc_trie_node(
        None,
        parent_annotations.clone(),
        parent_tokens.clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );

    let annotations = new_annotations(2);
    let tokens = new_tokens(&[11, 12]);
    let trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        annotations.clone(),
        tokens.clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );

    let trie_node = trie.peek_by_key(trie_node_key);
    assert_eq!(trie_node_key, trie_node.key());
    assert_eq!(&annotations, trie_node.annotations());
    assert_eq!(&tokens, trie_node.tokens());
}

#[test]
fn test_try_evict_by_key_external() {
    let trie = new_trie();

    let trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(2),
        new_tokens(&[3]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.free_trie_node(trie_node_key, 1);
    assert_eq!(TryEvictByKeyResult::Missing, trie.try_evict_by_key(trie_node_key));
    assert_eq!(0, trie.node_count());

    let trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(3),
        new_tokens(&[4]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    assert_eq!(TryEvictByKeyResult::Rejected, trie.try_evict_by_key(trie_node_key));
    assert_eq!(1, trie.node_count());

    trie.external_unpin_by_key(trie_node_key);
    assert_eq!(TryEvictByKeyResult::Success, trie.try_evict_by_key(trie_node_key));
    assert_eq!(0, trie.node_count());
}

#[test]
fn test_try_evict_by_key_child() {
    let trie = new_trie();

    let trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(2),
        new_tokens(&[3]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.free_trie_node(trie_node_key, 1);
    assert_eq!(TryEvictByKeyResult::Missing, trie.try_evict_by_key(trie_node_key));
    assert_eq!(0, trie.node_count());

    let trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(3),
        new_tokens(&[4]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.external_unpin_by_key(trie_node_key);
    trie.child_pin_by_key(trie_node_key);
    assert_eq!(TryEvictByKeyResult::Rejected, trie.try_evict_by_key(trie_node_key));
    assert_eq!(1, trie.node_count());

    trie.child_unpin_by_key(trie_node_key);
    assert_eq!(TryEvictByKeyResult::Success, trie.try_evict_by_key(trie_node_key));
    assert_eq!(0, trie.node_count());
}
