use super::test_utils::*;
use crate::runtime::decoder::KVBlockPlacement;
use crate::runtime::decoder::trie_cache::TrieNodeState;

#[test]
fn test_alloc_free_trie_node_wo_parent() {
    let trie = new_trie();

    let annotations = new_annotations(1);
    let tokens = new_tokens(&[1, 2, 3]);
    let trie_node_key = trie.alloc_trie_node(
        None,
        annotations.clone(),
        tokens.clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    assert_eq!(1, trie.node_count());
    assert_eq!(1, trie.num_pinned_trie_node());

    let trie_node = trie.peek_by_key(trie_node_key);
    assert_eq!(trie_node_key, trie_node.key());
    assert_eq!(TrieNodeState::Valid, trie_node.state());
    assert_eq!(1, trie_node.external_pin_count());
    assert_eq!(0, trie_node.child_pin_count());
    assert_eq!(None, trie_node.parent_node_key());
    assert_eq!(&annotations, trie_node.annotations());
    assert_eq!(&tokens, trie_node.tokens());
    assert!(matches!(trie_node.kv_placement(), KVBlockPlacement::Device { .. }));
    drop(trie_node);

    trie.free_trie_node(trie_node_key, 1);
    assert_eq!(0, trie.node_count());
    assert_eq!(0, trie.num_pinned_trie_node());
    assert!(trie.trie_nodes.get_ref(trie_node_key).is_none());
}

#[test]
fn test_alloc_free_trie_node_w_parent() {
    let trie = new_trie();

    let parent_annotations = new_annotations(1);
    let parent_tokens = new_tokens(&[1, 2, 3]);
    let parent_trie_node_key = trie.alloc_trie_node(
        None,
        parent_annotations,
        parent_tokens.clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );

    let annotations = new_annotations(2);
    let tokens = new_tokens(&[11, 12, 13]);
    let trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        annotations.clone(),
        tokens.clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    assert_eq!(2, trie.node_count());
    assert_eq!(2, trie.num_pinned_trie_node());

    let trie_node = trie.peek_by_key(trie_node_key);
    assert_eq!(trie_node_key, trie_node.key());
    assert_eq!(TrieNodeState::Valid, trie_node.state());
    assert_eq!(1, trie_node.external_pin_count());
    assert_eq!(0, trie_node.child_pin_count());
    assert_eq!(Some(parent_trie_node_key), trie_node.parent_node_key());
    assert_eq!(&annotations, trie_node.annotations());
    assert_eq!(&tokens, trie_node.tokens());
    assert!(matches!(trie_node.kv_placement(), KVBlockPlacement::Device { .. }));
    drop(trie_node);

    trie.free_trie_node(trie_node_key, 1);
    trie.free_trie_node(parent_trie_node_key, 1);
    assert_eq!(0, trie.node_count());
    assert_eq!(0, trie.num_pinned_trie_node());
    assert!(trie.trie_nodes.get_ref(trie_node_key).is_none());
}
