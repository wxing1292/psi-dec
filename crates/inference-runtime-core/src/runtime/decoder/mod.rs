mod allocator;
pub use allocator::KVBlockAllocationResult;
pub use allocator::KVBlockAllocator;
pub use allocator::KVBlockPlacement;
pub use allocator::StateBlockAllocationResult;
pub use allocator::StateBlockAllocator;
pub use allocator::StateBlockPlacement;
pub use allocator::TPKVBlockAllocator;
pub use allocator::TPStateBlockAllocator;

mod resource;
pub use resource::BlockAnnotation;
pub use resource::ResourceDigest;
pub use resource::ResourceSegment;

// pub mod hash_cache;
pub mod trie_cache;

// KV block layout
//
// Overall
// +-------+----+----+----+----+----+----+----+
// | Index | 0  | 1  | 2  | 3  | 4  | 5  | 6  |
// +-------+----+----+----+----+----+----+----+
// | Main  | t0 | t1 | t2 | t3 | t4 | t5 | t6 |
// | MTP0  | t1 | t2 | t3 | t4 | t5 | t6 |    |
// | MTP1  | t2 | t3 | t4 | t5 | t6 |    |    |
// | MTP2  | t3 | t4 | t5 | t6 |    |    |    |
// +-------+----+----+----+----+----+----+----+
//
// Prefill KV block layout:
//  input t0-t6
// +-------+----+----+----+----+
// | Index | 0  | 1  | 2  | 3  |
// +-------+----+----+----+----+
// | Main  | t0 | t1 | t2 | t3 |
// | MTP0  | t1 | t2 | t3 | t4 |
// | MTP1  | t2 | t3 | t4 | t5 |
// | MTP2  | t3 | t4 | t5 | t6 |
// +-------+----+----+----+----+
//
// Decode KV block layout:
//  input t0-t2, output t3 + s4-s6
// +-------+----+----+----+--------+
// | Index | 0  | 1  | 2  | sample |
// +-------+----+----+----+--------+
// | Main  | t0 | t1 | t2 | t3     |
// | MTP0  | t1 | t2 | t3 | s4     |
// | MTP1  | t2 | t3 | s4 | s5     |
// | MTP2  | t3 | s4 | s5 | s6     |
// +-------+----+----+----+--------+
//
// Decode KV block layout:
//  input t0-t2, s3, output t3 + s4-s6
//  NOTE: input s3 is rejected, resample to t3
// +-------+----+----+----+--------+
// | Index | 0  | 1  | 2  | sample |
// +-------+----+----+----+--------+
// | Main  | t0 | t1 | t2 | t3     |
// | MTP0  | t1 | t2 | t3 | s4     |
// | MTP1  | t2 | t3 | s4 | s5     |
// | MTP2  | t3 | s4 | s5 | s6     |
// +-------+----+----+----+--------+
//
// Decode KV block layout:
//  input t0-t2, s3, output t4 + s5-s7
//  NOTE: input s3 is accepted as t3, sample to t4
// +-------+----+----+----+----+--------+
// | Index | 0  | 1  | 2  | 3  | sample |
// +-------+----+----+----+----+--------+
// | Main  | t0 | t1 | t2 | t3 | t4     |
// | MTP0  | t1 | t2 | t3 | t4 | s5     |
// | MTP1  | t2 | t3 | t4 | s5 | s6     |
// | MTP2  | t3 | t4 | s5 | s6 | s7     |
// +-------+----+----+----+----+--------+
//
// Decode KV block layout:
//  input t0, output t1 + s2-s4
//  NOTE: input s3 is accepted as t3, sample to t4
// +-------+----+--------+
// | Index | 0  | sample |
// +-------+----+--------+
// | Main  | t0 | t1     |
// | MTP0  | t1 | s2     |
// | MTP1  | s2 | s3     |
// | MTP2  | s3 | s4     |
// +-------+----+--------+
