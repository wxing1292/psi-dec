//! Qwen3x DFlash2 Spec Prefill and Decode execution.
//!
//! DFlash2 owns two independent replay stages. Prefill projects selected Main residuals and writes persistent history
//! K/V. Decode runs an anchor plus MASK query block. Each layer combines sliding-history attention with full
//! bidirectional block attention. The output owner selects a probabilistic candidate path and writes sparse draft
//! distributions for Main rejection sampling.
//!
//! ```text
//! Main selected residual capture
//!             |
//!             v
//! +---------------- Spec Prefill ----------------+
//! | Main-feature FC -> hidden norm               |
//! |             |                                |
//! |             +-> layer Wk/Wv -> persistent KV |
//! +----------------------------------------------+
//!
//! [anchor, MASK, ..., MASK]
//!             |
//!             v
//! +---------------- Spec Decode -----------------------------------------+
//! | Main Embed                                                          |
//! |   -> per layer                                                       |
//! |      norm -> attention conv prepare                                 |
//! |           -> sliding history + bidirectional block GQA              |
//! |           -> attention conv finish -> residual                      |
//! |      norm -> MLP conv prepare -> MLP -> MLP conv finish -> residual |
//! |   -> final norm                                                      |
//! |   -> gather MASK rows -> Main Unembed -> raw Top-K                  |
//! |   -> candidate lattice -> probabilistic path                        |
//! |   -> proposal tokens + sparse draft distributions                   |
//! +---------------------------------------------------------------------+
//! ```

pub mod attention;
pub mod conv;
pub mod embed;
pub mod execution;
pub mod layer;
pub mod load;
pub mod main_feature;
pub mod model;
pub mod output;
