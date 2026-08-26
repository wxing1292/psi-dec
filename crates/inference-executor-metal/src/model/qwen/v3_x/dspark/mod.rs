//! Qwen3x DSpark Spec Prefill and Decode execution.
//!
//! `Qwen3xDSparkExecution` owns two independent recordings. Prefill creates persistent history K/V. Decode consumes
//! that history and creates one proposal block. When both recordings exist, the execution owner submits Spec Decode
//! prepare first, then Prefill, and then the remaining Spec Decode replay sequence.
//!
//! ```text
//! Residual capture from selected Main layers             Main decision
//! [Tmain, Lselected * H]                                 {sampled anchor}
//!             |                                                   |
//!             v                                                   v
//! +------------------ Spec Prefill ----------------+   +---------------- Spec Decode ----------------+
//! |                                                |   |                                             |
//! | Main-feature FC -> hidden norm                 |   | [anchor, MASK, ..., MASK] -> Embed          |
//! |                     |                          |   |                         |                   |
//! |                     v                          |   |                         v                   |
//! |            shared main_feature                 |   |               for each DSpark layer         |
//! |                     |                          |   |                         |                   |
//! |          +----------+----------+               |   |            input norm -> local Q/K/V       |
//! |          |          |          |               |   |                    /           \          |
//! |          v          v          v               |   |                   v             v          |
//! |       layer 0     layer 1   layer L-1           |   |       history GQA partial   block partial  |
//! |       Wk / Wv     Wk / Wv   Wk / Wv            |   |       persistent K/V       local bidi K/V  |
//! |          |          |          |               |   |                   \             /          |
//! |          v          v          v               |   |                    v           v           |
//! |       K norm + RoPE, V                         |   |                  partial reduce             |
//! |          |          |          |               |   |                         |                   |
//! |          v          v          v               |   |             output + residual + MLP        |
//! |       persistent paged history K/V             |   |                         |                   |
//! |                                                |   |                         v                   |
//! +------------------------------------------------+   |          FinalNorm -> Gather + Unembed      |
//!                                                      |                         |                   |
//!                                                      |                         v                   |
//!                                                      |       Markov correction + confidence        |
//!                                                      |                         |                   |
//!                                                      |                         v                   |
//!                                                      |       proposal tokens/probabilities         |
//!                                                      +---------------------------------------------+
//! ```
//!
//! Each selected Main layer writes every Main row directly into its capture columns. Prefill consumes this complete
//! row prefix. Decode owns the complete body and proposal output composition. Its local Q/K/V and attention partials
//! are ephemeral.

pub mod attention;
pub mod embed;
pub mod execution;
pub mod layer;
pub mod load;
pub mod main_feature;
pub mod model;
pub mod output;
pub mod sampling;
