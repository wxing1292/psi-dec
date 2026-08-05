use inference_executor_core::attn::GQAReplayShape;

use crate::attn::gqa::backend::GQAReplayTopology;

pub mod executor;
pub mod main;
pub mod plan;

mod mtp;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35GQAReplayKey {
    total_tokens: u32,
    total_q_token_tiles: u32,
    total_sdpa_map_task_templates: u32,
    topology: GQAReplayTopology,
}

impl Qwen35GQAReplayKey {
    pub fn new(shape: GQAReplayShape, topology: GQAReplayTopology) -> Self {
        shape.validate();
        Self {
            total_tokens: shape.total_tokens,
            total_q_token_tiles: shape.total_q_token_tiles,
            total_sdpa_map_task_templates: shape.total_sdpa_map_task_templates,
            topology,
        }
    }

    #[cfg(test)]
    pub fn debug_parts(&self) -> (u32, u32, u32, GQAReplayTopology) {
        (
            self.total_tokens,
            self.total_q_token_tiles,
            self.total_sdpa_map_task_templates,
            self.topology,
        )
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::GQAComputePath;
    use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;
    use inference_executor_core::attn::GQAReplayShape;

    use super::Qwen35GQAReplayKey;
    use crate::attn::gqa::backend::GQAReplayTopology;

    #[test]
    fn gqa_key_uses_capacities_and_ignores_active_counts() {
        let topology = single_topology();
        let smaller = GQAReplayShape::new(3, 4, 2, 4, 2, 4, false);
        let fuller = GQAReplayShape::new(4, 4, 3, 4, 3, 4, true);
        let base = Qwen35GQAReplayKey::new(smaller, topology);

        assert_eq!(base, Qwen35GQAReplayKey::new(fuller, topology));
        assert_ne!(
            base,
            Qwen35GQAReplayKey::new(GQAReplayShape::new(3, 6, 2, 4, 2, 4, false), topology)
        );
        assert_ne!(
            base,
            Qwen35GQAReplayKey::new(GQAReplayShape::new(3, 4, 2, 6, 2, 4, false), topology)
        );
        assert_ne!(
            base,
            Qwen35GQAReplayKey::new(GQAReplayShape::new(3, 4, 2, 4, 2, 6, false), topology)
        );
    }

    #[test]
    fn gqa_key_separates_compute_and_affine_topology() {
        let shape = GQAReplayShape::new(3, 4, 2, 4, 2, 4, false);
        let base = Qwen35GQAReplayKey::new(shape, single_topology());
        let variants = [
            GQAReplayTopology {
                compute_path: GQAComputePath::SingleQueryToken {
                    kv_token_tile_size: 128,
                    num_threads_per_threadblock: 256,
                    q_head_tile_size: 6,
                },
                ..single_topology()
            },
            GQAReplayTopology {
                compute_path: GQAComputePath::SingleQueryToken {
                    kv_token_tile_size: 256,
                    num_threads_per_threadblock: 128,
                    q_head_tile_size: 6,
                },
                ..single_topology()
            },
            GQAReplayTopology {
                compute_path: GQAComputePath::SingleQueryToken {
                    kv_token_tile_size: 256,
                    num_threads_per_threadblock: 256,
                    q_head_tile_size: 3,
                },
                ..single_topology()
            },
            GQAReplayTopology {
                compute_path: GQAComputePath::TiledQueryTokens {
                    q_token_tile_size: 8,
                    kv_token_tile_size: 16,
                    q_head_tile_size: 6,
                },
                ..single_topology()
            },
            GQAReplayTopology {
                qgkv_affine: AffineQuantizedMatmulKernelKind::QmmBm8Bn32,
                ..single_topology()
            },
            GQAReplayTopology {
                output_affine: AffineQuantizedMatmulKernelKind::QmmBm16Bn32,
                ..single_topology()
            },
        ];

        for topology in variants {
            assert_ne!(base, Qwen35GQAReplayKey::new(shape, topology));
        }

        let tiled = Qwen35GQAReplayKey::new(shape, tiled_topology());
        for compute_path in [
            GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 16,
                kv_token_tile_size: 16,
                q_head_tile_size: 6,
            },
            GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 8,
                kv_token_tile_size: 8,
                q_head_tile_size: 6,
            },
            GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 8,
                kv_token_tile_size: 16,
                q_head_tile_size: 3,
            },
        ] {
            assert_ne!(
                tiled,
                Qwen35GQAReplayKey::new(
                    shape,
                    GQAReplayTopology {
                        compute_path,
                        ..tiled_topology()
                    }
                )
            );
        }
    }

    fn single_topology() -> GQAReplayTopology {
        GQAReplayTopology {
            compute_path: GQAComputePath::SingleQueryToken {
                kv_token_tile_size: 256,
                num_threads_per_threadblock: 256,
                q_head_tile_size: 6,
            },
            qgkv_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            output_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
        }
    }

    fn tiled_topology() -> GQAReplayTopology {
        GQAReplayTopology {
            compute_path: GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 8,
                kv_token_tile_size: 16,
                q_head_tile_size: 6,
            },
            qgkv_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            output_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
        }
    }
}
