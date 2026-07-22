use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::attn::GQACore;

use crate::model::qwen::v3_5::plan::Qwen35DSparkLayerPlan;

/// Immutable row layout for a DSpark attention projection.
///
/// DSpark uses ordinary Qwen3 attention and therefore stores `Q | K | V`.
/// It deliberately does not use the target Qwen3.5 `Q | G | K | V` layout.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35DSparkQKVLayout {
    hidden_dim: u32,
    q_dim: u32,
    k_dim: u32,
    v_dim: u32,
    group_size: u32,
    bits: u32,
    dtype: Dtype,
}

impl Qwen35DSparkQKVLayout {
    pub fn from_plan(plan: &Qwen35DSparkLayerPlan) -> Self {
        Self::new(
            &plan.attention_core,
            plan.attention_metal.group_size,
            plan.attention_metal.bits,
            plan.attention_metal.dtype,
        )
    }

    pub fn new(core: &GQACore, group_size: u32, bits: u32, dtype: Dtype) -> Self {
        core.validate();
        assert_eq!(dtype, Dtype::Bfloat16, "DSpark attention requires BF16 affine storage");
        let layout = Self {
            hidden_dim: to_u32("DSpark hidden_dim", core.hidden_dim),
            q_dim: to_u32("DSpark q_dim", core.q_dim()),
            k_dim: to_u32("DSpark k_dim", core.k_dim()),
            v_dim: to_u32("DSpark v_dim", core.v_dim()),
            group_size,
            bits,
            dtype,
        };
        layout.qkv_shape(1).validate();
        layout
    }

    pub fn q_dim(self) -> u32 {
        self.q_dim
    }

    pub fn k_dim(self) -> u32 {
        self.k_dim
    }

    pub fn v_dim(self) -> u32 {
        self.v_dim
    }

    pub fn qkv_dim(self) -> u32 {
        self.q_dim
            .checked_add(self.k_dim)
            .and_then(|dim| dim.checked_add(self.v_dim))
            .expect("DSpark QKV dimension must fit u32")
    }

    pub fn qkv_shape(self, num_tokens: u32) -> AffineQuantizedMatmulShape {
        self.shape(num_tokens, self.qkv_dim())
    }

    pub fn q_shape(self, num_tokens: u32) -> AffineQuantizedMatmulShape {
        self.shape(num_tokens, self.q_dim)
    }

    pub fn k_shape(self, num_tokens: u32) -> AffineQuantizedMatmulShape {
        self.shape(num_tokens, self.k_dim)
    }

    pub fn v_shape(self, num_tokens: u32) -> AffineQuantizedMatmulShape {
        self.shape(num_tokens, self.v_dim)
    }

    pub fn k_weight_offset_bytes(self) -> usize {
        self.shape(1, self.q_dim).weight_bytes()
    }

    pub fn v_weight_offset_bytes(self) -> usize {
        self.k_weight_offset_bytes()
            .checked_add(self.shape(1, self.k_dim).weight_bytes())
            .expect("DSpark V weight offset must fit usize")
    }

    pub fn k_affine_offset_bytes(self) -> usize {
        self.shape(1, self.q_dim).affine_param_bytes()
    }

    pub fn v_affine_offset_bytes(self) -> usize {
        self.k_affine_offset_bytes()
            .checked_add(self.shape(1, self.k_dim).affine_param_bytes())
            .expect("DSpark V affine offset must fit usize")
    }

    fn shape(self, num_tokens: u32, output_dim: u32) -> AffineQuantizedMatmulShape {
        assert!(num_tokens > 0, "DSpark attention projection requires tokens");
        AffineQuantizedMatmulShape::same_dtype(
            num_tokens.try_into().expect("DSpark token count must fit i32"),
            output_dim.try_into().expect("DSpark output dimension must fit i32"),
            self.hidden_dim
                .try_into()
                .expect("DSpark hidden dimension must fit i32"),
            self.group_size.try_into().expect("DSpark group size must fit i32"),
            self.bits.try_into().expect("DSpark bits must fit i32"),
            self.dtype,
        )
    }
}

fn to_u32(name: &str, value: usize) -> u32 {
    value.try_into().unwrap_or_else(|_| panic!("{name} must fit u32"))
}

#[cfg(test)]
mod tests {
    use inference_executor_core::attn::GQACore;

    use super::*;

    #[test]
    fn qkv_layout_excludes_qwen35_activation_gate_and_offsets_kv_rows() {
        let core = GQACore::new(0, 5120, 128, 40, 8, 128.0_f32.sqrt().recip());
        let layout = Qwen35DSparkQKVLayout::new(&core, 64, 4, Dtype::Bfloat16);
        assert_eq!(layout.q_dim(), 5120);
        assert_eq!(layout.k_dim(), 1024);
        assert_eq!(layout.v_dim(), 1024);
        assert_eq!(layout.qkv_dim(), 7168);
        assert_eq!(core.qgkv_dim(), 12_288);

        let q = layout.shape(1, layout.q_dim());
        let k = layout.k_shape(1);
        assert_eq!(layout.k_weight_offset_bytes(), q.weight_bytes());
        assert_eq!(layout.v_weight_offset_bytes(), q.weight_bytes() + k.weight_bytes());
        assert_eq!(layout.k_affine_offset_bytes(), q.affine_param_bytes());
        assert_eq!(
            layout.v_affine_offset_bytes(),
            q.affine_param_bytes() + k.affine_param_bytes()
        );
    }
}
