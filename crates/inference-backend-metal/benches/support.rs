use half::bf16;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;

pub fn logits_fixture(tokens: usize, experts: usize) -> Vec<f32> {
    (0..tokens * experts)
        .map(|index| ((index * 17 + 11) % 97) as f32 * 0.03125 - 1.5)
        .collect()
}

pub fn hidden_fixture(rows: usize, hidden_dim: usize) -> Vec<f32> {
    (0..rows * hidden_dim)
        .map(|index| ((index * 13 + 5) % 31) as f32 * 0.0625 - 1.0)
        .collect()
}

pub fn route_probs_fixture(tokens: usize, topk_experts: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(tokens * topk_experts);
    for token in 0..tokens {
        let mut sum = 0.0f32;
        let start = values.len();
        for slot in 0..topk_experts {
            let value = 1.0 + ((token * 7 + slot * 3) % 11) as f32;
            values.push(value);
            sum += value;
        }
        for value in &mut values[start..] {
            *value /= sum;
        }
    }
    values
}

pub fn gate_fixture(tokens: usize) -> Vec<f32> {
    (0..tokens)
        .map(|index| ((index * 5) % 17) as f32 * 0.125 - 1.0)
        .collect()
}

pub fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
    Buffer::from_slice(device, &bits)
}

pub fn token_route_indices(tokens: usize, topk_experts: usize) -> Vec<u32> {
    (0..tokens * topk_experts)
        .map(|route| u32::try_from(route / topk_experts).expect("token route index must fit u32"))
        .collect()
}

pub fn expert_route_indices(tokens: usize, topk_experts: usize, num_experts: usize) -> Vec<u32> {
    (0..tokens * topk_experts)
        .map(|route| {
            u32::try_from((route * 7 + route / topk_experts) % num_experts).expect("expert index must fit u32")
        })
        .collect()
}

pub fn repeated_topk_expert_indices(tokens: usize, topk_experts: usize) -> Vec<u32> {
    (0..tokens * topk_experts)
        .map(|route| u32::try_from(route % topk_experts).expect("expert index must fit u32"))
        .collect()
}

pub fn identity_indices(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| u32::try_from(index).expect("identity index must fit u32"))
        .collect()
}

pub fn quantized_weight_stack_for_experts(device: &Device, experts: usize, bytes_per_expert: usize) -> Buffer {
    let total_bytes = experts * bytes_per_expert;
    let bytes = (0..total_bytes)
        .map(|index| ((index * 13 + 17) & 0xff) as u8)
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &bytes)
}

pub fn quantized_weight(device: &Device, len: usize) -> Buffer {
    let bytes = (0..len)
        .map(|index| ((index * 13 + 17) & 0xff) as u8)
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &bytes)
}

pub fn affine_param_fixture(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| 0.001 + ((index * 3) % 7) as f32 * 0.0001)
        .collect()
}

pub fn zero_fixture(len: usize) -> Vec<f32> {
    vec![0.0; len]
}
