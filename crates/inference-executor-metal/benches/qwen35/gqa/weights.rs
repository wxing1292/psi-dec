use super::super::*;

pub struct RealGQAWeights {
    pub qgkv_weight: Buffer,
    pub qgkv_scales: Buffer,
    pub qgkv_biases: Buffer,
    pub q_norm_weight: Buffer,
    pub k_norm_weight: Buffer,
    pub output_weight: Buffer,
    pub output_scales: Buffer,
    pub output_biases: Buffer,
}

impl RealGQAWeights {
    pub fn load(device: &Device, tensors: &SafeTensors<'_>, model: GQAModelProfile) -> Self {
        let prefix = format!("language_model.model.layers.{}.self_attn", model.model_layer_index);
        let q_weight = gqa_tensor_bytes(
            tensors,
            &format!("{prefix}.q_proj.weight"),
            safetensors::Dtype::U32,
            model,
        );
        let q_scales = gqa_tensor_bytes(
            tensors,
            &format!("{prefix}.q_proj.scales"),
            safetensors::Dtype::BF16,
            model,
        );
        let q_biases = gqa_tensor_bytes(
            tensors,
            &format!("{prefix}.q_proj.biases"),
            safetensors::Dtype::BF16,
            model,
        );
        let k_weight = gqa_tensor_bytes(
            tensors,
            &format!("{prefix}.k_proj.weight"),
            safetensors::Dtype::U32,
            model,
        );
        let k_scales = gqa_tensor_bytes(
            tensors,
            &format!("{prefix}.k_proj.scales"),
            safetensors::Dtype::BF16,
            model,
        );
        let k_biases = gqa_tensor_bytes(
            tensors,
            &format!("{prefix}.k_proj.biases"),
            safetensors::Dtype::BF16,
            model,
        );
        let v_weight = gqa_tensor_bytes(
            tensors,
            &format!("{prefix}.v_proj.weight"),
            safetensors::Dtype::U32,
            model,
        );
        let v_scales = gqa_tensor_bytes(
            tensors,
            &format!("{prefix}.v_proj.scales"),
            safetensors::Dtype::BF16,
            model,
        );
        let v_biases = gqa_tensor_bytes(
            tensors,
            &format!("{prefix}.v_proj.biases"),
            safetensors::Dtype::BF16,
            model,
        );
        let qgkv_weight = concat_parts(&[&q_weight, &k_weight, &v_weight]);
        let qgkv_scales = concat_parts(&[&q_scales, &k_scales, &v_scales]);
        let qgkv_biases = concat_parts(&[&q_biases, &k_biases, &v_biases]);
        validate_qgkv_sizes(&qgkv_weight, &qgkv_scales, &qgkv_biases, model);
        Self {
            qgkv_weight: Buffer::from_slice(device, &qgkv_weight),
            qgkv_scales: Buffer::from_slice(device, &qgkv_scales),
            qgkv_biases: Buffer::from_slice(device, &qgkv_biases),
            q_norm_weight: Buffer::from_slice(
                device,
                &gqa_bf16_tensor_as_f32(tensors, &format!("{prefix}.q_norm.weight"), model),
            ),
            k_norm_weight: Buffer::from_slice(
                device,
                &gqa_bf16_tensor_as_f32(tensors, &format!("{prefix}.k_norm.weight"), model),
            ),
            output_weight: Buffer::from_slice(
                device,
                &gqa_tensor_bytes(
                    tensors,
                    &format!("{prefix}.o_proj.weight"),
                    safetensors::Dtype::U32,
                    model,
                ),
            ),
            output_scales: Buffer::from_slice(
                device,
                &gqa_tensor_bytes(
                    tensors,
                    &format!("{prefix}.o_proj.scales"),
                    safetensors::Dtype::BF16,
                    model,
                ),
            ),
            output_biases: Buffer::from_slice(
                device,
                &gqa_tensor_bytes(
                    tensors,
                    &format!("{prefix}.o_proj.biases"),
                    safetensors::Dtype::BF16,
                    model,
                ),
            ),
        }
    }
}

pub struct MappedFile {
    ptr: *mut libc::c_void,
    len: usize,
}

impl MappedFile {
    pub fn open(path: &Path) -> Self {
        let file = File::open(path).unwrap_or_else(|err| panic!("unable to open {}: {err}", path.display()));
        let len = file
            .metadata()
            .unwrap_or_else(|err| panic!("unable to stat {}: {err}", path.display()))
            .len() as usize;
        assert!(len > 0, "safetensors shard must not be empty");
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            panic!("unable to mmap {}: {}", path.display(), std::io::Error::last_os_error());
        }
        unsafe {
            let _ = libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
        Self { ptr, len }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}
