use super::*;

pub fn run(args: Args) {
    let device = Device::system_default();
    let mapped = MappedFile::open(&args.model_dir.join(GDN_SHARD));
    let tensors = SafeTensors::deserialize(mapped.as_bytes()).unwrap_or_else(|err| {
        panic!(
            "unable to deserialize safetensors shard {}: {err:?}",
            args.model_dir.join(GDN_SHARD).display()
        )
    });
    let weights = RealGDNWeights::load(&device, &tensors);
    let contexts = if args.contexts.is_empty() {
        vec![0]
    } else {
        args.contexts
    };

    for num_tokens in args.tokens {
        for &num_reqs in &args.num_reqs {
            if !valid_num_reqs(num_tokens, num_reqs) {
                print_skip(num_tokens, num_reqs, None, None, "num_reqs_exceeds_tokens");
                continue;
            }
            for &existing_context_len in &contexts {
                let fixture = RealGDNFixture::new(&device, num_tokens, num_reqs, existing_context_len, &weights);
                fixture.measure(args.warmup_iters, args.iters, args.runs);
                if args.subcomponents {
                    fixture.measure_subcomponents(args.warmup_iters, args.iters, args.runs);
                }
            }
        }
    }
}

struct RealGDNFixture<'a> {
    device: Device,
    stream: Stream,
    num_tokens: u32,
    num_reqs: u32,
    existing_context_len: u32,
    next_hidden_state: Buffer,
    replay: ReplayProgram,
    hidden_state: Buffer,
    hidden_state_f32: Buffer,
    batch_metadata: GDNMetadataBuffers,
    conv_state: Buffer,
    next_conv_state: Buffer,
    recurrent_state_arena: Buffer,
    qkvabz: Buffer,
    projected_qkv: Buffer,
    a: Buffer,
    b: Buffer,
    z: Buffer,
    conv_qkv: Buffer,
    recurrent_output: Buffer,
    pre_output_hidden_states: Buffer,
    pre_output_hidden_states_bf16: Buffer,
    weights: &'a RealGDNWeights,
}

impl<'a> RealGDNFixture<'a> {
    fn new(
        device: &Device,
        num_tokens: u32,
        num_reqs: u32,
        existing_context_len: u32,
        weights: &'a RealGDNWeights,
    ) -> Self {
        assert!(
            valid_num_reqs(num_tokens, num_reqs),
            "GDN bench requires 1 <= num_reqs <= num_tokens"
        );
        let stream = Stream::new(device);
        let core = GDNCore {
            model_layer_index: GDN_LAYER,
            hidden_dim: HIDDEN_DIM,
            num_qk_heads: GDN_QK_HEADS,
            qk_head_dim: GDN_QK_HEAD_DIM,
            num_v_heads: GDN_V_HEADS,
            v_head_dim: GDN_V_HEAD_DIM,
            conv_kernel_size: GDN_CONV_KERNEL_SIZE,
            q_scale: (GDN_QK_HEAD_DIM as f32).sqrt().recip(),
        };
        let config = GDNMetalConfig {
            group_size: GROUP_SIZE,
            bits: BITS,
            recurrent_v_tile_size: 8,
            norm_eps: GDN_EPS,
            input_dtype: Dtype::Float32,
            qkvabz_affine_dtype: Dtype::Float32,
            output_affine_dtype: Dtype::Bfloat16,
        };
        let backend = GDN::new(device, core, config);
        let hidden_state = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM));
        let hidden_state_f32 = Buffer::new_zeroed(device, num_tokens as usize * HIDDEN_DIM * size_of::<f32>());
        let next_hidden_state =
            Buffer::new_zeroed(device, num_tokens as usize * HIDDEN_DIM * Dtype::Bfloat16.item_size());
        let num_tokens_per_req = request_token_counts(num_tokens, num_reqs);
        let batch_metadata = GDNMetadataBuffers::new(device, num_reqs as usize, num_tokens as usize);
        batch_metadata.update(
            &cu_tokens(&num_tokens_per_req)
                .into_iter()
                .map(|value| value as u32)
                .collect::<Vec<_>>(),
            &(0..num_reqs).collect::<Vec<_>>(),
            &(num_reqs..2 * num_reqs).collect::<Vec<_>>(),
            &vec![u32::MAX; num_tokens as usize],
        );
        let conv_state = Buffer::from_slice(
            device,
            &gdn_conv_state_fixture(
                existing_context_len,
                num_reqs as usize,
                2 * num_reqs as usize * GDN_CONV_DIM * (GDN_CONV_KERNEL_SIZE - 1),
            ),
        );
        let next_conv_state = Buffer::new_zeroed(
            device,
            2 * num_reqs as usize * GDN_CONV_DIM * (GDN_CONV_KERNEL_SIZE - 1) * size_of::<f32>(),
        );
        let recurrent_state_arena = Buffer::from_slice(
            device,
            &gdn_recurrent_state_fixture(
                existing_context_len,
                num_reqs as usize,
                2 * num_reqs as usize * GDN_V_HEADS * GDN_V_HEAD_DIM * GDN_QK_HEAD_DIM,
            ),
        );
        let qkvabz = Buffer::new_zeroed(
            device,
            num_tokens as usize * GDN_QKVABZ_DIM * config.internal_dtype().item_size(),
        );
        let projected_qkv = Buffer::new_zeroed(device, num_tokens as usize * GDN_CONV_DIM * size_of::<f32>());
        let a = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_HEADS * size_of::<f32>());
        let b = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_HEADS * size_of::<f32>());
        let z = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_DIM * size_of::<f32>());
        let conv_qkv = Buffer::new_zeroed(device, num_tokens as usize * GDN_CONV_DIM * size_of::<f32>());
        let recurrent_output = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_DIM * size_of::<f32>());
        let pre_output_hidden_states = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_DIM * size_of::<f32>());
        let pre_output_hidden_states_bf16 =
            Buffer::new_zeroed(device, num_tokens as usize * GDN_V_DIM * Dtype::Bfloat16.item_size());
        let mut builder = MetalReplayRuntime::new(&stream).create_recorder();
        let _ = <GDN as ReplayLayer>::record(
            &backend,
            &mut builder,
            GDNInput {
                hidden_state: &hidden_state,
                next_hidden_state: &next_hidden_state,
                scratch: GDNScratchBindings {
                    hidden_state_f32: &hidden_state_f32,
                    qkvabz: &qkvabz,
                    projected_qkv: &projected_qkv,
                    a: &a,
                    b: &b,
                    z: &z,
                    conv_qkv: &conv_qkv,
                    recurrent_output: &recurrent_output,
                    pre_output_hidden_states: &pre_output_hidden_states,
                    pre_output_hidden_states_bf16: &pre_output_hidden_states_bf16,
                },
                batch_metadata: &batch_metadata,
                state: GDNLayerStateBindings {
                    conv_state: &conv_state,
                    conv_state_offset_bytes: 0,
                    next_conv_state: &next_conv_state,
                    next_conv_state_offset_bytes: 0,
                    recurrent_state_arena: &recurrent_state_arena,
                    recurrent_state_arena_offset_bytes: 0,
                },
                materialize_candidate_states: false,
                weights: weights.as_borrowed(),
            },
        );
        let replay = builder.build();
        let fixture = Self {
            device: device.clone(),
            stream,
            num_tokens,
            num_reqs,
            existing_context_len,
            next_hidden_state,
            replay,
            hidden_state,
            hidden_state_f32,
            batch_metadata,
            conv_state,
            next_conv_state,
            recurrent_state_arena,
            qkvabz,
            projected_qkv,
            a,
            b,
            z,
            conv_qkv,
            recurrent_output,
            pre_output_hidden_states,
            pre_output_hidden_states_bf16,
            weights,
        };
        fixture.run();
        fixture
    }

    fn run(&self) {
        MetalReplayRuntime::new(&self.stream).submit_replay(&self.replay).wait();
    }

    fn measure(&self, warmup_iters: usize, iters: usize, runs: usize) {
        let samples = measure_runs(runs, warmup_iters, iters, || self.run());
        let _ = self.next_hidden_state.len_bytes();
        print_perf(
            self.num_tokens,
            self.num_reqs,
            Some(self.existing_context_len),
            Some("ragged_recurrent"),
            iters,
            &samples,
        );
    }

    fn measure_subcomponents(&self, warmup_iters: usize, iters: usize, runs: usize) {
        let device = &self.device;
        let qkvabz_projection = AffineQuantizedMatmulKernel::new(device, gdn_qkvabz_affine_shape(self.num_tokens));
        let projection_split = GDNProjectionSplitKernel::new(device);
        let core = GDNCoreKernels::new(device, gdn_core_config());
        let output_projection = AffineQuantizedMatmulKernel::new(device, gdn_output_affine_shape(self.num_tokens));

        let qkvabz_replay = build_single_invocation_replay(
            &self.stream,
            qkvabz_projection.invoke_with_shape(
                gdn_qkvabz_affine_shape(self.num_tokens),
                &self.qkvabz,
                0,
                &self.hidden_state_f32,
                0,
                &self.weights.qkvabz_weight,
                0,
                &self.weights.qkvabz_scales,
                0,
                &self.weights.qkvabz_biases,
                0,
            ),
        );
        let split_replay = build_single_invocation_replay(
            &self.stream,
            projection_split.invoke(
                GDNProjectionSplitShape::f32(
                    self.num_tokens,
                    GDN_CONV_DIM.try_into().expect("GDN qkv_dim must fit u32"),
                    GDN_V_HEADS.try_into().expect("GDN V heads must fit u32"),
                    GDN_V_DIM.try_into().expect("GDN V dim must fit u32"),
                ),
                GDNProjectionSplitBuffers {
                    qkvabz: &self.qkvabz,
                    projected_qkv: &self.projected_qkv,
                    a: &self.a,
                    b: &self.b,
                    z: &self.z,
                },
            ),
        );
        let core_replay = build_single_invocation_replay(
            &self.stream,
            core.invoke(
                GDNCoreShape {
                    num_reqs: self.num_reqs,
                    num_tokens: self.num_tokens,
                },
                GDNCoreBuffers {
                    projected_qkv: &self.projected_qkv,
                    a: &self.a,
                    b: &self.b,
                    z: &self.z,
                    conv_weight: &self.weights.conv_weight,
                    norm_weight: &self.weights.norm_weight,
                    a_log_decay: &self.weights.a_log_decay,
                    dt_bias: &self.weights.dt_bias,
                    cu_tokens: self.batch_metadata.cu_tokens(),
                    src_state_slots: self.batch_metadata.src_state_slots(),
                    dst_state_slots: self.batch_metadata.dst_state_slots(),
                    conv_state: &self.conv_state,
                    conv_state_offset_bytes: 0,
                    next_conv_state: &self.next_conv_state,
                    next_conv_state_offset_bytes: 0,
                    recurrent_state_arena: &self.recurrent_state_arena,
                    recurrent_state_arena_offset_bytes: 0,
                    conv_qkv: &self.conv_qkv,
                    recurrent_output: &self.recurrent_output,
                    pre_output_hidden_states: &self.pre_output_hidden_states,
                },
                (GDN_QK_HEAD_DIM as f32).sqrt().recip(),
                GDN_EPS,
            ),
        );
        let output_projection_replay = build_single_invocation_replay(
            &self.stream,
            output_projection.invoke_with_shape(
                gdn_output_affine_shape(self.num_tokens),
                &self.next_hidden_state,
                0,
                &self.pre_output_hidden_states,
                0,
                &self.weights.output_weight,
                0,
                &self.weights.output_scales,
                0,
                &self.weights.output_biases,
                0,
            ),
        );

        self.measure_subcomponent("qkvabz-proj", &qkvabz_replay, warmup_iters, iters, runs);
        self.measure_subcomponent("split", &split_replay, warmup_iters, iters, runs);
        self.measure_subcomponent("core", &core_replay, warmup_iters, iters, runs);
        self.measure_subcomponent("output-proj", &output_projection_replay, warmup_iters, iters, runs);
    }

    fn measure_subcomponent(&self, name: &str, replay: &ReplayProgram, warmup_iters: usize, iters: usize, runs: usize) {
        let samples = measure_runs(runs, warmup_iters, iters, || {
            MetalReplayRuntime::new(&self.stream).submit_replay(replay).wait();
        });
        print_named_perf(
            &format!("gdn.{name}"),
            self.num_tokens,
            self.num_reqs,
            Some(self.existing_context_len),
            iters,
            &samples,
        );
    }
}

fn gdn_core_config() -> GDNCoreConfig {
    GDNCoreConfig {
        num_qk_heads: GDN_QK_HEADS.try_into().expect("GDN qk heads must fit u32"),
        qk_head_dim: GDN_QK_HEAD_DIM.try_into().expect("GDN qk head dim must fit u32"),
        num_v_heads: GDN_V_HEADS.try_into().expect("GDN V heads must fit u32"),
        v_head_dim: GDN_V_HEAD_DIM.try_into().expect("GDN V head dim must fit u32"),
        conv_kernel_size: GDN_CONV_KERNEL_SIZE
            .try_into()
            .expect("GDN conv kernel size must fit u32"),
        v_dim_tile_size: 8,
    }
}

struct RealGDNWeights {
    qkvabz_weight: Buffer,
    qkvabz_scales: Buffer,
    qkvabz_biases: Buffer,
    conv_weight: Buffer,
    norm_weight: Buffer,
    a_log_decay: Buffer,
    dt_bias: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl RealGDNWeights {
    fn load(device: &Device, tensors: &SafeTensors<'_>) -> Self {
        let prefix = format!("language_model.model.layers.{GDN_LAYER}.linear_attn");
        let qkv_weight = tensor_bytes(
            tensors,
            &format!("{prefix}.in_proj_qkv.weight"),
            safetensors::Dtype::U32,
        );
        let a_weight = tensor_bytes(tensors, &format!("{prefix}.in_proj_a.weight"), safetensors::Dtype::U32);
        let b_weight = tensor_bytes(tensors, &format!("{prefix}.in_proj_b.weight"), safetensors::Dtype::U32);
        let z_weight = tensor_bytes(tensors, &format!("{prefix}.in_proj_z.weight"), safetensors::Dtype::U32);
        let qkv_scales = bf16_tensor_as_f32(tensors, &format!("{prefix}.in_proj_qkv.scales"));
        let a_scales = bf16_tensor_as_f32(tensors, &format!("{prefix}.in_proj_a.scales"));
        let b_scales = bf16_tensor_as_f32(tensors, &format!("{prefix}.in_proj_b.scales"));
        let z_scales = bf16_tensor_as_f32(tensors, &format!("{prefix}.in_proj_z.scales"));
        let qkv_biases = bf16_tensor_as_f32(tensors, &format!("{prefix}.in_proj_qkv.biases"));
        let a_biases = bf16_tensor_as_f32(tensors, &format!("{prefix}.in_proj_a.biases"));
        let b_biases = bf16_tensor_as_f32(tensors, &format!("{prefix}.in_proj_b.biases"));
        let z_biases = bf16_tensor_as_f32(tensors, &format!("{prefix}.in_proj_z.biases"));
        let qkvabz_weight = concat_parts(&[&qkv_weight, &a_weight, &b_weight, &z_weight]);
        let qkvabz_scales = concat_f32_parts(&[&qkv_scales, &a_scales, &b_scales, &z_scales]);
        let qkvabz_biases = concat_f32_parts(&[&qkv_biases, &a_biases, &b_biases, &z_biases]);
        validate_qkvabz_sizes(&qkvabz_weight, &qkvabz_scales, &qkvabz_biases);
        Self {
            qkvabz_weight: Buffer::from_slice(device, &qkvabz_weight),
            qkvabz_scales: Buffer::from_slice(device, &qkvabz_scales),
            qkvabz_biases: Buffer::from_slice(device, &qkvabz_biases),
            conv_weight: Buffer::from_slice(device, &bf16_tensor_as_f32(tensors, &format!("{prefix}.conv1d.weight"))),
            norm_weight: Buffer::from_slice(device, &bf16_tensor_as_f32(tensors, &format!("{prefix}.norm.weight"))),
            a_log_decay: Buffer::from_slice(device, &a_log_decay(tensors, &format!("{prefix}.A_log"))),
            dt_bias: Buffer::from_slice(device, &bf16_tensor_as_f32(tensors, &format!("{prefix}.dt_bias"))),
            output_weight: Buffer::from_slice(
                device,
                &tensor_bytes(tensors, &format!("{prefix}.out_proj.weight"), safetensors::Dtype::U32),
            ),
            output_scales: Buffer::from_slice(
                device,
                &tensor_bytes(tensors, &format!("{prefix}.out_proj.scales"), safetensors::Dtype::BF16),
            ),
            output_biases: Buffer::from_slice(
                device,
                &tensor_bytes(tensors, &format!("{prefix}.out_proj.biases"), safetensors::Dtype::BF16),
            ),
        }
    }

    fn as_borrowed(&self) -> GDNWeights<'_> {
        GDNWeights {
            qkvabz_weight: &self.qkvabz_weight,
            qkvabz_scales: &self.qkvabz_scales,
            qkvabz_biases: &self.qkvabz_biases,
            conv_weight: &self.conv_weight,
            norm_weight: &self.norm_weight,
            a_log_decay: &self.a_log_decay,
            dt_bias: &self.dt_bias,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}

struct MappedFile {
    ptr: *mut libc::c_void,
    len: usize,
}

impl MappedFile {
    fn open(path: &Path) -> Self {
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

    fn as_bytes(&self) -> &[u8] {
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
