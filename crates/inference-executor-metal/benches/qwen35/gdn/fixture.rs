use std::fs::File;
use std::mem::size_of;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use inference_backend_metal::components::gdn::compute as backend_compute;
use inference_backend_metal::components::gdn::qkvabz_split as backend_qkvabz_split;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::GDNCore;
use inference_executor_metal::attn::gdn::backend::GDN;
use inference_executor_metal::attn::gdn::backend::GDNInput;
use inference_executor_metal::attn::gdn::backend::GDNLayerStateBindings;
use inference_executor_metal::attn::gdn::backend::GDNMetalConfig;
use inference_executor_metal::attn::gdn::backend::GDNWeights;
use inference_executor_metal::attn::gdn::batch_metadata::GDNMetadataBuffers;
use inference_executor_metal::attn::gdn::scratch::GDNScratchBindings;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use safetensors::SafeTensors;

use crate::Args;
use crate::BITS;
use crate::GDN_CONV_DIM;
use crate::GDN_CONV_KERNEL_SIZE;
use crate::GDN_EPS;
use crate::GDN_LAYER;
use crate::GDN_QK_HEAD_DIM;
use crate::GDN_QK_HEADS;
use crate::GDN_QKVABZ_DIM;
use crate::GDN_SHARD;
use crate::GDN_V_DIM;
use crate::GDN_V_HEAD_DIM;
use crate::GDN_V_HEADS;
use crate::GROUP_SIZE;
use crate::HIDDEN_DIM;
use crate::build_single_invocation_replay;
use crate::concat_parts;
use crate::cu_tokens;
use crate::gdn_conv_state_fixture;
use crate::gdn_output_affine_config;
use crate::gdn_qkvabz_affine_config;
use crate::gdn_recurrent_state_fixture;
use crate::hidden_fixture;
use crate::measure_runs;
use crate::print_named_perf;
use crate::print_perf;
use crate::print_skip;
use crate::request_token_counts;
use crate::tensor_bytes;
use crate::valid_num_reqs;
use crate::validate_qkvabz_sizes;

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
                let fixture = RealGDNFixture::new(
                    &device,
                    num_tokens,
                    num_reqs,
                    existing_context_len,
                    args.candidate_states,
                    &weights,
                );
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
    materialize_candidate_states: bool,
    next_hidden_state: Buffer,
    replay: ReplayProgram,
    hidden_state: Buffer,
    batch_metadata: GDNMetadataBuffers,
    conv_state: Buffer,
    next_conv_state: Buffer,
    recurrent_state_arena: Buffer,
    qkvabz: Buffer,
    qkv: Buffer,
    a: Buffer,
    b: Buffer,
    z: Buffer,
    conv_qkv: Buffer,
    recurrent_output: Buffer,
    norm_gated_output: Buffer,
    weights: &'a RealGDNWeights,
}

impl<'a> RealGDNFixture<'a> {
    fn new(
        device: &Device,
        num_tokens: u32,
        num_reqs: u32,
        existing_context_len: u32,
        materialize_candidate_states: bool,
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
            norm_eps: GDN_EPS,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            qkvabz_scale_bias_dtype: Dtype::Bfloat16,
            output_scale_bias_dtype: Dtype::Bfloat16,
        };
        let backend = GDN::new(device, core, config);
        let hidden_state = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM));
        let next_hidden_state =
            Buffer::new_zeroed(device, num_tokens as usize * HIDDEN_DIM * Dtype::Bfloat16.item_size());
        let num_tokens_per_req = request_token_counts(num_tokens, num_reqs);
        let batch_metadata = GDNMetadataBuffers::new(device, num_reqs as usize, num_tokens as usize);
        let cu_tokens = cu_tokens(&num_tokens_per_req)
            .into_iter()
            .map(|value| value as u32)
            .collect::<Vec<_>>();
        let mut flat_materialized_state_slots = vec![u32::MAX; num_tokens as usize];
        if materialize_candidate_states {
            for (flat_token_index, state_slot) in flat_materialized_state_slots.iter_mut().enumerate() {
                *state_slot = num_reqs
                    .checked_add(u32::try_from(flat_token_index).expect("GDN bench flat token index must fit u32"))
                    .expect("GDN bench candidate state slot ID must fit u32");
            }
        } else {
            for (req_index, &flat_end) in cu_tokens.iter().skip(1).enumerate() {
                flat_materialized_state_slots[flat_end as usize - 1] = num_reqs
                    .checked_add(u32::try_from(req_index).expect("GDN bench request index must fit u32"))
                    .expect("GDN bench final state slot ID must fit u32");
            }
        }
        batch_metadata.update(
            &cu_tokens,
            &(0..num_reqs).collect::<Vec<_>>(),
            &(0..num_reqs).collect::<Vec<_>>(),
            &flat_materialized_state_slots,
            &flat_materialized_state_slots,
            num_reqs,
            num_tokens,
        );
        let num_state_slots = if materialize_candidate_states {
            num_reqs
                .checked_add(num_tokens)
                .expect("GDN bench candidate state-slot count must fit u32")
        } else {
            num_reqs
                .checked_mul(2)
                .expect("GDN bench source and destination state-slot count must fit u32")
        };
        let conv_state = Buffer::from_slice(
            device,
            &gdn_conv_state_fixture(
                existing_context_len,
                num_reqs as usize,
                num_state_slots as usize * GDN_CONV_DIM * (GDN_CONV_KERNEL_SIZE - 1),
            ),
        );
        let next_conv_state = Buffer::new_zeroed(
            device,
            num_state_slots as usize * GDN_CONV_DIM * (GDN_CONV_KERNEL_SIZE - 1) * size_of::<u16>(),
        );
        let recurrent_state_arena = Buffer::from_slice(
            device,
            &gdn_recurrent_state_fixture(
                existing_context_len,
                num_reqs as usize,
                num_state_slots as usize * GDN_V_HEADS * GDN_V_HEAD_DIM * GDN_QK_HEAD_DIM,
            ),
        );
        let qkvabz = Buffer::new_zeroed(device, num_tokens as usize * GDN_QKVABZ_DIM * size_of::<u16>());
        let qkv = Buffer::new_zeroed(device, num_tokens as usize * GDN_CONV_DIM * size_of::<u16>());
        let a = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_HEADS * size_of::<u16>());
        let b = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_HEADS * size_of::<u16>());
        let z = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_DIM * size_of::<u16>());
        let conv_qkv = Buffer::new_zeroed(device, num_tokens as usize * GDN_CONV_DIM * size_of::<u16>());
        let recurrent_output = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_DIM * size_of::<u16>());
        let norm_gated_output = Buffer::new_zeroed(device, num_tokens as usize * GDN_V_DIM * size_of::<u16>());
        let mut recorder = MetalReplayRuntime::new(&stream).create_recorder();
        let _ = <GDN as ReplayLayer>::record(
            &backend,
            &mut recorder,
            GDNInput {
                hidden_state: &hidden_state,
                next_hidden_state: &next_hidden_state,
                scratch: GDNScratchBindings {
                    qkvabz: &qkvabz,
                    qkv: &qkv,
                    a: &a,
                    b: &b,
                    z: &z,
                    conv_qkv: &conv_qkv,
                    recurrent_output: &recurrent_output,
                    norm_gated_output: &norm_gated_output,
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
                materialize_candidate_states,
                weights: weights.as_borrowed(),
                num_active_tokens: ReplayU32::Fixed(num_tokens),
            },
        );
        let replay = recorder.build();
        let fixture = Self {
            device: device.clone(),
            stream,
            num_tokens,
            num_reqs,
            existing_context_len,
            materialize_candidate_states,
            next_hidden_state,
            replay,
            hidden_state,
            batch_metadata,
            conv_state,
            next_conv_state,
            recurrent_state_arena,
            qkvabz,
            qkv,
            a,
            b,
            z,
            conv_qkv,
            recurrent_output,
            norm_gated_output,
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
            Some(if self.materialize_candidate_states {
                "forward_candidate_state"
            } else {
                "ragged_recurrent"
            }),
            iters,
            &samples,
        );
    }

    fn measure_subcomponents(&self, warmup_iters: usize, iters: usize, runs: usize) {
        let device = &self.device;
        let qkvabz_config = gdn_qkvabz_affine_config();
        let qkvabz = affine_quantized::Matmul::new(device, qkvabz_config);
        let qkvabz_to_qkv_a_b_z = backend_qkvabz_split::Compute::new(
            device,
            backend_qkvabz_split::Config::new(
                GDN_CONV_DIM.try_into().expect("GDN qkv_dim must fit u32"),
                GDN_V_HEADS.try_into().expect("GDN V heads must fit u32"),
                GDN_V_DIM.try_into().expect("GDN V dim must fit u32"),
            ),
        );
        let compute = backend_compute::Compute::new(device, gdn_compute_config());
        let output_config = gdn_output_affine_config();
        let output = affine_quantized::Matmul::new(device, output_config);

        let qkvabz_replay = build_single_invocation_replay(
            &self.stream,
            qkvabz.invoke(
                self.num_tokens,
                ReplayU32::Fixed(self.num_tokens),
                &self.qkvabz,
                0,
                &self.hidden_state,
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
            qkvabz_to_qkv_a_b_z.invoke(
                backend_qkvabz_split::Shape {
                    num_total_tokens: self.num_tokens,
                },
                backend_qkvabz_split::Buffers {
                    qkvabz: &self.qkvabz,
                    qkv: &self.qkv,
                    a: &self.a,
                    b: &self.b,
                    z: &self.z,
                },
                ReplayU32::Fixed(self.num_tokens),
            ),
        );
        let compute_shape = backend_compute::Shape {
            num_total_reqs: self.num_reqs,
            num_total_tokens: self.num_tokens,
        };
        let compute_buffers = backend_compute::Buffers {
            qkv: &self.qkv,
            a: &self.a,
            b: &self.b,
            z: &self.z,
            conv_weight: &self.weights.conv_weight,
            norm_weight: &self.weights.norm_weight,
            a_log: &self.weights.a_log,
            dt_bias: &self.weights.dt_bias,
            cu_tokens: self.batch_metadata.cu_tokens(),
            src_recurrent_state_slots: self.batch_metadata.src_recurrent_state_slots(),
            src_conv_state_slots: self.batch_metadata.src_conv_state_slots(),
            flat_recurrent_state_write_slots: self.batch_metadata.flat_recurrent_state_write_slots(),
            flat_conv_state_write_slots: self.batch_metadata.flat_conv_state_write_slots(),
            conv_state: &self.conv_state,
            conv_state_offset_bytes: 0,
            next_conv_state: &self.next_conv_state,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &self.recurrent_state_arena,
            recurrent_state_arena_offset_bytes: 0,
            conv_qkv: &self.conv_qkv,
            recurrent_output: &self.recurrent_output,
            norm_gated_output: &self.norm_gated_output,
        };
        let compute_replay = if self.materialize_candidate_states {
            build_single_invocation_replay(
                &self.stream,
                compute.invoke_with_candidate_state_update(
                    compute_shape,
                    compute_buffers,
                    ReplayU32::Fixed(self.num_reqs),
                    ReplayU32::Fixed(self.num_tokens),
                ),
            )
        } else {
            build_single_invocation_replay(
                &self.stream,
                compute.invoke(
                    compute_shape,
                    compute_buffers,
                    ReplayU32::Fixed(self.num_reqs),
                    ReplayU32::Fixed(self.num_tokens),
                ),
            )
        };
        let output_replay = build_single_invocation_replay(
            &self.stream,
            output.invoke(
                self.num_tokens,
                ReplayU32::Fixed(self.num_tokens),
                &self.next_hidden_state,
                0,
                &self.norm_gated_output,
                0,
                &self.weights.output_weight,
                0,
                &self.weights.output_scales,
                0,
                &self.weights.output_biases,
                0,
            ),
        );

        self.measure_subcomponent("qkvabz", &qkvabz_replay, warmup_iters, iters, runs);
        self.measure_subcomponent("qkvabz-to-qkv-a-b-z", &split_replay, warmup_iters, iters, runs);
        self.measure_subcomponent(
            if self.materialize_candidate_states {
                "compute_candidate_state"
            } else {
                "compute"
            },
            &compute_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent("output", &output_replay, warmup_iters, iters, runs);
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

fn gdn_compute_config() -> backend_compute::Config {
    backend_compute::Config {
        num_qk_heads: GDN_QK_HEADS.try_into().expect("GDN qk heads must fit u32"),
        qk_head_dim: GDN_QK_HEAD_DIM.try_into().expect("GDN qk head dim must fit u32"),
        num_v_heads: GDN_V_HEADS.try_into().expect("GDN V heads must fit u32"),
        v_head_dim: GDN_V_HEAD_DIM.try_into().expect("GDN V head dim must fit u32"),
        conv_kernel_size: GDN_CONV_KERNEL_SIZE
            .try_into()
            .expect("GDN conv kernel size must fit u32"),
        q_scale: (GDN_QK_HEAD_DIM as f32).sqrt().recip(),
        norm_eps: GDN_EPS,
    }
}

struct RealGDNWeights {
    qkvabz_weight: Buffer,
    qkvabz_scales: Buffer,
    qkvabz_biases: Buffer,
    conv_weight: Buffer,
    norm_weight: Buffer,
    a_log: Buffer,
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
        let qkv_scales = tensor_bytes(
            tensors,
            &format!("{prefix}.in_proj_qkv.scales"),
            safetensors::Dtype::BF16,
        );
        let a_scales = tensor_bytes(tensors, &format!("{prefix}.in_proj_a.scales"), safetensors::Dtype::BF16);
        let b_scales = tensor_bytes(tensors, &format!("{prefix}.in_proj_b.scales"), safetensors::Dtype::BF16);
        let z_scales = tensor_bytes(tensors, &format!("{prefix}.in_proj_z.scales"), safetensors::Dtype::BF16);
        let qkv_biases = tensor_bytes(
            tensors,
            &format!("{prefix}.in_proj_qkv.biases"),
            safetensors::Dtype::BF16,
        );
        let a_biases = tensor_bytes(tensors, &format!("{prefix}.in_proj_a.biases"), safetensors::Dtype::BF16);
        let b_biases = tensor_bytes(tensors, &format!("{prefix}.in_proj_b.biases"), safetensors::Dtype::BF16);
        let z_biases = tensor_bytes(tensors, &format!("{prefix}.in_proj_z.biases"), safetensors::Dtype::BF16);
        let qkvabz_weight = concat_parts(&[&qkv_weight, &a_weight, &b_weight, &z_weight]);
        let qkvabz_scales = concat_parts(&[&qkv_scales, &a_scales, &b_scales, &z_scales]);
        let qkvabz_biases = concat_parts(&[&qkv_biases, &a_biases, &b_biases, &z_biases]);
        validate_qkvabz_sizes(&qkvabz_weight, &qkvabz_scales, &qkvabz_biases);
        Self {
            qkvabz_weight: Buffer::from_slice(device, &qkvabz_weight),
            qkvabz_scales: Buffer::from_slice(device, &qkvabz_scales),
            qkvabz_biases: Buffer::from_slice(device, &qkvabz_biases),
            conv_weight: Buffer::from_slice(
                device,
                &tensor_bytes(tensors, &format!("{prefix}.conv1d.weight"), safetensors::Dtype::BF16),
            ),
            norm_weight: Buffer::from_slice(
                device,
                &tensor_bytes(tensors, &format!("{prefix}.norm.weight"), safetensors::Dtype::BF16),
            ),
            a_log: Buffer::from_slice(
                device,
                &tensor_bytes(tensors, &format!("{prefix}.A_log"), safetensors::Dtype::BF16),
            ),
            dt_bias: Buffer::from_slice(
                device,
                &tensor_bytes(tensors, &format!("{prefix}.dt_bias"), safetensors::Dtype::BF16),
            ),
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
            a_log: &self.a_log,
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
