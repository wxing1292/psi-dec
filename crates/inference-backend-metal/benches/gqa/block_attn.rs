use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::gqa::block_sdpa as backend_block_sdpa;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;

fn main() {
    let args = Args::parse();
    let device = Device::system_default();
    for &dtype in &args.dtypes {
        for &block_size in &args.block_sizes {
            for &num_requests in &args.num_requests {
                let setup_start = Instant::now();
                let fixture = Fixture::new(
                    &device,
                    block_size,
                    num_requests,
                    args.num_q_heads,
                    args.num_kv_heads,
                    args.head_dim,
                    dtype,
                );
                let setup_elapsed = setup_start.elapsed();
                let cache_miss_start = Instant::now();
                fixture.run();
                let cache_miss_elapsed = cache_miss_start.elapsed();
                fixture.warmup(args.warmup_iters);
                let samples = fixture.measure(args.iters, args.runs);
                print_result(&fixture, setup_elapsed, cache_miss_elapsed, args.iters, &samples);
            }
        }
    }
}

struct Args {
    block_sizes: Vec<u32>,
    num_requests: Vec<u32>,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    dtypes: Vec<Dtype>,
    warmup_iters: usize,
    iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            block_sizes: vec![7],
            num_requests: vec![1, 4],
            num_q_heads: 40,
            num_kv_heads: 8,
            head_dim: 128,
            dtypes: vec![Dtype::Bfloat16],
            warmup_iters: 20,
            iters: 100,
            runs: 5,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--block-sizes" => args.block_sizes = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--num-requests" => args.num_requests = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--num-q-heads" => args.num_q_heads = parse_u32(&next_arg(&mut values, &arg), &arg),
                "--num-kv-heads" => args.num_kv_heads = parse_u32(&next_arg(&mut values, &arg), &arg),
                "--head-dim" => args.head_dim = parse_u32(&next_arg(&mut values, &arg), &arg),
                "--dtypes" => args.dtypes = parse_dtypes(&next_arg(&mut values, &arg)),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.block_sizes.is_empty(), "--block-sizes must not be empty");
        assert!(args.block_sizes.iter().all(|&value| value > 0));
        assert!(!args.num_requests.is_empty(), "--num-requests must not be empty");
        assert!(args.num_requests.iter().all(|&value| value > 0));
        assert!(!args.dtypes.is_empty(), "--dtypes must not be empty");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args
    }
}

struct Fixture {
    stream: Stream,
    replay: ReplayProgram,
    _q: Buffer,
    _local_k: Buffer,
    _local_v: Buffer,
    _block_sdpa_map_task_template_indices: Buffer,
    _partial_exp_sums: Buffer,
    _partial_max_logits: Buffer,
    partial_output: Buffer,
    block_size: u32,
    num_requests: u32,
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    dtype: Dtype,
}

impl Fixture {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &Device,
        block_size: u32,
        num_requests: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        dtype: Dtype,
    ) -> Self {
        let num_tokens = num_requests
            .checked_mul(block_size)
            .expect("GQA block-attention bench token count must fit u32");
        let num_total_sdpa_map_task_templates = num_tokens
            .checked_mul(2)
            .and_then(u32::checked_next_power_of_two)
            .expect("GQA block-attention bench partial-output capacity must fit u32");
        let config = backend_block_sdpa::Config {
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            dtype,
        };
        let shape = backend_block_sdpa::Shape {
            num_tokens,
            num_total_sdpa_map_task_templates,
        };
        shape.validate(config);

        let q_elements = checked_product(&[num_tokens, num_q_heads, head_dim]);
        let kv_elements = checked_product(&[num_tokens, num_kv_heads, head_dim]);
        let partial_stat_elements = checked_product(&[num_total_sdpa_map_task_templates, num_q_heads]);
        let partial_output_elements = partial_stat_elements
            .checked_mul(head_dim as usize)
            .expect("GQA block-attention bench partial-output elements must fit usize");
        let q = pattern_buffer(device, q_elements, dtype, 0.003);
        let local_k = pattern_buffer(device, kv_elements, dtype, -0.002);
        let local_v = pattern_buffer(device, kv_elements, dtype, 0.004);
        let block_sdpa_map_task_template_indices = Buffer::from_slice(
            device,
            &(0..num_tokens)
                .map(|token_index| {
                    token_index
                        .checked_mul(2)
                        .and_then(|value| value.checked_add(1))
                        .expect("GQA block-attention bench TaskTemplate index must fit u32")
                })
                .collect::<Vec<_>>(),
        );
        let partial_exp_sums = Buffer::new_zeroed_elements(device, partial_stat_elements, Dtype::Float32);
        let partial_max_logits = Buffer::new_zeroed_elements(device, partial_stat_elements, Dtype::Float32);
        let partial_output = Buffer::new_zeroed_elements(device, partial_output_elements, dtype);
        let kernel = backend_block_sdpa::Compute::new(device, config);
        let stream = Stream::new(device);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            backend_block_sdpa::Buffers {
                q: &q,
                local_k: &local_k,
                local_v: &local_v,
                block_sdpa_map_task_template_indices: &block_sdpa_map_task_template_indices,
                partial_exp_sums: &partial_exp_sums,
                partial_max_logits: &partial_max_logits,
                partial_output: &partial_output,
            },
        ));
        let replay = builder.build();
        Self {
            stream,
            replay,
            _q: q,
            _local_k: local_k,
            _local_v: local_v,
            _block_sdpa_map_task_template_indices: block_sdpa_map_task_template_indices,
            _partial_exp_sums: partial_exp_sums,
            _partial_max_logits: partial_max_logits,
            partial_output,
            block_size,
            num_requests,
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype,
        }
    }

    fn run(&self) {
        self.stream.submit_replay(&self.replay).wait();
        black_box(&self.partial_output);
    }

    fn warmup(&self, warmup_iters: usize) {
        for _ in 0..warmup_iters {
            self.run();
        }
    }

    fn measure(&self, iters: usize, runs: usize) -> Vec<Duration> {
        (0..runs)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..iters {
                    self.run();
                }
                start.elapsed()
            })
            .collect()
    }
}

fn print_result(
    fixture: &Fixture,
    setup_elapsed: Duration,
    cache_miss_elapsed: Duration,
    iters: usize,
    samples: &[Duration],
) {
    let mut per_iteration_us = samples
        .iter()
        .map(|sample| sample.as_secs_f64() * 1.0e6 / iters as f64)
        .collect::<Vec<_>>();
    per_iteration_us.sort_by(f64::total_cmp);
    println!(
        "perf component=gqa-block-sdpa backend=metal operation=block-bidi-map dtype={} num_requests={} block_size={} \
         num_tokens={} num_q_heads={} num_kv_heads={} head_dim={} setup_us={:.3} cache_miss_us={:.3} iters={} runs={} \
         median_us={:.3} samples_us={:?}",
        dtype_name(fixture.dtype),
        fixture.num_requests,
        fixture.block_size,
        fixture.num_tokens,
        fixture.num_q_heads,
        fixture.num_kv_heads,
        fixture.head_dim,
        setup_elapsed.as_secs_f64() * 1.0e6,
        cache_miss_elapsed.as_secs_f64() * 1.0e6,
        iters,
        per_iteration_us.len(),
        median_of_sorted(&per_iteration_us),
        per_iteration_us,
    );
}

fn pattern_buffer(device: &Device, num_elements: usize, dtype: Dtype, scale: f32) -> Buffer {
    let values = (0..num_elements)
        .map(|index| ((index % 251) as f32 - 125.0) * scale)
        .collect::<Vec<_>>();
    match dtype {
        Dtype::Float32 => Buffer::from_slice(device, &values),
        Dtype::Bfloat16 => {
            Buffer::from_slice(
                device,
                &values
                    .into_iter()
                    .map(|value| bf16::from_f32(value).to_bits())
                    .collect::<Vec<_>>(),
            )
        },
        dtype => panic!("unsupported GQA block-attention bench dtype {dtype:?}"),
    }
}

fn checked_product(values: &[u32]) -> usize {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value as usize))
        .expect("GQA block-attention bench element count must fit usize")
}

fn median_of_sorted(samples: &[f64]) -> f64 {
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    }
}

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Float32 => "f32",
        Dtype::Bfloat16 => "bf16",
        dtype => panic!("unsupported GQA block-attention bench dtype {dtype:?}"),
    }
}

fn parse_dtypes(value: &str) -> Vec<Dtype> {
    value
        .split(',')
        .map(|part| {
            match part.trim() {
                "f32" => Dtype::Float32,
                "bf16" => Dtype::Bfloat16,
                other => panic!("unknown GQA block-attention bench dtype {other:?}"),
            }
        })
        .collect()
}

fn parse_u32_list(value: &str, flag: &str) -> Vec<u32> {
    value.split(',').map(|part| parse_u32(part.trim(), flag)).collect()
}

fn parse_u32(value: &str, flag: &str) -> u32 {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires u32 values"))
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a usize"))
}

fn next_arg(values: &mut impl Iterator<Item = String>, flag: &str) -> String {
    values.next().unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn print_help_and_exit() -> ! {
    let executable = PathBuf::from(std::env::args().next().unwrap_or_else(|| "gqa_block_attn".to_string()));
    println!(
        "{}\n--block-sizes 7\n--num-requests 1,4\n--num-q-heads 40\n--num-kv-heads 8\n--head-dim 128\n--dtypes \
         bf16,f32\n--warmup-iters N\n--iters N\n--runs N",
        executable.display()
    );
    std::process::exit(0);
}
