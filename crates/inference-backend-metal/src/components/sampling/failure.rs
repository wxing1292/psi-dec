use std::rc::Rc;

use objc2_metal::MTLComputePipelineState;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::stream::check::SubmissionCheck;

// Matches sampling_failure in sampling.metal. Only the status word is read on
// success. All diagnostic fields are captured by the winning GPU thread.
const FAILURE_WORDS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub enum SamplingOperation {
    Merge,
    Sample,
    WriteDistribution,
    SampleAndWriteDistribution,
    Rejection,
}

#[derive(Debug)]
pub struct SamplingFailure {
    buffer: Buffer,
    operation: SamplingOperation,
    shape: String,
}

impl SamplingFailure {
    pub fn record(
        recorder: &CommandRecorder<'_>,
        kernel: &CompiledKernel,
        operation: SamplingOperation,
        shape: String,
    ) {
        let device = Device::from_raw_retained(kernel.as_raw().device());
        let check = Rc::new(Self {
            buffer: Buffer::from_slice(&device, &[0_u32; FAILURE_WORDS]),
            operation,
            shape,
        });
        recorder.set_buffer_read_write(30, &check.buffer, 0);
        recorder.set_submission_check(check);
    }
}

impl SubmissionCheck for SamplingFailure {
    fn reset(&self) {
        self.buffer.write_typed(0, &[0_u32]);
    }

    fn assert_success(&self) {
        let mut code = [0_u8; 4];
        self.buffer.read_bytes(0, &mut code);
        if u32::from_ne_bytes(code) == 0 {
            return;
        }
        let mut bytes = [0_u8; FAILURE_WORDS * 4];
        self.buffer.read_bytes(0, &mut bytes);
        let words =
            std::array::from_fn(|index| u32::from_ne_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap()));
        panic!("{}", describe_failure(self.operation, &self.shape, words));
    }
}

fn describe_failure(operation: SamplingOperation, shape: &str, words: [u32; FAILURE_WORDS]) -> String {
    let [
        code,
        row,
        req_slot,
        sample_position,
        domain,
        distribution,
        seed,
        top_k,
        temperature,
        top_p,
        value,
        token,
        target_start,
        target_end,
        draft_start,
        draft_end,
    ] = words;
    let reason = match code {
        1 => "no finite logit candidates",
        2 => "invalid probability normalization",
        3 => "invalid sampling parameters or distribution ranges",
        4 => "invalid rejection probability mass",
        _ => "unknown sampling failure",
    };
    let domain = match domain {
        0x243f_6a88 => "Target (Main/bonus)",
        0x85a3_08d3 => "Draft (Spec)",
        0x1319_8a2e => "Accept",
        0x0370_7344 => "Resample",
        _ => "not applicable",
    };
    let optional = |value: u32| {
        if value == u32::MAX {
            "unavailable".to_owned()
        } else {
            value.to_string()
        }
    };
    let parameters = match operation {
        SamplingOperation::Merge => format!("top_k={top_k}"),
        SamplingOperation::Rejection => format!("top_k={top_k} seed={seed}"),
        _ => {
            format!(
                "top_k={top_k} temperature={} top_p={} seed={seed}",
                f32::from_bits(temperature),
                f32::from_bits(top_p),
            )
        },
    };
    let ranges = match operation {
        SamplingOperation::Rejection => {
            format!("\ntarget_range=[{target_start}, {target_end}) draft_range=[{draft_start}, {draft_end})")
        },
        _ => String::new(),
    };
    let operation = match operation {
        SamplingOperation::Merge => "top-k merge",
        SamplingOperation::Sample => "top-k sampling",
        SamplingOperation::WriteDistribution => "top-k write distribution",
        SamplingOperation::SampleAndWriteDistribution => "top-k sample and write distribution",
        SamplingOperation::Rejection => "sparse rejection sampling",
    };
    format!(
        "Metal {operation} failed: {reason}; row={row} code={code}\nreq_slot={} sample_position={} domain={domain} \
         distribution_index={}\n{parameters}; {shape}\ndiagnostic_value={} bits=0x{value:08x} \
         candidate_token={}{ranges}",
        optional(req_slot),
        optional(sample_position),
        optional(distribution),
        f32::from_bits(value),
        token as i32,
    )
}
