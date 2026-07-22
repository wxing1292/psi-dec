use std::collections::BTreeMap;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::BufWriter;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use half::bf16;
use inference_executor_core::model::qwen::v3_5::DSparkConfig;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkWeightBindings;
use safetensors::Dtype;
use safetensors::tensor::Metadata;
use safetensors::tensor::TensorInfo;

const SAFETENSORS_HEADER_LEN_BYTES: usize = 8;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 100_000_000;

#[derive(Debug)]
struct QuantizeError(String);

impl fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for QuantizeError {}

type Result<T> = std::result::Result<T, QuantizeError>;

#[derive(Debug)]
struct Args {
    input_dir: PathBuf,
    output_dir: PathBuf,
    group_size: usize,
    bits: usize,
    markov_w2_bits: usize,
}

#[derive(Clone, Copy, Debug)]
struct QuantizeOptions {
    group_size: usize,
    bits: usize,
    markov_w2_bits: usize,
}

#[derive(Clone, Debug)]
struct SafetensorsHeader {
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

#[derive(Clone, Debug)]
struct OutputTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    offset: usize,
    len_bytes: usize,
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("{}", usage());
        return Ok(());
    }
    let args = Args::parse(arguments)?;
    quantize_checkpoint(
        &args.input_dir,
        &args.output_dir,
        QuantizeOptions {
            group_size: args.group_size,
            bits: args.bits,
            markov_w2_bits: args.markov_w2_bits,
        },
    )?;
    Ok(())
}

impl Args {
    fn parse(args: impl IntoIterator<Item = std::ffi::OsString>) -> Result<Self> {
        let mut input_dir = None;
        let mut output_dir = None;
        let mut group_size = 64;
        let mut bits = 4;
        let mut markov_w2_bits = 8;
        let mut args = args.into_iter();
        while let Some(argument) = args.next() {
            let name = argument
                .to_str()
                .ok_or_else(|| error(format!("argument {argument:?} is not valid UTF-8")))?;
            let value = args
                .next()
                .ok_or_else(|| error(format!("missing value for {name}\n{}", usage())))?;
            match name {
                "--input-dir" => input_dir = Some(PathBuf::from(value)),
                "--output-dir" => output_dir = Some(PathBuf::from(value)),
                "--group-size" => group_size = parse_usize(name, &value)?,
                "--bits" => bits = parse_usize(name, &value)?,
                "--markov-w2-bits" => markov_w2_bits = parse_usize(name, &value)?,
                _ => return Err(error(format!("unknown argument {name:?}\n{}", usage()))),
            }
        }
        Ok(Self {
            input_dir: input_dir.ok_or_else(|| error(format!("missing --input-dir\n{}", usage())))?,
            output_dir: output_dir.ok_or_else(|| error(format!("missing --output-dir\n{}", usage())))?,
            group_size,
            bits,
            markov_w2_bits,
        })
    }
}

fn parse_usize(name: &str, value: &std::ffi::OsStr) -> Result<usize> {
    value
        .to_str()
        .ok_or_else(|| error(format!("value for {name} is not valid UTF-8")))?
        .parse::<usize>()
        .map_err(|err| error(format!("invalid integer for {name}: {err}")))
}

fn usage() -> &'static str {
    "usage: qwen35_dspark_quantize --input-dir DIR --output-dir DIR [--group-size 64] [--bits 4] [--markov-w2-bits 8]"
}

fn quantize_checkpoint(input_dir: &Path, output_dir: &Path, options: QuantizeOptions) -> Result<()> {
    validate_options(options)?;
    if output_dir.exists() {
        return Err(error(format!("output directory {output_dir:?} already exists")));
    }
    let input_config_path = input_dir.join("config.json");
    let input_config_bytes = std::fs::read(&input_config_path)
        .map_err(|err| error(format!("unable to read DSpark config {input_config_path:?}: {err}")))?;
    let mut config = serde_json::from_slice::<DSparkConfig>(&input_config_bytes)
        .map_err(|err| error(format!("unable to parse DSpark config {input_config_path:?}: {err}")))?;
    config
        .normalize_and_validate()
        .map_err(|err| error(format!("invalid DSpark config {input_config_path:?}: {err}")))?;
    let mut config_value = serde_json::from_slice::<serde_json::Value>(&input_config_bytes)
        .map_err(|err| error(format!("unable to preserve DSpark config {input_config_path:?}: {err}")))?;

    let input_checkpoint = input_dir.join("model.safetensors");
    if !input_checkpoint.is_file() {
        return Err(error(format!(
            "DSpark converter currently requires the linked single-file checkpoint at {input_checkpoint:?}"
        )));
    }
    let temp_dir = temporary_output_path(output_dir)?;
    std::fs::create_dir(&temp_dir).map_err(|err| {
        error(format!(
            "unable to create temporary output directory {temp_dir:?}: {err}"
        ))
    })?;
    let mut cleanup = TempDirCleanup {
        path: temp_dir.clone(),
        committed: false,
    };

    quantize_safetensors(&input_checkpoint, &temp_dir.join("model.safetensors"), &config, options)?;
    write_output_config(&mut config_value, &temp_dir.join("config.json"), options)?;
    std::fs::rename(&temp_dir, output_dir).map_err(|err| {
        error(format!(
            "unable to publish quantized DSpark checkpoint {temp_dir:?} as {output_dir:?}: {err}"
        ))
    })?;
    cleanup.committed = true;
    Ok(())
}

struct TempDirCleanup {
    path: PathBuf,
    committed: bool,
}

impl Drop for TempDirCleanup {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

fn temporary_output_path(output_dir: &Path) -> Result<PathBuf> {
    let parent = output_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(error(format!("output parent directory {parent:?} does not exist")));
    }
    let name = output_dir
        .file_name()
        .ok_or_else(|| error(format!("output directory {output_dir:?} has no file name")))?
        .to_string_lossy();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| error(format!("system clock is before UNIX epoch: {err}")))?
        .as_nanos();
    Ok(parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id())))
}

fn validate_options(options: QuantizeOptions) -> Result<()> {
    if !matches!(options.group_size, 32 | 64 | 128) {
        return Err(error(format!(
            "unsupported affine group_size={}; expected 32, 64, or 128",
            options.group_size
        )));
    }
    for (name, bits) in [("bits", options.bits), ("markov_w2_bits", options.markov_w2_bits)] {
        if !matches!(bits, 2 | 3 | 4 | 6 | 8) {
            return Err(error(format!("unsupported {name}={bits}; expected 2, 3, 4, 6, or 8")));
        }
    }
    Ok(())
}

fn quantize_safetensors(
    input_path: &Path,
    output_path: &Path,
    config: &DSparkConfig,
    options: QuantizeOptions,
) -> Result<()> {
    let mut input = File::open(input_path)
        .map_err(|err| error(format!("unable to open BF16 DSpark checkpoint {input_path:?}: {err}")))?;
    let header = read_header(&mut input, input_path)?;
    validate_source_tensors(&header, config, input_path)?;
    let output_tensors = build_output_tensors(&header, options)?;
    validate_output_tensor_names(&output_tensors, config)?;
    let output_by_name = output_tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<HashMap<_, _>>();
    let metadata = HashMap::from([
        ("format".to_string(), "psi-dec-dspark-affine".to_string()),
        ("group_size".to_string(), options.group_size.to_string()),
        ("bits".to_string(), options.bits.to_string()),
        ("markov_w2_bits".to_string(), options.markov_w2_bits.to_string()),
    ]);
    let tensor_infos = output_tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                TensorInfo {
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                    data_offsets: (tensor.offset, tensor.offset + tensor.len_bytes),
                },
            )
        })
        .collect::<Vec<_>>();
    let metadata = Metadata::new(Some(metadata), tensor_infos)
        .map_err(|err| error(format!("unable to build quantized safetensors metadata: {err}")))?;
    let mut metadata_bytes = serde_json::to_vec(&metadata)
        .map_err(|err| error(format!("unable to encode quantized safetensors metadata: {err}")))?;
    metadata_bytes.resize(
        metadata_bytes.len().next_multiple_of(SAFETENSORS_HEADER_LEN_BYTES),
        b' ',
    );
    let data_start = SAFETENSORS_HEADER_LEN_BYTES
        .checked_add(metadata_bytes.len())
        .ok_or_else(|| error("quantized safetensors header size must fit usize"))?;
    let data_len = output_tensors
        .last()
        .map_or(0, |tensor| tensor.offset + tensor.len_bytes);
    let total_len = data_start
        .checked_add(data_len)
        .ok_or_else(|| error("quantized safetensors file size must fit usize"))?;

    let output = File::create(output_path)
        .map_err(|err| error(format!("unable to create quantized checkpoint {output_path:?}: {err}")))?;
    output
        .set_len(to_u64("quantized safetensors file length", total_len)?)
        .map_err(|err| error(format!("unable to size quantized checkpoint {output_path:?}: {err}")))?;
    let mut output = BufWriter::with_capacity(1024 * 1024, output);
    output
        .write_all(&to_u64("quantized safetensors header length", metadata_bytes.len())?.to_le_bytes())
        .and_then(|_| output.write_all(&metadata_bytes))
        .map_err(|err| {
            error(format!(
                "unable to write quantized checkpoint header {output_path:?}: {err}"
            ))
        })?;

    for (name, info) in &header.tensors {
        let source = read_tensor(&mut input, &header, name, info, input_path)?;
        if info.shape.len() == 2 {
            let bits = bits_for_tensor(name, options);
            let (weights, scales, biases) = quantize_bf16_matrix(&source, &info.shape, options.group_size, bits)?;
            write_output_tensor(
                &mut output,
                data_start,
                output_by_name[&name[..]],
                &weights,
                output_path,
            )?;
            let base = weight_base(name)?;
            write_output_tensor(
                &mut output,
                data_start,
                output_by_name[&format!("{base}.scales")[..]],
                &scales,
                output_path,
            )?;
            write_output_tensor(
                &mut output,
                data_start,
                output_by_name[&format!("{base}.biases")[..]],
                &biases,
                output_path,
            )?;
        } else {
            write_output_tensor(&mut output, data_start, output_by_name[&name[..]], &source, output_path)?;
        }
    }
    output
        .flush()
        .map_err(|err| error(format!("unable to flush quantized checkpoint {output_path:?}: {err}")))?;
    drop(output);
    let mut output = File::open(output_path)
        .map_err(|err| error(format!("unable to reopen quantized checkpoint {output_path:?}: {err}")))?;
    let output_header = read_header(&mut output, output_path)?;
    if output_header.tensors.len() != output_tensors.len() {
        return Err(error(format!(
            "quantized checkpoint tensor count={} differs from planned count={}",
            output_header.tensors.len(),
            output_tensors.len()
        )));
    }
    Ok(())
}

fn read_header(file: &mut File, path: &Path) -> Result<SafetensorsHeader> {
    file.seek(SeekFrom::Start(0))
        .and_then(|_| {
            let mut bytes = [0u8; SAFETENSORS_HEADER_LEN_BYTES];
            file.read_exact(&mut bytes)?;
            Ok(bytes)
        })
        .map_err(|err| error(format!("unable to read safetensors header length from {path:?}: {err}")))
        .and_then(|header_len_bytes| {
            let header_len = usize::try_from(u64::from_le_bytes(header_len_bytes))
                .map_err(|_| error(format!("safetensors header in {path:?} is too large")))?;
            if header_len > MAX_SAFETENSORS_HEADER_BYTES {
                return Err(error(format!(
                    "safetensors header in {path:?} is {header_len} bytes; maximum supported is \
                     {MAX_SAFETENSORS_HEADER_BYTES}"
                )));
            }
            let mut bytes = vec![0u8; header_len];
            file.read_exact(&mut bytes)
                .map_err(|err| error(format!("unable to read safetensors header from {path:?}: {err}")))?;
            let mut values = serde_json::from_slice::<BTreeMap<String, serde_json::Value>>(&bytes)
                .map_err(|err| error(format!("unable to parse safetensors header from {path:?}: {err}")))?;
            values.remove("__metadata__");
            let mut tensors = BTreeMap::new();
            let mut by_offset = Vec::with_capacity(values.len());
            for (name, value) in values {
                let info = serde_json::from_value::<TensorInfo>(value)
                    .map_err(|err| error(format!("invalid tensor metadata for {name:?} in {path:?}: {err}")))?;
                by_offset.push((name.clone(), info.clone()));
                tensors.insert(name, info);
            }
            by_offset.sort_by_key(|(_, info)| info.data_offsets);
            Metadata::new(None, by_offset)
                .map_err(|err| error(format!("invalid safetensors offsets in {path:?}: {err}")))?;
            let data_start = SAFETENSORS_HEADER_LEN_BYTES
                .checked_add(header_len)
                .ok_or_else(|| error(format!("safetensors data offset in {path:?} must fit usize")))?;
            let data_len = tensors.values().map(|info| info.data_offsets.1).max().unwrap_or(0);
            let expected_len = data_start
                .checked_add(data_len)
                .ok_or_else(|| error(format!("safetensors file length for {path:?} must fit usize")))?;
            let actual_len = usize::try_from(
                file.metadata()
                    .map_err(|err| error(format!("unable to stat safetensors file {path:?}: {err}")))?
                    .len(),
            )
            .map_err(|_| error(format!("safetensors file {path:?} is too large for this platform")))?;
            if actual_len != expected_len {
                return Err(error(format!(
                    "safetensors file {path:?} length={actual_len} differs from metadata length={expected_len}"
                )));
            }
            Ok(SafetensorsHeader {
                data_start: to_u64("safetensors data offset", data_start)?,
                tensors,
            })
        })
}

fn validate_source_tensors(header: &SafetensorsHeader, config: &DSparkConfig, path: &Path) -> Result<()> {
    let expected = expected_source_tensor_names(config);
    let actual = header
        .tensors
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if actual != expected {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
        return Err(error(format!(
            "DSpark checkpoint {path:?} tensor set mismatch: missing={missing:?} unexpected={unexpected:?}"
        )));
    }
    for (name, info) in &header.tensors {
        if info.dtype != Dtype::BF16 {
            return Err(error(format!(
                "DSpark source tensor {name:?} must be BF16, found {:?}",
                info.dtype
            )));
        }
        if !matches!(info.shape.len(), 1 | 2) {
            return Err(error(format!(
                "DSpark source tensor {name:?} must be rank 1 or 2, shape={:?}",
                info.shape
            )));
        }
    }
    Ok(())
}

fn expected_source_tensor_names(config: &DSparkConfig) -> std::collections::BTreeSet<String> {
    Qwen35DSparkWeightBindings::from_config(config)
        .source_tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn validate_output_tensor_names(output_tensors: &[OutputTensor], config: &DSparkConfig) -> Result<()> {
    let expected = Qwen35DSparkWeightBindings::from_config(config)
        .tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>();
    let actual = output_tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<std::collections::BTreeSet<_>>();
    if actual != expected {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
        return Err(error(format!(
            "quantized DSpark tensor set mismatch: missing={missing:?} unexpected={unexpected:?}"
        )));
    }
    Ok(())
}

fn build_output_tensors(header: &SafetensorsHeader, options: QuantizeOptions) -> Result<Vec<OutputTensor>> {
    let mut tensors = Vec::new();
    for (name, info) in &header.tensors {
        if info.shape.len() == 2 {
            let bits = bits_for_tensor(name, options);
            let input_dim = info.shape[1];
            if !input_dim.is_multiple_of(options.group_size) {
                return Err(error(format!(
                    "tensor {name:?} input_dim={input_dim} must be divisible by group_size={}",
                    options.group_size
                )));
            }
            let packed_bits = input_dim
                .checked_mul(bits)
                .ok_or_else(|| error(format!("packed dimension for {name:?} must fit usize")))?;
            if !packed_bits.is_multiple_of(32) {
                return Err(error(format!(
                    "packed dimension for {name:?} must be divisible by 32 bits"
                )));
            }
            let mut packed_shape = info.shape.clone();
            packed_shape[1] = packed_bits / 32;
            tensors.push(output_tensor(name.clone(), Dtype::U32, packed_shape)?);
            let mut affine_shape = info.shape.clone();
            affine_shape[1] = input_dim / options.group_size;
            let base = weight_base(name)?;
            tensors.push(output_tensor(
                format!("{base}.scales"),
                Dtype::BF16,
                affine_shape.clone(),
            )?);
            tensors.push(output_tensor(format!("{base}.biases"), Dtype::BF16, affine_shape)?);
        } else {
            tensors.push(output_tensor(name.clone(), info.dtype, info.shape.clone())?);
        }
    }
    tensors.sort_by(|left, right| right.dtype.cmp(&left.dtype).then(left.name.cmp(&right.name)));
    let mut offset = 0usize;
    for tensor in &mut tensors {
        tensor.offset = offset;
        offset = offset
            .checked_add(tensor.len_bytes)
            .ok_or_else(|| error("quantized safetensors data length must fit usize"))?;
    }
    Ok(tensors)
}

fn output_tensor(name: String, dtype: Dtype, shape: Vec<usize>) -> Result<OutputTensor> {
    let elements = checked_product(&format!("tensor {name:?} element count"), &shape)?;
    let len_bits = elements
        .checked_mul(dtype.bitsize())
        .ok_or_else(|| error(format!("tensor {name:?} bit length must fit usize")))?;
    if !len_bits.is_multiple_of(8) {
        return Err(error(format!(
            "tensor {name:?} bit length={len_bits} must be byte aligned"
        )));
    }
    let len_bytes = len_bits / 8;
    Ok(OutputTensor {
        name,
        dtype,
        shape,
        offset: 0,
        len_bytes,
    })
}

fn read_tensor(
    file: &mut File,
    header: &SafetensorsHeader,
    name: &str,
    info: &TensorInfo,
    path: &Path,
) -> Result<Vec<u8>> {
    let start = header
        .data_start
        .checked_add(to_u64("source tensor offset", info.data_offsets.0)?)
        .ok_or_else(|| error(format!("source tensor {name:?} file offset must fit u64")))?;
    let len = info
        .data_offsets
        .1
        .checked_sub(info.data_offsets.0)
        .ok_or_else(|| error(format!("source tensor {name:?} has invalid offsets")))?;
    let mut data = vec![0u8; len];
    file.seek(SeekFrom::Start(start))
        .and_then(|_| file.read_exact(&mut data))
        .map_err(|err| error(format!("unable to read source tensor {name:?} from {path:?}: {err}")))?;
    Ok(data)
}

fn write_output_tensor(
    output: &mut BufWriter<File>,
    data_start: usize,
    tensor: &OutputTensor,
    data: &[u8],
    path: &Path,
) -> Result<()> {
    if data.len() != tensor.len_bytes {
        return Err(error(format!(
            "output tensor {:?} data length={} differs from planned length={}",
            tensor.name,
            data.len(),
            tensor.len_bytes
        )));
    }
    let offset = data_start
        .checked_add(tensor.offset)
        .ok_or_else(|| error(format!("output tensor {:?} file offset must fit usize", tensor.name)))?;
    output
        .seek(SeekFrom::Start(to_u64("output tensor file offset", offset)?))
        .and_then(|_| output.write_all(data))
        .map_err(|err| {
            error(format!(
                "unable to write output tensor {:?} to {path:?}: {err}",
                tensor.name
            ))
        })?;
    Ok(())
}

fn quantize_bf16_matrix(
    source: &[u8],
    shape: &[usize],
    group_size: usize,
    bits: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if shape.len() != 2 {
        return Err(error(format!("quantized matrix must be rank 2, shape={shape:?}")));
    }
    let rows = shape[0];
    let columns = shape[1];
    if !columns.is_multiple_of(group_size) {
        return Err(error(format!(
            "quantized matrix columns={columns} must be divisible by group_size={group_size}"
        )));
    }
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| error("quantized matrix element count must fit usize"))?;
    if source.len() != elements * 2 {
        return Err(error(format!(
            "BF16 matrix byte length={} differs from expected={}",
            source.len(),
            elements * 2
        )));
    }
    let groups_per_row = columns / group_size;
    let words_per_row = columns
        .checked_mul(bits)
        .ok_or_else(|| error("packed row bit count must fit usize"))?
        / 32;
    let mut packed = vec![0u8; rows * words_per_row * 4];
    let mut scales = vec![0u8; rows * groups_per_row * 2];
    let mut biases = vec![0u8; rows * groups_per_row * 2];
    let bins = ((1u32 << bits) - 1) as f32;
    for row in 0..rows {
        for group in 0..groups_per_row {
            let first = row * columns + group * group_size;
            let mut minimum = f32::INFINITY;
            let mut maximum = f32::NEG_INFINITY;
            for index in first..first + group_size {
                let value = read_bf16(source, index);
                if !value.is_finite() {
                    return Err(error(format!(
                        "BF16 matrix contains non-finite value at row={row} column={}",
                        index - row * columns
                    )));
                }
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
            let positive_scale = ((maximum - minimum) / bins).max(1e-7);
            let mut scale = if minimum.abs() > maximum.abs() {
                positive_scale
            } else {
                -positive_scale
            };
            let edge = if minimum.abs() > maximum.abs() {
                minimum
            } else {
                maximum
            };
            let q0 = (edge / scale).round();
            if q0 != 0.0 {
                scale = edge / q0;
            }
            let bias = if q0 == 0.0 { 0.0 } else { edge };
            let affine_index = row * groups_per_row + group;
            write_bf16(&mut scales, affine_index, scale);
            write_bf16(&mut biases, affine_index, bias);
            for column in group * group_size..(group + 1) * group_size {
                let value = read_bf16(source, row * columns + column);
                let quantized = ((value - bias) / scale).round().clamp(0.0, bins) as u32;
                pack_bits(
                    &mut packed[row * words_per_row * 4..][..words_per_row * 4],
                    column,
                    bits,
                    quantized,
                );
            }
        }
    }
    Ok((packed, scales, biases))
}

fn pack_bits(row: &mut [u8], index: usize, bits: usize, value: u32) {
    let bit_offset = index * bits;
    let word_index = bit_offset / 32;
    let shift = bit_offset % 32;
    let mut word = read_u32(row, word_index);
    word |= value << shift;
    write_u32(row, word_index, word);
    if shift + bits > 32 {
        let mut next = read_u32(row, word_index + 1);
        next |= value >> (32 - shift);
        write_u32(row, word_index + 1, next);
    }
}

fn read_u32(data: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(
        data[index * 4..index * 4 + 4]
            .try_into()
            .expect("u32 slice length is fixed"),
    )
}

fn write_u32(data: &mut [u8], index: usize, value: u32) {
    data[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
}

fn read_bf16(data: &[u8], index: usize) -> f32 {
    bf16::from_bits(u16::from_le_bytes(
        data[index * 2..index * 2 + 2]
            .try_into()
            .expect("BF16 slice length is fixed"),
    ))
    .to_f32()
}

fn write_bf16(data: &mut [u8], index: usize, value: f32) {
    data[index * 2..index * 2 + 2].copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
}

fn bits_for_tensor(name: &str, options: QuantizeOptions) -> usize {
    if name == "markov_head.markov_w2.weight" {
        options.markov_w2_bits
    } else {
        options.bits
    }
}

fn weight_base(name: &str) -> Result<&str> {
    name.strip_suffix(".weight")
        .ok_or_else(|| error(format!("quantized matrix name {name:?} must end in .weight")))
}

fn write_output_config(config: &mut serde_json::Value, path: &Path, options: QuantizeOptions) -> Result<()> {
    let object = config
        .as_object_mut()
        .ok_or_else(|| error("DSpark config root must be a JSON object"))?;
    let mut quantization = serde_json::Map::from_iter([
        ("group_size".to_string(), serde_json::Value::from(options.group_size)),
        ("bits".to_string(), serde_json::Value::from(options.bits)),
        ("mode".to_string(), serde_json::Value::from("affine")),
    ]);
    if options.markov_w2_bits != options.bits {
        quantization.insert(
            "markov_head.markov_w2".to_string(),
            serde_json::json!({ "bits": options.markov_w2_bits }),
        );
    }
    object.insert("quantization".to_string(), serde_json::Value::Object(quantization));
    let bytes = serde_json::to_vec_pretty(config)
        .map_err(|err| error(format!("unable to encode quantized DSpark config: {err}")))?;
    std::fs::write(path, bytes)
        .map_err(|err| error(format!("unable to write quantized DSpark config {path:?}: {err}")))?;
    Ok(())
}

fn checked_product(name: &str, factors: &[usize]) -> Result<usize> {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .ok_or_else(|| error(format!("{name} must fit usize")))
}

fn to_u64(name: &str, value: usize) -> Result<u64> {
    value
        .try_into()
        .map_err(|_| error(format!("{name}={value} must fit u64")))
}

fn error(message: impl Into<String>) -> QuantizeError {
    QuantizeError(message.into())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use safetensors::SafeTensors;
    use safetensors::tensor::View;
    use safetensors::tensor::serialize_to_file;

    use super::*;

    struct OwnedTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for &OwnedTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    #[test]
    fn test_quantize_matrix_matches_affine_contract() {
        let values = (0..128).map(|index| (index as f32 - 63.5) / 17.0).collect::<Vec<_>>();
        let source = bf16_bytes(&values);

        let (weights, scales, biases) = quantize_bf16_matrix(&source, &[2, 64], 64, 4).unwrap();

        assert_eq!(weights.len(), 64);
        assert_eq!(scales.len(), 4);
        assert_eq!(biases.len(), 4);
        for row in 0..2 {
            let scale = read_bf16(&scales, row);
            let bias = read_bf16(&biases, row);
            for column in 0..64 {
                let byte = weights[row * 32 + column / 2];
                let quantized = if column.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let reconstructed = scale * f32::from(quantized) + bias;
                assert!((reconstructed - values[row * 64 + column]).abs() < 0.3);
            }
        }
    }

    #[test]
    fn test_non_power_of_two_bit_packing() {
        for bits in [3, 6] {
            let count = 32;
            let mut packed = vec![0u8; count * bits / 8];
            let mask = (1u32 << bits) - 1;
            for index in 0..count {
                pack_bits(&mut packed, index, bits, index as u32 & mask);
            }
            for index in 0..count {
                assert_eq!(unpack_bits(&packed, index, bits), index as u32 & mask);
            }
        }
    }

    #[test]
    fn test_tiny_checkpoint_round_trip() {
        let root = std::env::temp_dir().join(format!(
            "psi-dspark-quantize-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let input_dir = root.join("input");
        let output_dir = root.join("output");
        std::fs::create_dir_all(&input_dir).unwrap();
        let config = tiny_config();
        std::fs::write(
            input_dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let parsed = serde_json::from_value::<DSparkConfig>(config).unwrap();
        let tensors = tiny_tensors(&parsed);
        serialize_to_file(
            tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
            None,
            &input_dir.join("model.safetensors"),
        )
        .unwrap();

        quantize_checkpoint(
            &input_dir,
            &output_dir,
            QuantizeOptions {
                group_size: 32,
                bits: 4,
                markov_w2_bits: 8,
            },
        )
        .unwrap();

        let bytes = std::fs::read(output_dir.join("model.safetensors")).unwrap();
        let checkpoint = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(checkpoint.tensor("fc.weight").unwrap().dtype(), Dtype::U32);
        assert_eq!(checkpoint.tensor("fc.weight").unwrap().shape(), [64, 16]);
        assert_eq!(checkpoint.tensor("fc.scales").unwrap().shape(), [64, 4]);
        assert_eq!(
            checkpoint.tensor("markov_head.markov_w2.weight").unwrap().shape(),
            [128, 16]
        );
        assert_eq!(checkpoint.tensor("norm.weight").unwrap().dtype(), Dtype::BF16);
        let output_config =
            serde_json::from_slice::<serde_json::Value>(&std::fs::read(output_dir.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(output_config["quantization"]["bits"], 4);
        assert_eq!(output_config["quantization"]["markov_head.markov_w2"]["bits"], 8);
        std::fs::remove_dir_all(root).unwrap();
    }

    fn tiny_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "block_size": 5,
            "dflash_config": {
                "causal_head": false,
                "causal": false,
                "mask_token_id": 127,
                "target_layer_ids": [0, 1]
            },
            "dtype": "bfloat16",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_target_layers": 2,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0,
            "max_position_embeddings": 8192,
            "vocab_size": 128,
            "markov_rank": 64,
            "markov_head_type": "vanilla",
            "layer_types": ["full_attention", "full_attention"]
        })
    }

    fn tiny_tensors(config: &DSparkConfig) -> BTreeMap<String, OwnedTensor> {
        let mut tensors = BTreeMap::new();
        insert_matrix(&mut tensors, "fc.weight", config.hidden_size, config.hidden_size * 2);
        insert_norm(&mut tensors, "hidden_norm.weight", config.hidden_size);
        insert_norm(&mut tensors, "norm.weight", config.hidden_size);
        insert_matrix(
            &mut tensors,
            "markov_head.markov_w1.weight",
            config.vocab_size,
            config.markov_rank,
        );
        insert_matrix(
            &mut tensors,
            "markov_head.markov_w2.weight",
            config.vocab_size,
            config.markov_rank,
        );
        for layer in 0..config.num_layers {
            let prefix = format!("layers.{layer}");
            insert_norm(
                &mut tensors,
                &format!("{prefix}.input_layernorm.weight"),
                config.hidden_size,
            );
            insert_norm(
                &mut tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.hidden_size,
            );
            insert_matrix(
                &mut tensors,
                &format!("{prefix}.self_attn.q_proj.weight"),
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
            );
            for projection in ["k_proj", "v_proj"] {
                insert_matrix(
                    &mut tensors,
                    &format!("{prefix}.self_attn.{projection}.weight"),
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                );
            }
            insert_matrix(
                &mut tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
            );
            insert_norm(
                &mut tensors,
                &format!("{prefix}.self_attn.q_norm.weight"),
                config.head_dim,
            );
            insert_norm(
                &mut tensors,
                &format!("{prefix}.self_attn.k_norm.weight"),
                config.head_dim,
            );
            for projection in ["gate_proj", "up_proj"] {
                insert_matrix(
                    &mut tensors,
                    &format!("{prefix}.mlp.{projection}.weight"),
                    config.intermediate_size,
                    config.hidden_size,
                );
            }
            insert_matrix(
                &mut tensors,
                &format!("{prefix}.mlp.down_proj.weight"),
                config.hidden_size,
                config.intermediate_size,
            );
        }
        tensors
    }

    fn insert_matrix(tensors: &mut BTreeMap<String, OwnedTensor>, name: &str, rows: usize, columns: usize) {
        let values = (0..rows * columns)
            .map(|index| ((index % 97) as f32 - 48.0) / 37.0)
            .collect::<Vec<_>>();
        tensors.insert(
            name.to_string(),
            OwnedTensor {
                dtype: Dtype::BF16,
                shape: vec![rows, columns],
                data: bf16_bytes(&values),
            },
        );
    }

    fn insert_norm(tensors: &mut BTreeMap<String, OwnedTensor>, name: &str, dimension: usize) {
        tensors.insert(
            name.to_string(),
            OwnedTensor {
                dtype: Dtype::BF16,
                shape: vec![dimension],
                data: bf16_bytes(&vec![1.0; dimension]),
            },
        );
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect()
    }

    fn unpack_bits(data: &[u8], index: usize, bits: usize) -> u32 {
        let bit_offset = index * bits;
        let word_index = bit_offset / 32;
        let shift = bit_offset % 32;
        let mask = (1u32 << bits) - 1;
        let mut value = read_u32(data, word_index) >> shift;
        if shift + bits > 32 {
            value |= read_u32(data, word_index + 1) << (32 - shift);
        }
        value & mask
    }
}
