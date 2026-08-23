use std::collections::BTreeMap;
use std::collections::BTreeSet;
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
use safetensors::Dtype;
use safetensors::tensor::Metadata;
use safetensors::tensor::TensorInfo;

const SAFETENSORS_HEADER_LEN_BYTES: usize = 8;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 100_000_000;

#[derive(Debug)]
pub struct QuantizeError(String);

impl fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for QuantizeError {}

pub type Result<T> = std::result::Result<T, QuantizeError>;

#[derive(Clone, Debug)]
pub struct SafetensorsHeader {
    data_start: u64,
    pub tensors: BTreeMap<String, TensorInfo>,
}

pub struct ConversionPlan {
    pub model_name: &'static str,
    pub format: &'static str,
    pub source_max_rank: usize,
    pub group_size: usize,
    pub bits: usize,
    pub bit_overrides: BTreeMap<String, usize>,
    pub unquantized_matrices: BTreeSet<String>,
    pub renamed_tensors: BTreeMap<String, String>,
    pub expected_output_names: BTreeSet<String>,
    pub metadata: HashMap<String, String>,
    pub config: serde_json::Value,
}

#[derive(Clone, Debug)]
struct OutputTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    offset: usize,
    len_bytes: usize,
}

pub fn convert_checkpoint(
    input_dir: &Path,
    output_dir: &Path,
    build_plan: impl FnOnce(&Path, &SafetensorsHeader, serde_json::Value) -> Result<ConversionPlan>,
) -> Result<()> {
    if output_dir.exists() {
        return Err(error(format!("output directory {output_dir:?} already exists")));
    }
    let input_config_path = input_dir.join("config.json");
    let input_config_bytes = std::fs::read(&input_config_path)
        .map_err(|err| error(format!("unable to read config {input_config_path:?}: {err}")))?;
    let config = serde_json::from_slice::<serde_json::Value>(&input_config_bytes)
        .map_err(|err| error(format!("unable to parse config {input_config_path:?}: {err}")))?;
    let input_checkpoint = input_dir.join("model.safetensors");
    if !input_checkpoint.is_file() {
        return Err(error(format!(
            "converter requires the linked single-file checkpoint at {input_checkpoint:?}"
        )));
    }

    let mut input = File::open(&input_checkpoint)
        .map_err(|err| error(format!("unable to open BF16 checkpoint {input_checkpoint:?}: {err}")))?;
    let header = read_header(&mut input, &input_checkpoint)?;
    let plan = build_plan(input_dir, &header, config)?;
    validate_plan(&plan)?;
    validate_source_tensors(&header, &plan)?;

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
    let output_checkpoint = temp_dir.join("model.safetensors");
    quantize_safetensors(&mut input, &input_checkpoint, &output_checkpoint, &header, &plan)?;
    write_safetensors_index(
        &output_checkpoint,
        &temp_dir.join("model.safetensors.index.json"),
        plan.model_name,
    )?;
    write_output_config(&plan.config, &temp_dir.join("config.json"), plan.model_name)?;
    std::fs::rename(&temp_dir, output_dir).map_err(|err| {
        error(format!(
            "unable to publish quantized {} checkpoint {temp_dir:?} as {output_dir:?}: {err}",
            plan.model_name
        ))
    })?;
    cleanup.committed = true;
    Ok(())
}

impl ConversionPlan {
    fn output_name<'a>(&'a self, source_name: &'a str) -> &'a str {
        self.renamed_tensors
            .get(source_name)
            .map_or(source_name, String::as_str)
    }

    fn quantizes(&self, name: &str, info: &TensorInfo) -> bool {
        info.shape.len() == 2 && !self.unquantized_matrices.contains(name)
    }

    fn bits_for(&self, name: &str) -> usize {
        self.bit_overrides.get(name).copied().unwrap_or(self.bits)
    }
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

fn validate_plan(plan: &ConversionPlan) -> Result<()> {
    if !matches!(plan.group_size, 32 | 64 | 128) {
        return Err(error(format!(
            "unsupported affine group_size={}; expected 32, 64, or 128",
            plan.group_size
        )));
    }
    for (name, bits) in
        std::iter::once(("bits", plan.bits)).chain(plan.bit_overrides.iter().map(|(name, &bits)| (name.as_str(), bits)))
    {
        if !matches!(bits, 2 | 3 | 4 | 6 | 8) {
            return Err(error(format!("unsupported {name}={bits}; expected 2, 3, 4, 6, or 8")));
        }
    }
    Ok(())
}

fn validate_source_tensors(header: &SafetensorsHeader, plan: &ConversionPlan) -> Result<()> {
    for (name, info) in &header.tensors {
        if info.dtype != Dtype::BF16 {
            return Err(error(format!(
                "{} source tensor {name:?} must be BF16, found {:?}",
                plan.model_name, info.dtype
            )));
        }
        if !(1..=plan.source_max_rank).contains(&info.shape.len()) {
            return Err(error(format!(
                "{} source tensor {name:?} must have rank 1 through {}, shape={:?}",
                plan.model_name, plan.source_max_rank, info.shape
            )));
        }
    }
    Ok(())
}

fn quantize_safetensors(
    input: &mut File,
    input_path: &Path,
    output_path: &Path,
    header: &SafetensorsHeader,
    plan: &ConversionPlan,
) -> Result<()> {
    let output_tensors = build_output_tensors(header, plan)?;
    validate_output_tensor_names(&output_tensors, plan)?;
    let output_by_name = output_tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<HashMap<_, _>>();
    let mut metadata = plan.metadata.clone();
    metadata.extend([
        ("format".to_string(), plan.format.to_string()),
        ("group_size".to_string(), plan.group_size.to_string()),
        ("bits".to_string(), plan.bits.to_string()),
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
        .set_len(total_len as u64)
        .map_err(|err| error(format!("unable to size quantized checkpoint {output_path:?}: {err}")))?;
    let mut output = BufWriter::with_capacity(1024 * 1024, output);
    output
        .write_all(&(metadata_bytes.len() as u64).to_le_bytes())
        .and_then(|_| output.write_all(&metadata_bytes))
        .map_err(|err| {
            error(format!(
                "unable to write quantized checkpoint header {output_path:?}: {err}"
            ))
        })?;

    for (name, info) in &header.tensors {
        let source = read_tensor(input, header, name, info, input_path)?;
        let output_name = plan.output_name(name);
        if plan.quantizes(name, info) {
            let bits = plan.bits_for(name);
            let (weights, scales, biases) = quantize_bf16_matrix(&source, &info.shape, plan.group_size, bits)?;
            write_output_tensor(
                &mut output,
                data_start,
                output_by_name[output_name],
                &weights,
                output_path,
            )?;
            let base = weight_base(output_name)?;
            write_output_tensor(
                &mut output,
                data_start,
                output_by_name[format!("{base}.scales").as_str()],
                &scales,
                output_path,
            )?;
            write_output_tensor(
                &mut output,
                data_start,
                output_by_name[format!("{base}.biases").as_str()],
                &biases,
                output_path,
            )?;
        } else {
            write_output_tensor(
                &mut output,
                data_start,
                output_by_name[output_name],
                &source,
                output_path,
            )?;
        }
    }
    output
        .flush()
        .map_err(|err| error(format!("unable to flush quantized checkpoint {output_path:?}: {err}")))?;
    Ok(())
}

fn build_output_tensors(header: &SafetensorsHeader, plan: &ConversionPlan) -> Result<Vec<OutputTensor>> {
    let mut tensors = Vec::new();
    for (name, info) in &header.tensors {
        let output_name = plan.output_name(name);
        if plan.quantizes(name, info) {
            let bits = plan.bits_for(name);
            let input_dim = info.shape[1];
            if !input_dim.is_multiple_of(plan.group_size) {
                return Err(error(format!(
                    "tensor {name:?} input_dim={input_dim} must be divisible by group_size={}",
                    plan.group_size
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
            tensors.push(output_tensor(output_name.to_string(), Dtype::U32, packed_shape)?);
            let mut affine_shape = info.shape.clone();
            affine_shape[1] = input_dim / plan.group_size;
            let base = weight_base(output_name)?;
            tensors.push(output_tensor(
                format!("{base}.scales"),
                Dtype::BF16,
                affine_shape.clone(),
            )?);
            tensors.push(output_tensor(format!("{base}.biases"), Dtype::BF16, affine_shape)?);
        } else {
            tensors.push(output_tensor(output_name.to_string(), info.dtype, info.shape.clone())?);
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

fn validate_output_tensor_names(output_tensors: &[OutputTensor], plan: &ConversionPlan) -> Result<()> {
    let actual = output_tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    if actual != plan.expected_output_names {
        let missing = plan
            .expected_output_names
            .difference(&actual)
            .cloned()
            .collect::<Vec<_>>();
        let unexpected = actual
            .difference(&plan.expected_output_names)
            .cloned()
            .collect::<Vec<_>>();
        return Err(error(format!(
            "quantized {} tensor set mismatch: missing={missing:?} unexpected={unexpected:?}",
            plan.model_name
        )));
    }
    Ok(())
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
    Ok(OutputTensor {
        name,
        dtype,
        shape,
        offset: 0,
        len_bytes: len_bits / 8,
    })
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
                data_start: data_start as u64,
                tensors,
            })
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
        .checked_add(info.data_offsets.0 as u64)
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
        .seek(SeekFrom::Start(offset as u64))
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
    let source_bytes = elements
        .checked_mul(Dtype::BF16.bitsize() / 8)
        .ok_or_else(|| error("quantized matrix byte length must fit usize"))?;
    if source.len() != source_bytes {
        return Err(error(format!(
            "BF16 matrix byte length={} differs from expected={source_bytes}",
            source.len()
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

            // The stored BF16 values define dequantization. Derive codes from
            // those exact values instead of the transient F32 calculation.
            let scale = bf16::from_f32(scale).to_f32();
            let bias = bf16::from_f32(bias).to_f32();
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

fn weight_base(name: &str) -> Result<&str> {
    name.strip_suffix(".weight")
        .ok_or_else(|| error(format!("quantized matrix name {name:?} must end in .weight")))
}

fn write_output_config(config: &serde_json::Value, path: &Path, model_name: &str) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(config)
        .map_err(|err| error(format!("unable to encode quantized {model_name} config: {err}")))?;
    std::fs::write(path, bytes)
        .map_err(|err| error(format!("unable to write quantized {model_name} config {path:?}: {err}")))?;
    Ok(())
}

fn write_safetensors_index(checkpoint_path: &Path, index_path: &Path, model_name: &str) -> Result<()> {
    let mut checkpoint = File::open(checkpoint_path).map_err(|err| {
        error(format!(
            "unable to open affine {model_name} checkpoint {checkpoint_path:?}: {err}"
        ))
    })?;
    let header = read_header(&mut checkpoint, checkpoint_path)?;
    let total_size = header
        .tensors
        .values()
        .map(|tensor| tensor.data_offsets.1)
        .max()
        .unwrap_or(0);
    let weight_map = header
        .tensors
        .keys()
        .map(|name| (name, "model.safetensors"))
        .collect::<BTreeMap<_, _>>();
    let index = serde_json::json!({
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map,
    });
    let bytes = serde_json::to_vec_pretty(&index)
        .map_err(|err| error(format!("unable to encode affine {model_name} safetensors index: {err}")))?;
    std::fs::write(index_path, bytes).map_err(|err| {
        error(format!(
            "unable to write affine {model_name} safetensors index {index_path:?}: {err}"
        ))
    })?;
    Ok(())
}

fn checked_product(name: &str, factors: &[usize]) -> Result<usize> {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .ok_or_else(|| error(format!("{name} must fit usize")))
}

pub fn error(message: impl Into<String>) -> QuantizeError {
    QuantizeError(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_codes_match_stored_bf16_parameters() {
        let mut values = vec![0.0; 64];
        values[0] = -9.9375;
        values[1] = 9.9375;
        values[2] = -2.140625;
        let source = values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();

        let (weights, scales, biases) = quantize_bf16_matrix(&source, &[1, 64], 64, 4).unwrap();

        let scale = read_bf16(&scales, 0);
        let bias = read_bf16(&biases, 0);
        for column in 0..64 {
            let source_value = read_bf16(&source, column);
            let expected = ((source_value - bias) / scale).round().clamp(0.0, 15.0) as u32;
            assert_eq!(unpack_bits(&weights, column, 4), expected);
        }
        assert_eq!(unpack_bits(&weights, 2, 4), 8);
    }

    #[test]
    fn test_packed_codes_cross_u32_boundaries() {
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
