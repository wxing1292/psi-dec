use inference_backend_metal::components::audio_encoder_layout;
use inference_backend_metal::components::layer_norm;
use inference_backend_metal::components::residual_add;
use inference_backend_metal::components::tower_block_attention;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgramBuilder;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::bias_activation_bf16;
use inference_backend_metal::operators::conv2d_unfold;
use inference_backend_metal::operators::matmul_bf16;
use inference_executor_core::checkpoint::SafeTensorStore;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRAudioConfig;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRAudioSource;
use inference_executor_core::model::qwen::v3_asr::audio_output_rows;
use inference_executor_core::model::qwen::v3_asr::weight_layout::Qwen3ASRAffineWeightBindings;
use inference_executor_core::model::qwen::v3_asr::weight_layout::Qwen3ASRAudioLayerWeightBindings;
use inference_executor_core::model::qwen::v3_asr::weight_layout::Qwen3ASRAudioWeightBindings;
use inference_executor_core::model::qwen::v3_asr::weight_layout::Qwen3ASRNormWeightBindings;
use inference_runtime_core::memory::OffsetAllocation;

use crate::model::resource_arena::MetalResourceArena;

const BF16_BYTES: usize = 2;
const CONV_CHANNELS: u32 = 480;

pub struct AudioTower {
    config: Qwen3ASRAudioConfig,
    dimensions: AudioDimensions,
    device: Device,
    stream: Stream,
    unfold: conv2d_unfold::Kernel,
    layout: audio_encoder_layout::Compute,
    bias_activation: bias_activation_bf16::Kernel,
    residual_add: residual_add::Compute,
    conv: [Affine; 3],
    conv_out: Linear,
    attention: tower_block_attention::Compute,
    layers: Vec<AudioLayer>,
    ln_post: Norm,
    proj1: Affine,
    proj2: Affine,
}

struct AudioLayer {
    self_attention_norm: Norm,
    q: Affine,
    k: Affine,
    v: Affine,
    output: Affine,
    final_norm: Norm,
    fc1: Affine,
    fc2: Affine,
}

struct Affine {
    matmul: matmul_bf16::Matmul,
    weight: Buffer,
    bias: Buffer,
    output_dim: u32,
}

struct Linear {
    matmul: matmul_bf16::Matmul,
    weight: Buffer,
}

struct Norm {
    compute: layer_norm::Compute,
    weight: Buffer,
    bias: Buffer,
}

#[derive(Clone, Copy)]
struct AudioDimensions {
    num_mel_bins: u32,
    d_model: u32,
    ffn_dim: u32,
    output_dim: u32,
    frames_per_chunk: u32,
    attention_window_rows: u32,
}

impl AudioTower {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: Qwen3ASRAudioConfig,
        bindings: Qwen3ASRAudioWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        let dimensions = AudioDimensions::from_config(&config)?;
        let downsample = as_u32(config.downsample_hidden_size, "audio downsample_hidden_size")?;
        let conv = [
            Affine::load_conv(device, store, bindings.conv[0].clone(), 1, downsample)?,
            Affine::load_conv(device, store, bindings.conv[1].clone(), downsample, downsample)?,
            Affine::load_conv(device, store, bindings.conv[2].clone(), downsample, downsample)?,
        ];
        let conv_out = Linear::load(
            device,
            store,
            &bindings.conv_out_weight,
            downsample * dimensions.num_mel_bins.div_ceil(8),
            dimensions.d_model,
        )?;
        let layers = bindings
            .layers
            .into_iter()
            .map(|bindings| AudioLayer::load(device, store, bindings, dimensions.d_model, dimensions.ffn_dim))
            .collect::<Result<Vec<_>, _>>()?;
        let ln_post = Norm::load(device, store, bindings.ln_post, dimensions.d_model)?;
        let proj1 = Affine::load(device, store, bindings.proj1, dimensions.d_model, dimensions.d_model)?;
        let proj2 = Affine::load(device, store, bindings.proj2, dimensions.d_model, dimensions.output_dim)?;
        let num_heads = as_u32(config.encoder_attention_heads, "audio attention heads")?;
        let head_dim = dimensions.d_model / num_heads;
        Ok(Self {
            config,
            dimensions,
            device: device.clone(),
            stream: Stream::new(device),
            unfold: conv2d_unfold::Kernel::new(device),
            layout: audio_encoder_layout::Compute::new(device),
            bias_activation: bias_activation_bf16::Kernel::new(device),
            residual_add: residual_add::Compute::new(device, residual_add::Config::bf16()),
            conv,
            conv_out,
            attention: tower_block_attention::Compute::new(
                device,
                tower_block_attention::Config { num_heads, head_dim },
            ),
            layers,
            ln_post,
            proj1,
            proj2,
        })
    }

    pub fn encode(&self, source: &Qwen3ASRAudioSource, arena: &MetalResourceArena, allocation: &OffsetAllocation) {
        debug_assert_eq!(source.num_mel_bins(), self.config.num_mel_bins);
        debug_assert!(source.num_frames() <= self.config.max_source_positions * 2);
        let shape = AudioShape::new(source.num_frames(), self.dimensions);
        debug_assert_eq!(
            allocation.len_bytes(),
            shape.output_bytes() as u64,
            "Qwen3-ASR resource allocation must match the Audio Tower output"
        );
        let output_offset_bytes = allocation.offset_bytes() as usize;

        let source_buffer = Buffer::from_slice(&self.device, source.features());
        let chunked = bf16_buffer(&self.device, shape.chunked_values());
        let max_unfold = bf16_buffer(&self.device, shape.max_unfold_values());
        let conv_a = bf16_buffer(&self.device, shape.max_conv_values());
        let conv_b = bf16_buffer(&self.device, shape.max_conv_values());
        let flattened = bf16_buffer(&self.device, shape.flattened_values());
        let projected = bf16_buffer(&self.device, shape.projected_values());
        let hidden_a = bf16_buffer(&self.device, shape.hidden_values());
        let hidden_b = bf16_buffer(&self.device, shape.hidden_values());
        let norm = bf16_buffer(&self.device, shape.hidden_values());
        let q = bf16_buffer(&self.device, shape.hidden_values());
        let k = bf16_buffer(&self.device, shape.hidden_values());
        let v = bf16_buffer(&self.device, shape.hidden_values());
        let branch = bf16_buffer(&self.device, shape.hidden_values());
        let ffn = bf16_buffer(&self.device, shape.ffn_values());

        let mut replay = self.stream.create_replay_program();
        replay.record(self.layout.invoke_chunk_log_mel(
            audio_encoder_layout::LogMelChunkShape {
                num_mel_bins: shape.num_mel_bins,
                num_frames: shape.num_frames,
                frames_per_chunk: shape.chunk_frames,
            },
            &source_buffer,
            &chunked,
        ));

        let conv_shapes = shape.conv_shapes();
        let mut input = &chunked;
        for ((affine, conv_shape), output) in self.conv.iter().zip(conv_shapes).zip([&conv_a, &conv_b, &conv_a]) {
            replay.record_with_barrier_before(self.unfold.invoke(conv_shape, input, &max_unfold));
            replay.record_with_barrier_before(affine.matmul.invoke(
                conv_shape.output_rows(),
                output,
                0,
                &max_unfold,
                0,
                &affine.weight,
                0,
            ));
            replay.record_with_barrier_before(self.bias_activation.invoke(
                bias_activation_bf16::Shape {
                    num_rows: conv_shape.output_rows(),
                    num_columns: affine.output_dim,
                },
                bias_activation_bf16::Activation::Gelu,
                output,
                0,
                &affine.bias,
                output,
                0,
            ));
            input = output;
        }

        let final_conv_shape = conv_shapes[2];
        replay.record_with_barrier_before(self.layout.invoke_flatten_conv(
            audio_encoder_layout::FlattenConvShape {
                batch: final_conv_shape.batch,
                height: final_conv_shape.output_height(),
                width: final_conv_shape.output_width(),
                channels: CONV_CHANNELS,
            },
            input,
            &flattened,
        ));
        replay.record_with_barrier_before(self.conv_out.matmul.invoke(
            shape.projected_rows,
            &projected,
            0,
            &flattened,
            0,
            &self.conv_out.weight,
            0,
        ));
        replay.record_with_barrier_before(self.layout.invoke_compact_position(
            audio_encoder_layout::CompactPositionShape {
                num_rows: shape.num_rows,
                source_rows_per_chunk: shape.rows_per_chunk,
                hidden_dim: shape.d_model,
            },
            &projected,
            &hidden_a,
        ));

        let mut residual = &hidden_a;
        let mut next = &hidden_b;
        for layer in &self.layers {
            replay.record_with_barrier_before(layer.self_attention_norm.invoke(shape.num_rows, residual, &norm));
            for (affine, destination) in [(&layer.q, &q), (&layer.k, &k), (&layer.v, &v)] {
                affine.record(
                    &mut replay,
                    shape.num_rows,
                    &norm,
                    destination,
                    bias_activation_bf16::Activation::Identity,
                    &self.bias_activation,
                );
            }
            replay.record_with_barrier_before(self.attention.invoke(
                tower_block_attention::Shape {
                    num_rows: shape.num_rows,
                    block_size: self.dimensions.attention_window_rows,
                },
                tower_block_attention::Buffers {
                    query: &q,
                    key: &k,
                    value: &v,
                    output: &branch,
                },
            ));
            layer.output.record(
                &mut replay,
                shape.num_rows,
                &branch,
                &q,
                bias_activation_bf16::Activation::Identity,
                &self.bias_activation,
            );
            self.record_residual_add(&mut replay, shape, residual, &q, next);
            std::mem::swap(&mut residual, &mut next);

            replay.record_with_barrier_before(layer.final_norm.invoke(shape.num_rows, residual, &norm));
            layer.fc1.record(
                &mut replay,
                shape.num_rows,
                &norm,
                &ffn,
                bias_activation_bf16::Activation::Gelu,
                &self.bias_activation,
            );
            layer.fc2.record(
                &mut replay,
                shape.num_rows,
                &ffn,
                &q,
                bias_activation_bf16::Activation::Identity,
                &self.bias_activation,
            );
            self.record_residual_add(&mut replay, shape, residual, &q, next);
            std::mem::swap(&mut residual, &mut next);
        }

        replay.record_with_barrier_before(self.ln_post.invoke(shape.num_rows, residual, &norm));
        self.proj1.record(
            &mut replay,
            shape.num_rows,
            &norm,
            &q,
            bias_activation_bf16::Activation::Gelu,
            &self.bias_activation,
        );
        replay.record_with_barrier_before(self.proj2.matmul.invoke(
            shape.num_rows,
            arena.storage().buffer(),
            output_offset_bytes,
            &q,
            0,
            &self.proj2.weight,
            0,
        ));
        replay.record_with_barrier_before(self.bias_activation.invoke(
            bias_activation_bf16::Shape {
                num_rows: shape.num_rows,
                num_columns: shape.output_dim,
            },
            bias_activation_bf16::Activation::Identity,
            arena.storage().buffer(),
            output_offset_bytes,
            &self.proj2.bias,
            arena.storage().buffer(),
            output_offset_bytes,
        ));
        self.stream.submit_replay(&replay.build()).wait();
    }

    fn record_residual_add<'a>(
        &'a self,
        replay: &mut ReplayProgramBuilder,
        shape: AudioShape,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        output: &'a Buffer,
    ) {
        replay.record_with_barrier_before(self.residual_add.invoke_values(
            residual_add::Shape {
                num_values: shape.hidden_values(),
            },
            residual_add::Buffers { lhs, rhs, output },
        ));
    }
}

impl AudioDimensions {
    fn from_config(config: &Qwen3ASRAudioConfig) -> Result<Self, ModelExecutorError> {
        let frames_per_chunk = as_u32(config.n_window * 2, "audio frames per chunk")?;
        let output_rows_per_chunk = frames_per_chunk.div_ceil(8);
        let chunks_per_attention_window = config.n_window_infer / (config.n_window * 2);
        Ok(Self {
            num_mel_bins: as_u32(config.num_mel_bins, "audio Mel bins")?,
            d_model: as_u32(config.d_model, "audio d_model")?,
            ffn_dim: as_u32(config.encoder_ffn_dim, "audio encoder_ffn_dim")?,
            output_dim: as_u32(config.output_dim, "audio output_dim")?,
            frames_per_chunk,
            attention_window_rows: output_rows_per_chunk
                * as_u32(chunks_per_attention_window, "audio attention chunks")?,
        })
    }
}

impl AudioLayer {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3ASRAudioLayerWeightBindings,
        d_model: u32,
        ffn_dim: u32,
    ) -> Result<Self, ModelExecutorError> {
        Ok(Self {
            self_attention_norm: Norm::load(device, store, bindings.self_attention_norm, d_model)?,
            q: Affine::load(device, store, bindings.q, d_model, d_model)?,
            k: Affine::load(device, store, bindings.k, d_model, d_model)?,
            v: Affine::load(device, store, bindings.v, d_model, d_model)?,
            output: Affine::load(device, store, bindings.output, d_model, d_model)?,
            final_norm: Norm::load(device, store, bindings.final_norm, d_model)?,
            fc1: Affine::load(device, store, bindings.fc1, d_model, ffn_dim)?,
            fc2: Affine::load(device, store, bindings.fc2, ffn_dim, d_model)?,
        })
    }
}

impl Affine {
    fn load_conv(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3ASRAffineWeightBindings,
        input_channels: u32,
        output_dim: u32,
    ) -> Result<Self, ModelExecutorError> {
        let expected_shape = [output_dim as usize, 3, 3, input_channels as usize];
        let tensor = load_bf16_bytes(store, &bindings.weight, &expected_shape)?;
        let actual_input_dim = 9 * input_channels;
        let input_dim = actual_input_dim.next_multiple_of(16);
        let weight = if input_dim == actual_input_dim {
            Buffer::from_slice(device, &tensor)
        } else {
            let actual_row_bytes = actual_input_dim as usize * BF16_BYTES;
            let padded_row_bytes = input_dim as usize * BF16_BYTES;
            let mut padded = vec![0; output_dim as usize * padded_row_bytes];
            for row in 0..output_dim as usize {
                padded[row * padded_row_bytes..row * padded_row_bytes + actual_row_bytes]
                    .copy_from_slice(&tensor[row * actual_row_bytes..(row + 1) * actual_row_bytes]);
            }
            Buffer::from_slice(device, &padded)
        };
        let bias = load_bf16(device, store, &bindings.bias, &[output_dim as usize])?;
        Ok(Self::new(device, input_dim, output_dim, weight, bias))
    }

    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3ASRAffineWeightBindings,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<Self, ModelExecutorError> {
        let weight = load_bf16(
            device,
            store,
            &bindings.weight,
            &[output_dim as usize, input_dim as usize],
        )?;
        let bias = load_bf16(device, store, &bindings.bias, &[output_dim as usize])?;
        Ok(Self::new(device, input_dim, output_dim, weight, bias))
    }

    fn new(device: &Device, input_dim: u32, output_dim: u32, weight: Buffer, bias: Buffer) -> Self {
        Self {
            matmul: matmul_bf16::Matmul::new(device, matmul_bf16::Config { input_dim, output_dim }),
            weight,
            bias,
            output_dim,
        }
    }

    fn record<'a>(
        &'a self,
        replay: &mut ReplayProgramBuilder,
        num_rows: u32,
        input: &'a Buffer,
        output: &'a Buffer,
        activation: bias_activation_bf16::Activation,
        bias_activation: &'a bias_activation_bf16::Kernel,
    ) {
        replay.record_with_barrier_before(self.matmul.invoke(num_rows, output, 0, input, 0, &self.weight, 0));
        replay.record_with_barrier_before(bias_activation.invoke(
            bias_activation_bf16::Shape {
                num_rows,
                num_columns: self.output_dim,
            },
            activation,
            output,
            0,
            &self.bias,
            output,
            0,
        ));
    }
}

impl Linear {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        name: &str,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<Self, ModelExecutorError> {
        Ok(Self {
            matmul: matmul_bf16::Matmul::new(device, matmul_bf16::Config { input_dim, output_dim }),
            weight: load_bf16(device, store, name, &[output_dim as usize, input_dim as usize])?,
        })
    }
}

impl Norm {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3ASRNormWeightBindings,
        hidden_dim: u32,
    ) -> Result<Self, ModelExecutorError> {
        Ok(Self {
            compute: layer_norm::Compute::new(device, layer_norm::Config { hidden_dim, eps: 1e-5 }),
            weight: load_bf16(device, store, &bindings.weight, &[hidden_dim as usize])?,
            bias: load_bf16(device, store, &bindings.bias, &[hidden_dim as usize])?,
        })
    }

    fn invoke<'a>(&'a self, num_rows: u32, input: &'a Buffer, output: &'a Buffer) -> layer_norm::Invocation<'a> {
        self.compute.invoke(
            layer_norm::Shape { num_rows },
            layer_norm::Buffers {
                input,
                weight: &self.weight,
                bias: &self.bias,
                output,
            },
        )
    }
}

#[derive(Clone, Copy)]
struct AudioShape {
    num_frames: u32,
    num_mel_bins: u32,
    num_chunks: u32,
    chunk_frames: u32,
    rows_per_chunk: u32,
    projected_rows: u32,
    num_rows: u32,
    d_model: u32,
    ffn_dim: u32,
    output_dim: u32,
}

impl AudioShape {
    fn new(num_frames: usize, dimensions: AudioDimensions) -> Self {
        debug_assert!(num_frames <= u32::MAX as usize);
        let num_frames = num_frames as u32;
        let num_chunks = num_frames.div_ceil(dimensions.frames_per_chunk);
        let chunk_frames = num_frames.min(dimensions.frames_per_chunk);
        let rows_per_chunk = chunk_frames.div_ceil(8);
        Self {
            num_frames,
            num_mel_bins: dimensions.num_mel_bins,
            num_chunks,
            chunk_frames,
            rows_per_chunk,
            projected_rows: num_chunks * rows_per_chunk,
            num_rows: audio_output_rows(num_frames as usize) as u32,
            d_model: dimensions.d_model,
            ffn_dim: dimensions.ffn_dim,
            output_dim: dimensions.output_dim,
        }
    }

    fn conv_shapes(self) -> [conv2d_unfold::Shape; 3] {
        let first = conv2d_unfold::Shape {
            batch: self.num_chunks,
            input_height: self.num_mel_bins,
            input_width: self.chunk_frames,
            channels: 1,
        };
        let second = conv2d_unfold::Shape {
            batch: self.num_chunks,
            input_height: first.output_height(),
            input_width: first.output_width(),
            channels: CONV_CHANNELS,
        };
        let third = conv2d_unfold::Shape {
            batch: self.num_chunks,
            input_height: second.output_height(),
            input_width: second.output_width(),
            channels: CONV_CHANNELS,
        };
        [first, second, third]
    }

    fn chunked_values(self) -> u32 {
        self.num_chunks * self.num_mel_bins * self.chunk_frames
    }

    fn max_unfold_values(self) -> u32 {
        let [first, second, third] = self.conv_shapes();
        first
            .output_values()
            .max(second.output_values())
            .max(third.output_values())
    }

    fn max_conv_values(self) -> u32 {
        let [first, second, third] = self.conv_shapes();
        first.output_rows().max(second.output_rows()).max(third.output_rows()) * CONV_CHANNELS
    }

    fn flattened_values(self) -> u32 {
        self.conv_shapes()[2].output_rows() * CONV_CHANNELS
    }

    fn projected_values(self) -> u32 {
        self.projected_rows * self.d_model
    }

    fn hidden_values(self) -> u32 {
        self.num_rows * self.d_model
    }

    fn ffn_values(self) -> u32 {
        self.num_rows * self.ffn_dim
    }

    fn output_bytes(self) -> usize {
        self.num_rows as usize * self.output_dim as usize * BF16_BYTES
    }
}

fn load_bf16(
    device: &Device,
    store: &mut SafeTensorStore,
    name: &str,
    expected_shape: &[usize],
) -> Result<Buffer, ModelExecutorError> {
    Ok(Buffer::from_slice(
        device,
        &load_bf16_bytes(store, name, expected_shape)?,
    ))
}

fn load_bf16_bytes(
    store: &mut SafeTensorStore,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<u8>, ModelExecutorError> {
    let tensor = store.tensor_bytes(name, safetensors::Dtype::BF16)?;
    if tensor.shape() != expected_shape {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3-ASR tensor {name:?} has shape {:?}; expected {expected_shape:?}",
            tensor.shape()
        )));
    }
    Ok(tensor.into_data())
}

fn bf16_buffer(device: &Device, num_values: u32) -> Buffer {
    Buffer::new_uninit(device, num_values as usize * Dtype::Bfloat16.item_size())
}

fn as_u32(value: usize, name: &str) -> Result<u32, ModelExecutorError> {
    u32::try_from(value).map_err(|_| ModelExecutorError::custom(format!("Qwen3-ASR {name} must fit u32")))
}

#[cfg(test)]
#[path = "audio_test.rs"]
mod tests;
