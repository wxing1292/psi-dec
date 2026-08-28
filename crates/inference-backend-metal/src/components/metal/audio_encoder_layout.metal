#include <metal_stdlib>
using namespace metal;

kernel void audio_chunk_log_mel_f32_to_bf16(
    device const float* source [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& num_mel_bins [[buffer(2)]],
    constant uint& num_frames [[buffer(3)]],
    constant uint& frames_per_chunk [[buffer(4)]],
    constant uint& num_chunks [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const ulong total = (ulong)num_chunks * num_mel_bins * frames_per_chunk;
  if ((ulong)index >= total) return;
  const uint local_frame = index % frames_per_chunk;
  const ulong chunk_bin = index / frames_per_chunk;
  const uint bin = chunk_bin % num_mel_bins;
  const uint chunk = chunk_bin / num_mel_bins;
  const uint frame = chunk * frames_per_chunk + local_frame;
  output[index] = frame < num_frames ? bfloat(source[(ulong)bin * num_frames + frame]) : bfloat(0.0f);
}

kernel void audio_flatten_conv_bf16(
    device const bfloat* source [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const ulong total = (ulong)batch * width * channels * height;
  if ((ulong)index >= total) return;
  const uint height_index = index % height;
  const ulong channel_width_batch = index / height;
  const uint channel = channel_width_batch % channels;
  const ulong width_batch = channel_width_batch / channels;
  const uint width_index = width_batch % width;
  const uint batch_index = width_batch / width;
  const ulong source_index =
      (((ulong)batch_index * height + height_index) * width + width_index) * channels + channel;
  output[index] = source[source_index];
}

kernel void audio_compact_position_bf16(
    device const bfloat* source [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& num_rows [[buffer(2)]],
    constant uint& source_rows_per_chunk [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const ulong total = (ulong)num_rows * hidden_dim;
  if ((ulong)index >= total) return;
  const uint dim = index % hidden_dim;
  const uint row = index / hidden_dim;
  const uint chunk = row / 13u;
  const uint local_row = row - chunk * 13u;
  const ulong source_index = ((ulong)chunk * source_rows_per_chunk + local_row) * hidden_dim + dim;
  const uint half_dim = hidden_dim / 2u;
  const uint frequency_index = dim < half_dim ? dim : dim - half_dim;
  const float increment = precise::log(10000.0f) / float(half_dim - 1u);
  const float phase = float(local_row) * precise::exp(-increment * float(frequency_index));
  const float position = dim < half_dim ? precise::sin(phase) : precise::cos(phase);
  output[index] = bfloat(float(source[source_index]) + position);
}
