#include <metal_stdlib>
using namespace metal;

kernel void conv2d_unfold_3x3_stride2_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& input_height [[buffer(3)]],
    constant uint& input_width [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& output_height [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    uint value_index [[thread_position_in_grid]]) {
  const ulong output_rows = (ulong)batch * output_height * output_width;
  const uint actual_patch_values = 9u * channels;
  const uint patch_values = (actual_patch_values + 15u) & ~15u;
  const ulong total_values = output_rows * patch_values;
  if ((ulong)value_index >= total_values) return;
  const uint patch_index = value_index % patch_values;
  if (patch_index >= actual_patch_values) {
    output[value_index] = bfloat(0.0f);
    return;
  }
  const ulong output_row = value_index / patch_values;
  const uint channel = patch_index % channels;
  const uint kernel_index = patch_index / channels;
  const int kernel_y = (int)(kernel_index / 3u);
  const int kernel_x = (int)(kernel_index % 3u);
  const uint output_x = output_row % output_width;
  const ulong batch_y = output_row / output_width;
  const uint output_y = batch_y % output_height;
  const uint batch_index = batch_y / output_height;
  const int input_y = (int)(output_y * 2u) + kernel_y - 1;
  const int input_x = (int)(output_x * 2u) + kernel_x - 1;
  if (input_y < 0 || input_y >= (int)input_height || input_x < 0 || input_x >= (int)input_width) {
    output[value_index] = bfloat(0.0f);
    return;
  }
  const ulong input_index =
      (((ulong)batch_index * input_height + (uint)input_y) * input_width + (uint)input_x) * channels + channel;
  output[value_index] = input[input_index];
}
