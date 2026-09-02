//! GQA-local FP8 E4M3FN storage conversion.
//!
//! GQA uses one fixed unit scale for K/V cache values. The conversion rounds
//! to nearest with ties to even. It saturates finite and infinite inputs to
//! `+/-448`. It preserves NaN as the E4M3FN NaN encoding.

use half::bf16;

pub const MAX_FINITE: f32 = 448.0;

pub fn f32_to_bf16(value: f32) -> bf16 {
    bf16::from_f32(value)
}

pub fn bf16_to_f32(value: bf16) -> f32 {
    value.to_f32()
}

pub fn bf16_to_fp8_e4m3(value: bf16) -> u8 {
    let value = bf16_to_f32(value);
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return sign | 0x7f;
    }
    if !magnitude.is_finite() || magnitude >= MAX_FINITE {
        return sign | 0x7e;
    }
    if magnitude < 2.0_f32.powi(-6) {
        let mantissa = round_ties_even(magnitude * 512.0) as u8;
        return sign | mantissa.min(8);
    }
    let mut exponent = magnitude.log2().floor() as i32;
    let mut mantissa = round_ties_even((magnitude * 2.0_f32.powi(-exponent) - 1.0) * 8.0) as u8;
    if mantissa == 8 {
        exponent += 1;
        mantissa = 0;
    }
    let encoded_exponent = exponent + 7;
    if encoded_exponent > 15 || (encoded_exponent == 15 && mantissa >= 7) {
        return sign | 0x7e;
    }
    sign | ((encoded_exponent as u8) << 3) | mantissa
}

pub fn fp8_e4m3_to_bf16(bits: u8) -> bf16 {
    let exponent = u32::from((bits >> 3) & 0x0f);
    let mantissa = u32::from(bits & 0x07);
    if exponent == 15 && mantissa == 7 {
        return f32_to_bf16(f32::NAN.copysign(if bits & 0x80 == 0 { 1.0 } else { -1.0 }));
    }
    let magnitude = if exponent == 0 {
        mantissa as f32 / 512.0
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
    };
    f32_to_bf16(if bits & 0x80 == 0 { magnitude } else { -magnitude })
}

fn round_ties_even(value: f32) -> f32 {
    value.round_ties_even()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e4m3fn_fixed_encodings_and_saturation() {
        let cases = [
            (0.0, 0x00),
            (-0.0, 0x80),
            (2.0_f32.powi(-9), 0x01),
            (2.0_f32.powi(-6), 0x08),
            (1.0, 0x38),
            (1.5, 0x3c),
            (448.0, 0x7e),
            (-448.0, 0xfe),
            (449.0, 0x7e),
            (-449.0, 0xfe),
            (f32::INFINITY, 0x7e),
            (f32::NEG_INFINITY, 0xfe),
        ];
        for (value, expected) in cases {
            assert_eq!(bf16_to_fp8_e4m3(f32_to_bf16(value)), expected, "value={value}");
        }
        assert_eq!(bf16_to_fp8_e4m3(f32_to_bf16(f32::NAN)) & 0x7f, 0x7f);
    }

    #[test]
    fn test_e4m3fn_rounds_ties_to_even() {
        assert_eq!(
            bf16_to_fp8_e4m3(f32_to_bf16(1.0625)),
            bf16_to_fp8_e4m3(f32_to_bf16(1.0))
        );
        assert_eq!(
            bf16_to_fp8_e4m3(f32_to_bf16(1.1875)),
            bf16_to_fp8_e4m3(f32_to_bf16(1.25))
        );
    }

    #[test]
    fn test_e4m3fn_exhaustive_finite_round_trip() {
        for bits in u8::MIN..=u8::MAX {
            if bits & 0x7f == 0x7f {
                continue;
            }
            assert_eq!(bf16_to_fp8_e4m3(fp8_e4m3_to_bf16(bits)), bits);
        }
    }

    #[test]
    fn test_e4m3fn_quantization_error_bounds() {
        let min_normal = 2.0_f32.powi(-6);
        for sample in 0..=4096 {
            let value = min_normal * sample as f32 / 4096.0;
            let source = f32_to_bf16(value);
            let quantized = bf16_to_f32(fp8_e4m3_to_bf16(bf16_to_fp8_e4m3(source)));
            let absolute_error = (quantized - bf16_to_f32(source)).abs();
            assert!(
                absolute_error <= 2.0_f32.powi(-10),
                "subnormal value={value} absolute_error={absolute_error}"
            );
        }
        for sample in 0..=458_736 {
            let value = min_normal + sample as f32 / 1024.0;
            if value > MAX_FINITE {
                break;
            }
            let source = f32_to_bf16(value);
            let source_f32 = bf16_to_f32(source);
            let quantized = bf16_to_f32(fp8_e4m3_to_bf16(bf16_to_fp8_e4m3(source)));
            let relative_error = (quantized - source_f32).abs() / source_f32;
            assert!(
                relative_error <= 0.0625,
                "normal value={value} relative_error={relative_error}"
            );
        }
    }
}
