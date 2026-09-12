constant ushort FP8_E4M3_SUBNORMAL_BF16_BITS[8] = {
    0x0000, 0x3b00, 0x3b80, 0x3bc0, 0x3c00, 0x3c20, 0x3c40, 0x3c60,
};

inline ushort fp8_e4m3_to_bf16_bits(uchar bits) {
    const ushort sign = ushort(bits & uchar(0x80)) << 8;
    const ushort exponent = (ushort(bits) >> 3) & ushort(0x0f);
    const ushort mantissa = ushort(bits) & ushort(0x07);
    const ushort normal = sign | ushort((exponent + 120) << 7) | ushort(mantissa << 4);
    const ushort subnormal = sign | FP8_E4M3_SUBNORMAL_BF16_BITS[mantissa];
    const ushort finite = select(normal, subnormal, exponent == 0);
    return select(finite, ushort(sign | 0x7fc0), exponent == 15 && mantissa == 7);
}

inline uint fp8_e4m3x2_to_bf16x2(uint bits) {
    return uint(fp8_e4m3_to_bf16_bits(uchar(bits)))
        | (uint(fp8_e4m3_to_bf16_bits(uchar(bits >> 8))) << 16);
}

inline uint4 fp8_e4m3x8_to_bf16x8(uint2 bits) {
    return uint4(
        fp8_e4m3x2_to_bf16x2(bits.x),
        fp8_e4m3x2_to_bf16x2(bits.x >> 16),
        fp8_e4m3x2_to_bf16x2(bits.y),
        fp8_e4m3x2_to_bf16x2(bits.y >> 16)
    );
}
