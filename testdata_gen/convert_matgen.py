import numpy as np
import torch
import struct
import math

M = 8
N = 10

def export_matrix(name, matrix):
    matrix = matrix.flatten()
    with open(f"{name}.gen", "w") as f:
        for i in range(len(matrix)):
            f.write(f"{matrix[i]}")
            if i != len(matrix) - 1:
                f.write(",")

with open("generated/convert_size.gen", "w") as f:
    f.write(f"#define M {M}\n")
    f.write(f"#define N {N}\n")

def s32_to_fp32(s32_val):
    return float(s32_val)
def fp32_to_fp16_with_mode(f32_val, mode='round'):
    f32_bytes = struct.pack('!f', f32_val)
    x = struct.unpack('!I', f32_bytes)[0]
    
    sign = x & 0x80000000
    exponent = ((x >> 23) & 0xFF) - 127
    mantissa = x & 0x007FFFFF

    if (x & 0x7F800000) == 0x7F800000:
        return (sign >> 16) | 0x7C00 | ((mantissa >> 13) & 0x03FF)

    if exponent > 15:
        return (sign >> 16) | 0x7C00

    if exponent < -14:
        shift = -14 - exponent
        if shift > 24:
            return sign >> 16
        
        mantissa |= 0x00800000
        mantissa >>= shift

        round_bit = 1 << (shift - 1)
        sticky_bit = 1 if (mantissa & ((1 << (shift - 1)) - 1)) else 0
        
        if mode == 'round':
            mantissa += (round_bit & (mantissa | sticky_bit))
        elif mode == 'ceil':
            if not sign:
                mantissa += (round_bit | sticky_bit)
        elif mode == 'floor':
            if sign:
                mantissa += (round_bit | sticky_bit)
        
        return (sign >> 16) | (mantissa >> 13)

    fp16_mantissa = mantissa >> 13
    round_bit = (mantissa >> 12) & 1
    sticky_bit = 1 if (mantissa & 0xFFF) else 0

    if mode == 'round':
        fp16_mantissa += (round_bit & (fp16_mantissa | sticky_bit))
    elif mode == 'ceil':
        if not sign:
            fp16_mantissa += (round_bit | sticky_bit)
    elif mode == 'floor':
        if sign:
            fp16_mantissa += (round_bit | sticky_bit)

    if fp16_mantissa > 0x03FF:
        exponent += 1
        fp16_mantissa = 0

    return (sign >> 16) | (((exponent + 15) << 10) & 0x7C00) | (fp16_mantissa & 0x03FF)

def fp32_to_bf16_with_mode(f32_val, mode='round'):
    f32_bytes = struct.pack('!f', f32_val)
    x = struct.unpack('!I', f32_bytes)[0]
    
    sign = x & 0x80000000
    exponent = (x >> 23) & 0xFF
    mantissa = x & 0x007FFFFF

    if exponent == 0xFF:
        return (x >> 16) | 0x007F
    if exponent == 0:
        return (x >> 16) & 0x8000

    truncated = x >> 16
    round_bit = (x >> 15) & 1
    sticky_bits = x & 0x7FFF

    if mode == 'round':
        if round_bit and (sticky_bits or (truncated & 1)):
            truncated += 1
    elif mode == 'ceil':
        if not sign and (round_bit or sticky_bits):
            truncated += 1
    elif mode == 'floor':
        if sign and (round_bit or sticky_bits):
            truncated += 1

    if (truncated & 0x7F80) == 0x7F80:
        truncated = (truncated & 0x8000) | 0x7F80

    return truncated & 0xFFFF

import struct
import numpy as np

def fp32_to_fp8_e4m3_with_mode(f32_val, mode='round'):
    x = struct.unpack('>I', struct.pack('>f', f32_val))[0]
    
    sign = (x >> 31) & 0x1
    exponent = ((x >> 23) & 0xFF) - 127
    mantissa = x & 0x007FFFFF

    if (x & 0x7F800000) == 0x7F800000:
        return 0xFF if sign else 0x7F

    if exponent < -127:
        return sign << 7

    e4m3_exponent = exponent + 7
    e4m3_mantissa = (mantissa >> 20) & 0x7

    if e4m3_exponent <= 0:
        return sign << 7

    if e4m3_exponent >= 15:
        return (sign << 7) | 0x7F

    round_bit = (mantissa >> 19) & 0x1
    sticky_bits = 1 if (mantissa & 0x0007FFFF) else 0

    result = (sign << 7) | (e4m3_exponent << 3) | e4m3_mantissa

    if mode == 'round':
        if round_bit and (sticky_bits or (e4m3_mantissa & 1)):
            result += 1
    elif mode == 'ceil':
        if not sign and (round_bit or sticky_bits):
            result += 1
    elif mode == 'floor':
        if sign and (round_bit or sticky_bits):
            result += 1

    if (result & 0x7F) == 0x7F:
        result = (sign << 7) | 0x7F

    return result & 0xFF

np.random.seed(0)

a = np.random.uniform(-100.0, 100.0, size=(M, N)).astype(np.float32)

a_floor = np.floor(a).astype(np.int32)
a_ceil = np.ceil(a).astype(np.int32)
a_round = np.round(a).astype(np.int32)

a_fp16_round = np.array([fp32_to_fp16_with_mode(x, 'round') for x in a.flatten()]).reshape(a.shape)
a_bf16_round = np.array([fp32_to_bf16_with_mode(x, 'round') for x in a.flatten()]).reshape(a.shape)
a_fp8_e4m3_round = np.array([fp32_to_fp8_e4m3_with_mode(x, 'round') for x in a.flatten()]).reshape(a.shape)

a_floor_to_fp32 = np.array([s32_to_fp32(x) for x in a_floor.flatten()]).reshape(a.shape)

export_matrix("generated/convert_float_a_row", a)
export_matrix("generated/convert_a_floor_row", a_floor)
export_matrix("generated/convert_a_ceil_row", a_ceil)
export_matrix("generated/convert_a_round_row", a_round)
export_matrix("generated/convert_a_fp16_round_row", a_fp16_round)
export_matrix("generated/convert_a_bf16_round_row", a_bf16_round)
export_matrix("generated/convert_a_fp8_e4m3_round_row", a_fp8_e4m3_round)
export_matrix("generated/convert_a_floor_to_fp32_row", a_floor_to_fp32)
