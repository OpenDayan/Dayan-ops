M = 16
N = 32

def export_matrix(name, matrix):
    matrix = matrix.flatten()
    with open(f"{name}.gen", "w") as f:
        for i in range(len(matrix)):
            f.write(f"{matrix[i]}")
            if i != len(matrix) - 1:
                f.write(",")

with open("generated/elementwise_int_size.gen", "w") as f:
    f.write(f"#define M {M}\n")
    f.write(f"#define N {N}\n")

import numpy as np
np.random.seed(0)
a = np.random.randint(-100, 100, size=(M, N), dtype=np.int32)
b = np.random.randint(-100, 100, size=(M, N), dtype=np.int32)
d = np.random.randint(0, 5, size=(M, N), dtype=np.int32)

c_add = a + b
c_sub = a - b
c_mul = a * b

def srl_s32(a, b):
    return ((a.astype(np.int64) & 0xFFFFFFFF) >> b).astype(np.int32)

def sra_s32(a, b):
    return np.right_shift(a, b)

def sll_s32(a, b):
    return np.left_shift(a, b)

c_srl = srl_s32(a, d)
c_sra = sra_s32(a, d)
c_sll = sll_s32(a, d)

c_and = a & b
c_or  = a | b
c_xor = a ^ b

export_matrix("generated/elementwise_int_a_row", a)
export_matrix("generated/elementwise_int_b_row", b)
export_matrix("generated/elementwise_int_shift_row", d)
export_matrix("generated/elementwise_int_c_add_row", c_add)
export_matrix("generated/elementwise_int_c_sub_row", c_sub)
export_matrix("generated/elementwise_int_c_mul_row", c_mul)
export_matrix("generated/elementwise_int_c_srl_row", c_srl)
export_matrix("generated/elementwise_int_c_sra_row", c_sra)
export_matrix("generated/elementwise_int_c_sll_row", c_sll)
export_matrix("generated/elementwise_int_c_and_row", c_and)
export_matrix("generated/elementwise_int_c_or_row", c_or)
export_matrix("generated/elementwise_int_c_xor_row", c_xor)
