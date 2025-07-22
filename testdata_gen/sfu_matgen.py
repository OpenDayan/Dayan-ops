import numpy as np
import torch
import struct
import math

M = 64
N = 128

def export_matrix(name, matrix):
    matrix = matrix.flatten()
    with open(f"{name}.gen", "w") as f:
        for i in range(len(matrix)):
            f.write(f"{matrix[i]}")
            if i != len(matrix) - 1:
                f.write(",")

with open("generated/sfu_size.gen", "w") as f:
    f.write(f"#define M {M}\n")
    f.write(f"#define N {N}\n")

import struct
import numpy as np


np.random.seed(0)

a = np.random.uniform(1.0, 30.0, size=(M, N)).astype(np.float32)

sin_a = np.sin(a)
cos_a = np.cos(a)
inv_a = 1 / a
inv_sqrt_a = 1 / np.sqrt(np.abs(a))
exp_a = np.exp(a)

export_matrix("generated/sfu_float_a_row", a)
export_matrix("generated/sfu_a_recip_row", inv_a)
export_matrix("generated/sfu_a_rsqrt_row", inv_sqrt_a)
export_matrix("generated/sfu_a_sin_row", sin_a)
export_matrix("generated/sfu_a_cos_row", cos_a)
export_matrix("generated/sfu_a_exp_row", exp_a)