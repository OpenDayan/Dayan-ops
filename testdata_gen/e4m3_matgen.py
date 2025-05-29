M = 19
N = 38
K = 27

import torch

def export_matrix(name, matrix):
    matrix = matrix.flatten()
    with open(f"{name}.gen", "w") as f:
        for i in range(len(matrix)):
            f.write(f"{matrix[i]}")
            if i != len(matrix) - 1:
                f.write(",")

with open("generated/e4m3_mat_size.gen", "w") as f:
    f.write(f"#define M {M}\n")
    f.write(f"#define N {N}\n")
    f.write(f"#define K {K}\n")

import numpy as np
a = torch.randn(M, K).to(torch.float8_e4m3fn).to(torch.float32)
b = torch.randn(K, N).to(torch.float8_e4m3fn).to(torch.float32)
c = torch.matmul(a, b)
print(c.dtype)
export_matrix("generated/e4m3_a_col", a.T)
export_matrix("generated/e4m3_b_col", b.T)
export_matrix("generated/e4m3_c_col", c.T)
export_matrix("generated/e4m3_a_row", a)
export_matrix("generated/e4m3_b_row", b)
export_matrix("generated/e4m3_c_row", c)
