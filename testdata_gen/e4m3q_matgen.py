M = 1536
N = 2
K = 1536
QK = 32

import torch

def export_matrix(name, matrix):
    matrix = matrix.flatten()
    with open(f"{name}.gen", "w") as f:
        for i in range(len(matrix)):
            f.write(f"{matrix[i]}")
            if i != len(matrix) - 1:
                f.write(",")

with open("generated/e4m3q_mat_size.gen", "w") as f:
    f.write(f"#define M {M}\n")
    f.write(f"#define N {N}\n")
    f.write(f"#define K {K}\n")
    f.write(f"#define QK {QK}\n")

def conv(input):
    scale = torch.zeros((input.shape[0], input.shape[1] // QK), dtype=torch.float32)
    output = torch.zeros_like(input, dtype=torch.float8_e4m3fn)
    for row_id in range(input.shape[0]):
        row = input[row_id]
        for q in range(len(row) // QK):
            partial = row[q * QK:(q + 1) * QK]
            m = torch.abs(row[q * QK:(q + 1) * QK])
            m = m.max()
            d = m/448
            partial = 448 * partial / m
            scale[row_id, q] = d
            output[row_id, q * QK:(q + 1) * QK] = partial.to(torch.float8_e4m3fn)
            # print(partial.to(torch.float8_e4m3fn))
    return scale, output

def e4m3q_matmul(a, b, a_scale, b_scale):
    # a[m, k]
    # b[n, k]
    assert(a.shape[1] == b.shape[1])
    # print(a_scale, b_scale)
    result = torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    for ki in range(a.shape[1] // QK):
        a_part = a[:, ki * QK:(ki + 1) * QK]
        b_part = b[:, ki * QK:(ki + 1) * QK]
        a_scale_part = a_scale[:, ki]
        b_scale_part = b_scale[:, ki]
        pr = torch.matmul(a_part.to(torch.float32), b_part.T.to(torch.float32))
        s = torch.outer(a_scale_part, b_scale_part)
        # print(pr)
        result += pr * s
    return result

import numpy as np
torch.manual_seed(0)
a = torch.randn(M, K)
a_scale, a_data = conv(a)
b = torch.randn(K, N)
b_scale, b_data = conv(b.T)
c = torch.matmul(a, b)
c_q = e4m3q_matmul(a_data, b_data, a_scale, b_scale)
# print(c)
# print(c_q)
print(torch.max((c_q - c).abs()))
export_matrix("generated/e4m3q_a_row", a_data)
export_matrix("generated/e4m3q_b_col", b_data)
export_matrix("generated/e4m3q_c_col", c_q.T)
export_matrix("generated/e4m3q_bs_col", b_scale)
export_matrix("generated/e4m3q_as_row", a_scale)
