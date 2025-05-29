M = 169
N = 269
K = 369

def export_matrix(name, matrix):
    matrix = matrix.flatten()
    with open(f"{name}.gen", "w") as f:
        for i in range(len(matrix)):
            f.write(f"{matrix[i]}")
            if i != len(matrix) - 1:
                f.write(",")

with open("generated/int_mat_size.gen", "w") as f:
    f.write(f"#define M {M}\n")
    f.write(f"#define N {N}\n")
    f.write(f"#define K {K}\n")

import numpy as np
np.random.seed(0)
a = np.random.rand(M, K)
a = (a * 256 - 128)
a = a.astype(np.int8)
a = a.astype(np.int32)
b = np.random.rand(K, N)
b = (b * 256 - 128)
b = b.astype(np.int8)
a = a.astype(np.int32)
c = a @ b
c.astype(np.int32)
print(c.dtype)
export_matrix("generated/int_a_col", a.T)
export_matrix("generated/int_b_col", b.T)
export_matrix("generated/int_c_col", c.T)
export_matrix("generated/int_a_row", a)
export_matrix("generated/int_b_row", b)
export_matrix("generated/int_c_row", c)
