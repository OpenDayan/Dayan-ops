M = 8
N = 16

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
a = np.random.rand(M, N)
a = (a * 256 - 128)
a = a.astype(np.int32)
b = np.random.rand(M, N)
b = (b * 256 - 128)
b = b.astype(np.int32)
c = a + b
c.astype(np.int32)
print(c.dtype)
export_matrix("generated/elementwise_int_a_row", a)
export_matrix("generated/elementwise_int_b_row", b)
export_matrix("generated/elementwise_int_c_row", c)
