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
a = np.random.randint(0, 200, size=(M, N), dtype=np.int32)
b = np.random.randint(-100, 100, size=(M, N), dtype=np.int32)
c_add = a + b
c_sub = a - b
c_mul = a * b
export_matrix("generated/elementwise_int_a_row", a)
export_matrix("generated/elementwise_int_b_row", b)
export_matrix("generated/elementwise_int_c_add_row", c_add)
export_matrix("generated/elementwise_int_c_sub_row", c_sub)
export_matrix("generated/elementwise_int_c_mul_row", c_mul)
