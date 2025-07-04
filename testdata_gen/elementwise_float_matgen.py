M = 64
N = 32

def export_matrix(name, matrix):
    matrix = matrix.flatten()
    with open(f"{name}.gen", "w") as f:
        for i in range(len(matrix)):
            f.write(f"{matrix[i]}")
            if i != len(matrix) - 1:
                f.write(",")

with open("generated/elementwise_float_size.gen", "w") as f:
    f.write(f"#define M {M}\n")
    f.write(f"#define N {N}\n")

import numpy as np
np.random.seed(0)

a = np.random.uniform(0.0, 200.0, size=(M, N)).astype(np.float32)
b = np.random.uniform(-100.0, 100.0, size=(M, N)).astype(np.float32)
c_add = a + b
c_sub = a - b
c_mul = a * b

c_max = np.max(a, axis=0)
c_min = np.min(a, axis=0)
c_sum = np.sum(a, axis=0)

export_matrix("generated/elementwise_float_a_row", a)
export_matrix("generated/elementwise_float_b_row", b)
export_matrix("generated/elementwise_float_c_add_row", c_add)
export_matrix("generated/elementwise_float_c_sub_row", c_sub)
export_matrix("generated/elementwise_float_c_mul_row", c_mul)

export_matrix("generated/elementwise_float_c_max_row", c_max)
export_matrix("generated/elementwise_float_c_min_row", c_min)
export_matrix("generated/elementwise_float_c_sum_row", c_sum)
