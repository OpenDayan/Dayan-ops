M = 64
N = 96
K = 128

def export_matrix(name, matrix):
    matrix = matrix.T.flatten()
    with open(f"{name}.gen", "w") as f:
        for i in range(len(matrix)):
            f.write(f"{matrix[i]}")
            if i != len(matrix) - 1:
                f.write(",")


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
export_matrix("a", a)
export_matrix("b", b)
export_matrix("c", c)