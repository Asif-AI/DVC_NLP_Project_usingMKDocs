import numpy as nu
from scipy.sparse import csr_matrix

A = np.array([
    [1,0,0,0,0,1,0,],
    [0,1,0,0,0,1,0,],
    [0,0,0,0,1,0,0,],
])

print(A)

s = csr_matrix(A)
print(A)

