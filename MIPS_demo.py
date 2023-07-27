import numpy as np

# Define the data matrix (D x N)
matrix_x = np.array([[1, 4, 3, 5, 2],
                     [6, 9, 8, 10, 7],
                     [11, 14, 13, 15, 12]])

# Define the query matrix (D x Q)
matrix_q = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])

d, n = matrix_x.shape
_, Q = matrix_q.shape

matrix_x_t = np.ascontiguousarray(matrix_x.T)
matrix_q_t = np.ascontiguousarray(matrix_q.T)
np.random.seed(123)
RANDOM = np.random.normal(0, 1, (d, 1000))  # Generate N(0, 1)
L = np.matmul(matrix_x_t, RANDOM)  # Compute the inner products
# Step 4: Find the closest r_i for each query in QUERY
closest_indices = np.argmax(np.matmul(matrix_q_t, RANDOM), axis=1)

# Step 5: Find the top MIPS based on the inner products order of r1
k = 2
topk_MIPS = []
for i in range(Q):
    closest_index = closest_indices[i]  # Index of the closest r_i for the current query
    # Find the index of the top MIPS based on the inner product order of r_i
    # top1_index = np.argmax(L[:, closest_index])
    topk_indices = np.argpartition(L[:, closest_index], -k)[-k:]

    # Sort topk_indices, make sure the first index is always the largest one, and so on
    topk_indices = topk_indices[np.argsort(-L[topk_indices, closest_index])]
    topk_MIPS.append(topk_indices)

print(topk_MIPS)
