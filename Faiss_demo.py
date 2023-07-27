import numpy as np
import faiss

# Define the data matrix (D x N)
matrix_x = np.array([[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15]])

matrix_x = np.array([[1, 4, 3, 5, 2],
                     [6, 9, 8, 10, 7],
                     [11, 14, 13, 15, 12]])

# Define the query matrix (D x Q)
matrix_q = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])

# Transpose the data matrix to N x D and Q x D because FAISS expects this shape
matrix_x_t = np.ascontiguousarray(matrix_x.T)
matrix_q_t = np.ascontiguousarray(matrix_q.T)

dim, param = matrix_x_t.shape[1], 'Flat'
measure = faiss.METRIC_INNER_PRODUCT  # inner product
# measure = faiss.METRIC_L2  # Euclidean distance
index = faiss.index_factory(dim, param, measure)
index.add(matrix_x_t)
k = 2
D, I = index.search(matrix_q_t, k)  # search

print(I.T)
