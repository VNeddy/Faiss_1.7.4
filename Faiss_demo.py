import numpy as np
import faiss

# Replace the file reading with direct data
DATA = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
], dtype='float32')  # A 3 x 4 matrix

QUERY = np.array([
    [0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9]
], dtype='float32')  # A 2 x 4 matrix

(n, d) = DATA.shape

dim, param = 4, 'Flat'
measure = faiss.METRIC_INNER_PRODUCT  # inner product
# measure = faiss.METRIC_L2  # Euclidean distance
index = faiss.index_factory(dim, param, measure)
index.add(DATA)
k = 1
D, I = index.search(QUERY, k)     # search

for i, query in enumerate(QUERY):
    print(f"For query {query}, the nearest vector in the dataset is {DATA[I[i, 0]]}")
