import numpy as np

DATA = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
])

QUERY = np.array([
    [0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9]
])

n, d = DATA.shape
Q, _ = QUERY.shape

RANDOM = np.random.normal(0, 1, (d, 1000))  # Generate N(0, 1)
L = np.matmul(DATA, RANDOM)  # Compute the inner products

# Step 4: Find the closest r_i for each query in QUERY
closest_indices = np.argmax(np.matmul(QUERY, RANDOM), axis=1)  # Find the indices of the closest r_i for each query

# Step 5: Find the top MIPS based on the inner products order of r1
top1_MIPS = []
for i in range(Q):
    closest_index = closest_indices[i]  # Index of the closest r_i for the current query
    top1_index = np.argmax(L[:, closest_index])  # Find the index of the top MIPS based on the inner product order of r_i
    top1_MIPS.append(DATA[top1_index])

for i in range(Q):
    print(f"For query {QUERY[i]}, the nearest vector in the dataset is {top1_MIPS[i]}")
