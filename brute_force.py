import numpy as np
import heapq
import time


def bf_topk_mips(matrix_x, matrix_q, top_k, data_n, query_q, save_output=False):
    # Initialize the result matrix
    mat_topk = np.zeros((top_k, query_q), dtype=int)

    # For each query
    for q in range(query_q):
        # Record the start time
        start_time = time.time()

        # Initialize the priority queue
        que_topk = []

        # Compute the inner product of the query vector and the data matrix
        vec_query = matrix_q[:, q]
        vec_res = vec_query.T @ matrix_x

        # Insert the inner product results into the priority queue
        for n in range(data_n):
            f_value = vec_res[n]

            # If we do not have enough top K points, insert directly
            if len(que_topk) < top_k:
                heapq.heappush(que_topk, (f_value, n))
            else:
                # If the new value is larger than the smallest value in the queue,
                # replace the smallest value with the new value
                if f_value > que_topk[0][0]:
                    heapq.heapreplace(que_topk, (f_value, n))

        # Save the results into the result matrix
        for k in range(top_k - 1, -1, -1):
            mat_topk[k, q] = heapq.heappop(que_topk)[1]

    # Print the time
    print(f"BF Querying Time in ms is {(time.time() - start_time) * 1000}")

    # Save the result if needed
    if save_output:
        np.savetxt(f"BF_TopK_{top_k}.txt", mat_topk, fmt="%i")

    return mat_topk


# Define the data matrix (D x N)
matrix_x = np.array([[1, 4, 3, 5, 2],
                     [6, 9, 8, 10, 7],
                     [11, 14, 13, 15, 12]])

# Define the query matrix (D x Q)
matrix_q = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])

# Number of data points
data_n = matrix_x.shape[1]
# Number of queries
query_q = 2
# Number of top MIPS entries to retrieve
top_k = 2

# Run the function
mat_topk = bf_topk_mips(matrix_x, matrix_q, top_k, data_n, query_q)

print(mat_topk)
