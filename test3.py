import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import time


def generate_corrected_sparse_tridiagonal_matrix(n, diagonal_value=5, off_diagonal_value=1):
    """
    Generates a sparse tridiagonal matrix, ensuring no overlaps.

    Args:
        n: Dimension of the system (size of the matrix A).
        diagonal_value: Value for the diagonal elements.
        off_diagonal_value: Value for the off-diagonal elements.

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    ### TODO: Complete code here 
    # Main diagonal
    main_diag = np.full(n, diagonal_value+off_diagonal_value)

    # Construct sparse matrix
    data = main_diag
    #print(data)
    rows = np.arange(n) # might need to use np.concatenate
    #print(rows)
    cols = np.arange(n)
    #print(cols)
    As = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    for i in range(n):
        for j in range (n):
            if i == j:
                A_dense[i, i] = diagonal_value
            else:
                A_dense[i,j] = random.randint(-100,100)

    b = np.zeros_like(x0)
    return As, A_dense, b



def jacobi_dense(A, b, x0, tol=1e-6, max_iter=1000):
    n = A.shape[0]
    x = x0.copy()
    errors = []
    start_time = time.time()

    for i in range(max_iter):
        x_new = x.copy()
        for j in range(n):
            x_new[j] = (b[j] - np.dot(A[j, :], x) + A[j, j] * x[j]) / A[j, j]
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        x = x_new.copy()
        if error < tol:
            break

    end_time = time.time()
    time_taken = end_time - start_time

    return x, i + 1, time_taken, errors


def jacobi_sparse(A, b, x0, tol=1e-7, max_iter=10000):

    n=A.shape[0]
    x=x0.copy()
    x_new = x0.copy()
    D=A.diagonal()
    L_U=A-sparse.diag(D)
    for  i in range (max_iter):
        x_new=(b-L_U.dot(x))/D
        if np.linalg.norm(x_new-x,ord=np.inf)<tol:
            break
        x=x_new
    start_time = time.time()
    end_time = time.time()
    time_taken = end_time - start_time
    return x_new, 1, time_taken

# Example usage:
n=10000 
x0 = np.zeros(n)  ## initial guess
A_sparse, A_dense_v1, b = generate_corrected_sparse_tridiagonal_matrix(n) 
A_dense_v2 = A_sparse.toarray()  # Convert to dense format for comparison





# Classical Jacobi (dense)
x_dense, iter_dense, time_dense = jacobi_dense(A_dense_v2, b, x0) 

# Jacobi for sparse matrix
x_sparse, iter_sparse, time_sparse = jacobi_sparse(A_sparse, b, x0)

print(f"Iterations (dense): {iter_dense}, Time (dense): {time_dense:.4f} seconds")
print(f"Iterations (sparse): {iter_sparse}, Time (sparse): {time_sparse:.4f} seconds")

#x_exact = np.linalg.solve(A_dense_v2, b)
#print(x_exact)
#print(x_sparse)

### TODO: 
# Implement a small for loop comparing the times required for both approaches as a function of the dimension n
