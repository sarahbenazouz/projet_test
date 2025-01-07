import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=10, off_diagonal_value=4):

    ### TODO: Fill your code here
    # Diagonales principales et hors-diagonales
    main_diag = np.full(n, diagonal_value)
    off_diag = np.full(n - 1, off_diagonal_value)

    # Matrice creuse en CSR
    data = np.concatenate([main_diag, off_diag, off_diag])
    rows = np.concatenate([np.arange(n), np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n - 1)])
    As = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    for i in range(n):
        A_dense[i, i] = diagonal_value

    b = np.random.rand(n)
    return As, A_dense, b

def generate_sparse_tridiagonal_matrix(n):
    ### TODO: Fill your code here.
    h=1/n+1
    main_diag = np.full(n, 2/h**2)
    off_diag = np.full(n - 1, -1/h**2)

    # Matrice creuse en CSR
    data = np.concatenate([main_diag, off_diag, off_diag])
    rows = np.concatenate([np.arange(n), np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n - 1)])
    As = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    
    # Right-hand side vector
    b = np.random.rand(n)

    return As,  A_dense, b


def jacobi_sparse_with_error(A, b, x0,x_exact, tol=1e-6, max_iter=1000):
    n=A.shape[0]
    errors = []
    x=x0.copy()
    x_new = x0.copy()
    D=A.diagonal()
    L_U=A-sparse.diags(D)
    for  i in range (max_iter):
        x_new=(b-L_U.dot(x))/D
        if np.linalg.norm(x_new-x,ord=np.inf)<tol:
            break
        x=x_new
    error = np.linalg.norm(x_new-x)
    errors.append(error)

    return x, i + 1, errors

def gauss_seidel_sparse_with_error(A, b, x0, x_exact, tol=1e-6, max_iter=1000):
    
  n = A.shape[0]
  x = x0.copy()
  errors = []
  for k in range(max_iter):
    x_new = np.zeros_like(x)
    for i in range(n):
            S1=np.dot(A[i,:i].toarray(),x_new[:i])
            S2=np.dot(A[i,i+1:].toarray(),x[i+1:])
            x_new[i] = (b[i] - S1 -S2) / A[i, i]
    error = np.linalg.norm(x_exact-x_new)
    errors.append(error)
    x = x_new
    if error < tol:
      break

  return x, i + 1, errors

### TODO: 
# Set up all the important parameters
# Set up all useful plotting tools
n=10
As,Ad,b = generate_sparse_tridiagonal_matrix(n)
x0=np.zeros(np.size(b))


# Jacobi spare
x_j, iter_j, error1 = jacobi_sparse_with_error(As, b, x0)

# GS sparse
x_g, iter_g, error2 = gauss_seidel_sparse_with_error(As, b, x0)

print(f"Jacobi (Sparse): {iter_j} itérations, erreur = {error1} ")
print(f"GS (Sparse) : {iter_g} itérations, erreur = {error2} ")

