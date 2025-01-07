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
        for j in range(n):
            if i == j:
                A_dense[i, i] = main_diag
            if i > 0:
                A_dense[i, i-1] = off_diag
            if i < n-1:
                A_dense[i, i+1] = off_diag

    b = np.random.rand(n)
    return As, A_dense, b

def generate_sparse_tridiagonal_matrix(n):
    ### TODO: Fill your code here.
    h=1/(n+1)
    main_diag = np.full(n, 2/h**2)
    off_diag = np.full(n - 1, -1/h**2)

    # Matrice creuse en CSR
    data = np.concatenate([main_diag, off_diag, off_diag])
    rows = np.concatenate([np.arange(n), np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n - 1)])
    As = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A_dense[i, i] = 2/h**2
            if i > 0:
                A_dense[i, i-1] = -1/h**2
            if i < n-1:
                A_dense[i, i+1] = -1/h**2
    
    # Right-hand side vector
    b = np.random.rand(n)

    return As,  A_dense, b

eps=10E-7
max=1000


def jacobi_sparse_with_error(A, b, x0, x_exact, tol=eps, max_iter=max):
    n = A.shape[0]
    errors = []
    x = x0.copy()
    D = A.diagonal()
    L_U = A - sparse.diags(D)
    for i in range(max_iter):
        x_new = (b - L_U.dot(x)) / D
        error = np.linalg.norm(x_new - x_exact,ord=np.inf)
        errors.append(error)
        x = x_new
        if error < tol:
            break
        x = x_new
    return x, i + 1, errors

def gauss_seidel_sparse_with_error(A, b, x0, x_exact,tol=eps, max_iter=max):
  n = A.shape[0]
  x = x0.copy()
  errors = []
  for k in range(max_iter):
    x_new = np.zeros_like(x)
    for i in range(n):
        S1=np.dot(A[i,:i].toarray(),x_new[:i]).item()
        S2=np.dot(A[i,i+1:].toarray(),x[i+1:]).item()
        x_new[i] = (b[i] - S1 -S2) / A[i, i]
    error = np.linalg.norm(x_new-x_exact,ord=np.inf)
    errors.append(error)
    x = x_new
    if error < tol:
        return x, k + 1, errors


omega= 1.6

def sor_with_error(A, b, x0, x_exact,omega=omega, tol=eps, max_iter=max):
  n = A.shape[0]
  x = x0.copy()
  errors = []
  for k in range(max_iter):
    x_new = np.zeros_like(x)
    for i in range(n):
        S1=np.dot(A[i,:i].toarray(),x_new[:i])
        S2=np.dot(A[i,i+1:].toarray(),x[i+1:])
        x_new[i] = omega * ((b[i] - S1 - S2) / A[i, i]) + (1 - omega) * x[i]
    error = np.linalg.norm(x_new-x_exact,ord=np.inf)
    errors.append(error)
    x = x_new
    if error < tol:
      return x, k + 1, errors

  

#omega: 1.6 ou 1.7 . tester avec omega=1= courbes gs et sor doivent etre la même 
### TODO: 
# Set up all the important parameters
# Set up all useful plotting tools

n=5


As,Ad,b = generate_sparse_tridiagonal_matrix(n)
x0=np.zeros(np.size(b))
print(x0)

eps=10E-7
max_iter=10000
print("matrice trian:")
print(As)
print("matrice dense:")
print(Ad)
x_exact = np.linalg.solve(Ad, b)

# Jacobi spare
x_j, iter_j, error1 = jacobi_sparse_with_error(As, b, x0,x_exact,eps,max)

# GS sparse
x_g, iter_g, error2 = gauss_seidel_sparse_with_error(As, b, x0,x_exact,eps,max)

#SOR
x_s, iter_s, error3 = sor_with_error(As, b, x0, x_exact,omega,eps,max)


print(f"Jacobi (Sparse): {iter_j} itérations, erreur = {error1} ")
print(f"GS (Sparse) : {iter_g} itérations, erreur = {error2} ")
print(f"iteration jacobi:{iter_j}")
print(f"iteration GS:{iter_g}")
print(f"iteration SOR:{iter_s}")


def plot_error_convergence(error1, error2,error3, iter_j, iter_g,iter_s):
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(error1) + 1), error1, 'b-', label='Jacobi')
    plt.semilogy(range(1, len(error2) + 1), error2, 'r-', label='Gauss-Seidel')
    plt.semilogy(range(1, len(error3) + 1), error3, 'g-', label='SOR')
    plt.xlabel('Iterations')
    plt.ylabel('Error (log scale)')
    plt.title(f'Error Convergence: Jacobi vs Gauss-Seidel with n={n}')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_error_convergence(error1, error2,error3, iter_j, iter_g,iter_s)
