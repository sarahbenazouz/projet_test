import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import time


def generate_corrected_sparse_tridiagonal_matrix(n, diagonal_value=5, off_diagonal_value=1):
    """
    Génère une matrice tridiagonale creuse et sa version dense.

    Args:
        n: Taille de la matrice.
        diagonal_value: Valeur des éléments diagonaux.
        off_diagonal_value: Valeur des éléments hors diagonale.

    Returns:
        As: Matrice creuse (csr_matrix).
        A_dense: Matrice dense (numpy.array).
        b: Vecteur du second membre (numpy.array).
    """
    # Diagonales principales et hors-diagonales
    main_diag = np.full(n, diagonal_value)
    off_diag = np.full(n - 1, off_diagonal_value)

    # Matrice creuse en CSR
    data = np.concatenate([main_diag, off_diag, off_diag])
    rows = np.concatenate([np.arange(n), np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n - 1)])
    As = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Matrice dense
    A_dense = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A_dense[i, i] = diagonal_value
            else:
                A_dense[i, j] = 3  # Valeur hors-diagonale

    # Vecteur b
    b = np.random.rand(n)

    return As, A_dense, b


def jacobi_dense(A, b, x0, tol=1e-6, max_iter=1000):
    """
    Méthode Jacobi pour matrices denses.
    """
    n = A.shape[0]
    x = x0.copy()
    errors = []

    start_time = time.time()

    for i in range(max_iter):
        x_new = np.zeros_like(x)
        for j in range(n):
            x_new[j] = (b[j] - np.dot(A[j, :], x) + A[j, j] * x[j]) / A[j, j]

        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)

        if error < tol:
            break

        x = x_new

    end_time = time.time()
    return x, i + 1, end_time - start_time


def jacobi_sparse(As, b, x0, tol=1e-7, max_iter=10000):
    """
    Méthode Jacobi pour matrices creuses.
    """
    n = As.shape[0]
    x = x0.copy()
    D = As.diagonal()
    L_U = As - sparse.diags(D)

    start_time = time.time()
    for i in range(max_iter):
        x_new = (b - L_U.dot(x)) / D
        error = np.linalg.norm(x_new - x, ord=np.inf)

        if error < tol:
            break

        x = x_new

    end_time = time.time()
    return x, i + 1, end_time - start_time


# Exemple d'utilisation
n = 100  # Taille de la matrice
x0 = np.zeros(n)  # Vecteur initial
A_sparse, A_dense, b = generate_corrected_sparse_tridiagonal_matrix(n)

# Jacobi dense
x_dense, iter_dense, time_dense = jacobi_dense(A_dense, b, x0)

# Jacobi creux
x_sparse, iter_sparse, time_sparse = jacobi_sparse(A_sparse, b, x0)

print(f"Jacobi (Dense): {iter_dense} itérations, temps = {time_dense:.4f} secondes")
print(f"Jacobi (Sparse): {iter_sparse} itérations, temps = {time_sparse:.4f} secondes")

# Comparaison pour différentes dimensions
dimensions = [10, 100, 1000, 5000]
results = []

for n in dimensions:
    x0 = np.zeros(n)
    A_sparse, A_dense, b = generate_corrected_sparse_tridiagonal_matrix(n)

    # Jacobi dense
    _, iter_dense, time_dense = jacobi_dense(A_dense, b, x0)

    # Jacobi creux
    _, iter_sparse, time_sparse = jacobi_sparse(A_sparse, b, x0)

    results.append((n, iter_dense, time_dense, iter_sparse, time_sparse))

# Résultats finaux
print("\nComparaison des méthodes Jacobi (Dense vs Sparse) :")
print(f"{'n':>10} | {'Itérations (Dense)':>20} | {'Temps (Dense)':>15} | {'Itérations (Sparse)':>20} | {'Temps (Sparse)':>15}")
print("-" * 85)
for n, iter_dense, time_dense, iter_sparse, time_sparse in results:
    print(f"{n:>10} | {iter_dense:>20} | {time_dense:>15.4f} | {iter_sparse:>20} | {time_sparse:>15.4f}")
