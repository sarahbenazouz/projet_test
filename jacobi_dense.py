import numpy as np
import matplotlib.pyplot as plt

def generate_linear_system(n):
    A = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, i] = 5 * (i + 1)  # Set diagonal elements
                #A[i, i] = 10 * n # Set diagonal dominance
            else:
                A[i, j] = -1  # Set off-diagonal elements
    
    # Generate random vector b
    b = np.random.rand(n)

    return A, b

   
A0 = np.array([[2,-1],[-1,2]])
A1 = np.array([[3,0,4],[7,4,2],[-1,1,2]])
A2 = np.array([[-3,3,-6],[-4,7,-8],[5,7,-9]])
A3 = np.array([[4,1,1],[2,-9,0],[0,-8,-6]])
A4 = np.array([[7,6,9],[4,5,-4],[-7,-3,8]])


def jacobi_method(A, b, x0, tol=1e-5, max_iter=1000):

  n = A.shape[0]
  x = x0.copy()
  errors = []
  for i in range(max_iter):
    x_new = np.zeros_like(x)
    for j in range(n):
            x_new[j] = (b[j] - np.dot(A[j, :], x) + A[j, j]* x[j]) / A[j, j]
    error = np.linalg.norm(x_new-x)
    errors.append(error)
    x = x_new
    if error < tol:
      break

  return x, i + 1, errors

#Booleen qui test si la matrice est à diagonale dominante
def diagonale_dominance(A):
    n = A.shape[0]
    for i in range(n):
        # Somme des éléments hors diagonale sur la ligne i
        sum_non_diag = sum(abs(A[i, j]) for j in range(n) if j != i)
        
        # Vérifier la condition de dominance diagonale
        if abs(A[i, i]) < sum_non_diag:
            return False  # Si une ligne ne respecte pas la condition, ce n'est pas dominant
    
    return True  # Toutes les lignes respectent la condition

def rayon_spectral(A):
    vp=np.linalg.eigvals(A)
    p=max(abs(vp))
    return p

def plot_error(errors, iterations):
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations), errors, marker='o', linestyle='-')
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations (Jacobi Method)")
    plt.grid(True)
    plt.show()


# Example usage:
n = 3
M, b = generate_linear_system(n)  # Generate a linear system
x0 = np.zeros(np.size(b))


A=M
#test domination
print(diagonale_dominance(A))

#test rayon spectral
print(rayon_spectral(A))
# Solve using Jacobi method
x_jacobi, iterations, errors = jacobi_method(A, b, x0)


# Calculate exact solution
x_exact = np.linalg.solve(A, b)

# Print results
print(f"Iterations: {iterations}")
print(f"Solution Jacobi: {x_jacobi}")
print(f"Exact solution: {x_exact}")

# Plot the error
plot_error(errors, iterations)
