import numpy as np
import matplotlib.pyplot as plt

def generate_linear_system(n):
    A = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, i] = 5 * (i + 1)  # Set diagonal elements
            else:
                A[i, j] = -1  # Set off-diagonal elements
    
    # Generate random vector b
    b = np.random.rand(n)

    return A, b


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
n = 100
A, b = generate_linear_system(n)  # Generate a linear system
x0 = np.zeros(np.size(b))



# Solve using Jacobi method
x_jacobi, iterations, errors = jacobi_method(A, b, x0)

# Calculate exact solution
x_exact = np.linalg.solve(A, b)

# Print results
print(f"Iterations: {iterations}")
print(f"Solution Jacobi: {x_jacobi}")
print(f"Exact solution: {x_exact}")
print(A)

# Plot the error
plot_error(errors, iterations)