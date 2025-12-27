"""
Quick test to demonstrate improved convergence with initial guess.
"""
import numpy as np
from src.core.projections import solve_pressure_poisson, build_laplacian_matrix

# Create a test divergence field
height, width = 64, 64
h = 1.0
rho = 1.0
dt = 0.01

# Create a smooth divergence pattern (simulates real flow)
x = np.linspace(0, 2*np.pi, width)
y = np.linspace(0, 2*np.pi, height)
X, Y = np.meshgrid(x, y)
divergence = np.sin(X) * np.cos(Y)

# Pre-build Laplacian matrix
A = build_laplacian_matrix(height, width, h)

print("Testing CG convergence with and without initial guess...")
print("=" * 60)

# Test 1: Cold start (no initial guess)
print("\nTest 1: First solve (no initial guess)")
pressure1 = solve_pressure_poisson(
    divergence, h, rho, dt,
    max_iterations=100,
    tolerance=1e-6,
    laplacian_matrix=A,
    initial_guess=None
)

# Test 2: Slightly perturbed divergence (simulates next timestep)
divergence2 = divergence + 0.01 * np.sin(2*X) * np.cos(2*Y)

print("\nTest 2: Second solve WITHOUT initial guess")
pressure2_cold = solve_pressure_poisson(
    divergence2, h, rho, dt,
    max_iterations=100,
    tolerance=1e-6,
    laplacian_matrix=A,
    initial_guess=None
)

print("\nTest 3: Second solve WITH initial guess from previous solution")
pressure2_warm = solve_pressure_poisson(
    divergence2, h, rho, dt,
    max_iterations=100,
    tolerance=1e-6,
    laplacian_matrix=A,
    initial_guess=pressure1  # Use previous solution
)

print("\n" + "=" * 60)
print("Summary:")
print("Using the previous pressure as an initial guess allows CG to")
print("converge faster, especially when pressure changes smoothly")
print("between timesteps (typical in fluid simulations).")
print("=" * 60)
