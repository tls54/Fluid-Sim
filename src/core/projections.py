"""
Pressure projection to enforce incompressibility.

Boundary Conditions:
- Pressure: Zero Dirichlet (p = 0 at domain edges)
- Velocity: Normal component enforced by solver (see solver.py)

The pressure Poisson equation ∇²p = (ρ/dt)·∇·u is solved with:
- Zero pressure at boundaries (implicit in Laplacian stencil)
- Interior points use 5-point stencil
- Origin pinned to p[0,0] = 0 to break degeneracy
"""
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

from ..utils.boundary import compute_gradient_with_boundaries, compute_divergence_with_boundaries


def compute_divergence(velocity, h, boundary_type='no-slip'):
    """
    Compute divergence of velocity field: ∇·u = ∂u/∂x + ∂v/∂y

    Uses unified boundary handling from boundary utilities module.

    Args:
        velocity: Velocity field [H, W, 2]
        h: Grid spacing
        boundary_type: Boundary condition type ('no-slip', 'free-slip', 'periodic')

    Returns:
        divergence: Divergence field [H, W]
    """
    # Use shared boundary utilities for consistency
    return compute_divergence_with_boundaries(velocity, h, boundary_type)


def build_laplacian_matrix(height, width, h):
    """
    Build sparse Laplacian matrix for pressure Poisson equation.
    
    Matrix A such that A·p = ∇²p (discretized).

    ∇²p ≈ (p[i+1,j] + p[i-1,j] + p[i,j+1] + p[i,j-1] - 4·p[i,j]) / h²

    Args:
        height: Grid height
        width: Grid width
        h: Grid spacing
    
    Returns:
        A: Sparse matrix [N, N] where N = height * width
    """
    N = height * width
    A = lil_matrix((N, N))
    
    def idx(i, j):
        """Convert 2D indices to 1D flattened index."""
        return j * width + i
    
    # Build matrix row by row
    for j in range(height):
        for i in range(width):
            row = idx(i, j)
            
            # Special case: fix pressure at origin to break degeneracy
            if i == 0 and j == 0:
                A[row, row] = 1.0
                continue
            
            # Center coefficient
            A[row, row] = -4.0 / (h * h)
            
            # Add neighbors (with boundary checks)
            if i > 0:
                A[row, idx(i-1, j)] = 1.0 / (h * h)
            if i < width - 1:
                A[row, idx(i+1, j)] = 1.0 / (h * h)
            if j > 0:
                A[row, idx(i, j-1)] = 1.0 / (h * h)
            if j < height - 1:
                A[row, idx(i, j+1)] = 1.0 / (h * h)
            
    return A.tocsr()



def solve_pressure_poisson(divergence, h, rho, dt, max_iterations=50, tolerance=1e-4, laplacian_matrix=None):
    """
    Solve Poisson equation for pressure: ∇²p = (ρ/dt)·∇·u

    Uses Conjugate Gradient iterative solver.

    Args:
        divergence: Divergence field [H, W]
        h: Grid spacing
        rho: Fluid density
        dt: Timestep
        max_iterations: Max CG iterations
        tolerance: Convergence tolerance
        laplacian_matrix: Pre-built Laplacian matrix (optional, for performance)

    Returns:
        pressure: Pressure field [H, W]
    """
    height, width = divergence.shape
    N = height * width

    # Build or use cached Laplacian matrix
    if laplacian_matrix is not None:
        A = laplacian_matrix
    else:
        A = build_laplacian_matrix(height, width, h)
    
    # Build RHS: b = (rho/dt) * divergence
    b = (rho / dt) * divergence.flatten()
    
    # Fix pressure at origin (p[0,0] = 0) to break degeneracy
    b[0] = 0.0
    
    # Solve A·p = b using Conjugate Gradient
    p_flat, info = cg(A, b, maxiter=max_iterations, rtol=tolerance)
    
    if info > 0:
        print(f"Warning: CG did not converge in {max_iterations} iterations")
    elif info < 0:
        print(f"Error: CG failed with code {info}")
    
    # Reshape back to 2D
    pressure = p_flat.reshape((height, width))
    
    return pressure


def apply_pressure_gradient(velocity, pressure, h, rho, dt, boundary_type='no-slip'):
    """
    Subtract pressure gradient from velocity to make it divergence-free.

    u_new = u - (dt/ρ)·∇p

    Uses unified boundary handling from boundary utilities module.

    Args:
        velocity: Velocity field before projection [H, W, 2]
        pressure: Pressure field [H, W]
        h: Grid spacing
        rho: Fluid density
        dt: Timestep
        boundary_type: Boundary condition type ('no-slip', 'free-slip', 'periodic')

    Returns:
        velocity: Corrected velocity field [H, W, 2]
    """
    # Use shared boundary utilities for consistent gradient computation
    dp_dx, dp_dy = compute_gradient_with_boundaries(pressure, h, boundary_type)

    # Apply correction: u = u - (dt/rho) * ∇p
    velocity[:, :, 0] -= (dt / rho) * dp_dx
    velocity[:, :, 1] -= (dt / rho) * dp_dy

    return velocity