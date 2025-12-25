import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg


def compute_divergence(velocity, h):
    """
    Compute divergence of velocity field: ∇·u = ∂u/∂x + ∂v/∂y
    
    Uses central differences for interior points, one-sided for boundaries.
    
    Args:
        velocity: Velocity field [H, W, 2]
        h: Grid spacing
    
    Returns:
        divergence: Divergence field [H, W]
    """
    height, width = velocity.shape[:2]
    divergence = np.zeros((height, width))
    
    # Extract u and v components
    u = velocity[:, :, 0]
    v = velocity[:, :, 1]
    
    # Interior: central differences
    du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
    dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * h)
    divergence[1:-1, 1:-1] = du_dx + dv_dy
    
    # Boundaries: one-sided differences
    
    # Left edge (i=0): forward difference for ∂u/∂x
    divergence[1:-1, 0] = (u[1:-1, 1] - u[1:-1, 0]) / h + \
                          (v[2:, 0] - v[:-2, 0]) / (2 * h)
    
    # Right edge (i=-1): backward difference for ∂u/∂x
    divergence[1:-1, -1] = (u[1:-1, -1] - u[1:-1, -2]) / h + \
                           (v[2:, -1] - v[:-2, -1]) / (2 * h)
    
    # Bottom edge (j=0): forward difference for ∂v/∂y
    divergence[0, 1:-1] = (u[0, 2:] - u[0, :-2]) / (2 * h) + \
                          (v[1, 1:-1] - v[0, 1:-1]) / h
    
    # Top edge (j=-1): backward difference for ∂v/∂y
    divergence[-1, 1:-1] = (u[-1, 2:] - u[-1, :-2]) / (2 * h) + \
                           (v[-1, 1:-1] - v[-2, 1:-1]) / h
    
    # Corners: both one-sided
    divergence[0, 0] = (u[0, 1] - u[0, 0]) / h + (v[1, 0] - v[0, 0]) / h
    divergence[0, -1] = (u[0, -1] - u[0, -2]) / h + (v[1, -1] - v[0, -1]) / h
    divergence[-1, 0] = (u[-1, 1] - u[-1, 0]) / h + (v[-1, 0] - v[-2, 0]) / h
    divergence[-1, -1] = (u[-1, -1] - u[-1, -2]) / h + (v[-1, -1] - v[-2, -1]) / h
    
    return divergence


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



def solve_pressure_poisson(divergence, h, rho, dt, max_iterations=50, tolerance=1e-4):
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
    
    Returns:
        pressure: Pressure field [H, W]
    """
    height, width = divergence.shape
    N = height * width
    
    # Build Laplacian matrix
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


def apply_pressure_gradient(velocity, pressure, h, rho, dt):
    """
    Subtract pressure gradient from velocity to make it divergence-free.
    
    u_new = u - (dt/ρ)·∇p
    
    Args:
        velocity: Velocity field before projection [H, W, 2]
        pressure: Pressure field [H, W]
        h: Grid spacing
        rho: Fluid density
        dt: Timestep
    
    Returns:
        velocity: Corrected velocity field [H, W, 2]
    """
    height, width = pressure.shape
    dp_dx = np.zeros((height, width))
    dp_dy = np.zeros((height, width))
    
    # Interior: central differences
    dp_dx[1:-1, 1:-1] = (pressure[1:-1, 2:] - pressure[1:-1, :-2]) / (2 * h)
    dp_dy[1:-1, 1:-1] = (pressure[2:, 1:-1] - pressure[:-2, 1:-1]) / (2 * h)
    
    # Boundaries: one-sided differences
    # Left edge (i=0): forward difference
    dp_dx[1:-1, 0] = (pressure[1:-1, 1] - pressure[1:-1, 0]) / h
    # Right edge (i=-1): backward difference
    dp_dx[1:-1, -1] = (pressure[1:-1, -1] - pressure[1:-1, -2]) / h
    
    # Bottom edge (j=0): forward difference
    dp_dy[0, 1:-1] = (pressure[1, 1:-1] - pressure[0, 1:-1]) / h
    # Top edge (j=-1): backward difference
    dp_dy[-1, 1:-1] = (pressure[-1, 1:-1] - pressure[-2, 1:-1]) / h
    
    # Corners (just use neighboring values)
    dp_dx[0, 0] = (pressure[0, 1] - pressure[0, 0]) / h
    dp_dx[0, -1] = (pressure[0, -1] - pressure[0, -2]) / h
    dp_dx[-1, 0] = (pressure[-1, 1] - pressure[-1, 0]) / h
    dp_dx[-1, -1] = (pressure[-1, -1] - pressure[-1, -2]) / h
    
    dp_dy[0, 0] = (pressure[1, 0] - pressure[0, 0]) / h
    dp_dy[0, -1] = (pressure[1, -1] - pressure[0, -1]) / h
    dp_dy[-1, 0] = (pressure[-1, 0] - pressure[-2, 0]) / h
    dp_dy[-1, -1] = (pressure[-1, -1] - pressure[-2, -1]) / h
    
    # Apply correction: u = u - (dt/rho) * ∇p
    velocity[:, :, 0] -= (dt / rho) * dp_dx
    velocity[:, :, 1] -= (dt / rho) * dp_dy

    return velocity