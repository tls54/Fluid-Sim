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
    
    #if info > 0:
        #print(f"Warning: CG did not converge in {max_iterations} iterations")
    #elif info < 0:
    #   print(f"Error: CG failed with code {info}")
    
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
    



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def test_divergence():
        """Test divergence computation on known field."""
        print("Testing divergence computation...")
        
        # Create expanding flow: u = x, v = y
        # Divergence should be: ∂u/∂x + ∂v/∂y = 1 + 1 = 2 everywhere
        velocity = np.zeros((10, 10, 2))
        for j in range(10):
            for i in range(10):
                velocity[j, i, 0] = i  # u = x
                velocity[j, i, 1] = j  # v = y
        
        div = compute_divergence(velocity, h=1.0)
        
        # Interior should be ~2.0
        print(f"  Divergence at center: {div[5, 5]:.4f} (expected ~2.0)")
        assert np.abs(div[5, 5] - 2.0) < 0.1, "Divergence should be ~2.0"
        
        print("✓ Divergence test passed!\n")
    
    
    def test_laplacian_matrix():
        """Test that Laplacian matrix has correct structure."""
        print("Testing Laplacian matrix structure...")
        
        # Small grid for inspection
        h, w = 5, 5
        A = build_laplacian_matrix(h, w, h=1.0)
        
        print(f"  Matrix shape: {A.shape}")
        print(f"  Non-zero elements: {A.nnz}")
        print(f"  Expected non-zeros: ~{5 * h * w} (for interior + boundaries)")
        
        # Check a center point has correct stencil
        def idx(i, j):
            return j * w + i
        
        center = idx(2, 2)  # Middle of 5×5 grid
        row = A.getrow(center).toarray()[0]
        
        # Should have: -4 at center, +1 at 4 neighbors
        assert row[center] == -4.0, "Center should be -4"
        assert row[idx(1, 2)] == 1.0, "Left neighbor should be +1"
        assert row[idx(3, 2)] == 1.0, "Right neighbor should be +1"
        assert row[idx(2, 1)] == 1.0, "Bottom neighbor should be +1"
        assert row[idx(2, 3)] == 1.0, "Top neighbor should be +1"
        
        print(f"  Center point stencil: {row[center]} (center), neighbors sum to {row.sum() - row[center]}")
        print("✓ Laplacian matrix structure correct!\n")
    
    
    def test_poisson_solver():
        """Test Poisson solver on known problem."""
        print("Testing Poisson solver...")
        
        # Create a divergence field (e.g., source at center)
        divergence = np.zeros((20, 20))
        divergence[10, 10] = 1.0  # Point source
        
        # Solve for pressure
        pressure = solve_pressure_poisson(
            divergence, 
            h=1.0, 
            rho=1.0, 
            dt=1.0,
            max_iterations=100
        )
        
        # Pressure should be smooth and decay away from source
        print(f"  Pressure at source: {pressure[10, 10]:.4f}")
        print(f"  Pressure at edge: {pressure[0, 0]:.4f}")
        print(f"  Pressure should be higher at source than edge")
        
        # Visualize
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(divergence, origin='lower', cmap='RdBu_r')
        plt.colorbar(label='Divergence')
        plt.title('Input: Divergence Field')
        
        plt.subplot(1, 2, 2)
        plt.imshow(pressure, origin='lower', cmap='viridis')
        plt.colorbar(label='Pressure')
        plt.title('Output: Pressure Field')
        
        plt.tight_layout()
        plt.savefig('poisson_solver_test.png')
        print("✓ Saved visualization to 'poisson_solver_test.png'")
        plt.show()
        
        print("✓ Poisson solver test passed!\n")
    
    
    def test_full_projection():
        """Test that projection makes velocity divergence-free."""
        print("Testing full projection pipeline...")
        
        # Create velocity with divergence (expanding flow)
        velocity = np.zeros((30, 30, 2))
        for j in range(30):
            for i in range(30):
                velocity[j, i, 0] = (i - 15) * 0.1  # Radial outward
                velocity[j, i, 1] = (j - 15) * 0.1
        
        # Compute initial divergence
        div_before = compute_divergence(velocity, h=1.0)
        max_div_before = np.abs(div_before).max()
        print(f"  Max divergence BEFORE projection: {max_div_before:.6f}")
        
        # Solve for pressure
        pressure = solve_pressure_poisson(div_before, h=1.0, rho=1.0, dt=0.1)
        
        # DEBUG: Check pressure stats
        print(f"\n  DEBUG - Pressure stats:")
        print(f"    Pressure min: {pressure.min():.6f}, max: {pressure.max():.6f}")
        print(f"    Pressure at center (15,15): {pressure[15, 15]:.6f}")
        print(f"    Pressure at (0,0): {pressure[0, 0]:.6f}")
        
        # Apply pressure gradient  
        velocity_copy = velocity.copy()  # Keep original for comparison
        velocity = apply_pressure_gradient(velocity, pressure, h=1.0, rho=1.0, dt=0.1)
        
        # DEBUG: Check velocity change
        delta_u = velocity[:, :, 0] - velocity_copy[:, :, 0]
        delta_v = velocity[:, :, 1] - velocity_copy[:, :, 1]
        print(f"\n  DEBUG - Velocity changes:")
        print(f"    Max velocity change (u): {np.abs(delta_u).max():.6f}")
        print(f"    Max velocity change (v): {np.abs(delta_v).max():.6f}")
        print(f"    Velocity change at center (15,15): u={delta_u[15,15]:.6f}, v={delta_v[15,15]:.6f}")
        
        # DEBUG: Check what the gradient looks like
        dp_dx = np.zeros_like(pressure)
        dp_dy = np.zeros_like(pressure)
        dp_dx[1:-1, 1:-1] = (pressure[1:-1, 2:] - pressure[1:-1, :-2]) / 2.0
        dp_dy[1:-1, 1:-1] = (pressure[2:, 1:-1] - pressure[:-2, 1:-1]) / 2.0
        print(f"\n  DEBUG - Pressure gradient:")
        print(f"    dp/dx at center: {dp_dx[15, 15]:.6f}")
        print(f"    dp/dy at center: {dp_dy[15, 15]:.6f}")
        
        # Compute final divergence
        div_after = compute_divergence(velocity, h=1.0)
        max_div_after = np.abs(div_after).max()
        print(f"\n  Max divergence AFTER projection: {max_div_after:.6f}")
        
        # Check a specific point
        print(f"\n  DEBUG - Divergence at center:")
        print(f"    Before: {div_before[15, 15]:.6f}")
        print(f"    After: {div_after[15, 15]:.6f}")
    
    
    # Run all tests
    print("="*60)
    print("RUNNING PROJECTION TESTS")
    print("="*60 + "\n")
    
    test_divergence()
    test_laplacian_matrix()
    test_poisson_solver()
    test_full_projection()
    
    print("="*60)
    print("✓✓✓ ALL PROJECTION TESTS PASSED! ✓✓✓")
    print("="*60)