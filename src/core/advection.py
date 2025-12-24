# src/core/advection.py
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter

from ..utils.interpolation import bilinear_interpolate

def semi_lagrangian_advect(q, u, v, dt, h, boundary='clamp'):
    """
    Basic semi-Lagrangian advection (first-order accurate).
    
    Traces particles backward in time and interpolates.
    
    Args:
        q: Quantity to advect [H, W] - the field being transported
        u: X-velocity component [H, W] - how fast in x-direction
        v: Y-velocity component [H, W] - how fast in y-direction
        dt: Timestep - how far forward in time
        h: Grid spacing - physical distance between grid points
        boundary: Boundary condition ('clamp', 'periodic', 'zero')
    
    Returns:
        Advected quantity [H, W]
    """
    height, width = q.shape
    
    # Create coordinate grids for all grid points
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    # Trace backward along velocity field
    x_prev = x_grid - u * dt / h
    y_prev = y_grid - v * dt / h
    
    # Interpolate q at those previous positions
    q_new = bilinear_interpolate(q, x_prev, y_prev, boundary)
    
    return q_new


def advect_maccormack(q, u, v, dt, h, boundary='clamp'):
    """
    MacCormack advection scheme (second-order accurate).
    
    Uses predictor-corrector with error estimation and clamping.
    
    Args:
        q: Quantity to advect [H, W]
        u: X-velocity component [H, W]
        v: Y-velocity component [H, W]
        dt: Timestep
        h: Grid spacing
        boundary: Boundary condition
    
    Returns:
        Advected quantity with error correction [H, W]
    """
    # Step 1: Forward advection (predictor)
    # Trace backward to predict where things came from
    q_hat = semi_lagrangian_advect(q, u, v, dt, h, boundary)
    
    # Step 2: Backward advection (corrector)
    # Trace forward from the prediction to estimate error
    # Note: We negate velocity to go forward instead of backward
    q_tilde = semi_lagrangian_advect(q_hat, -u, -v, dt, h, boundary)
    
    # Step 3: Error correction
    # The difference (q - q_tilde) estimates the error
    # We add half of it back to improve accuracy
    error = 0.5 * (q - q_tilde)
    q_corrected = q_hat + error
    
    # Step 4: Clamp to prevent overshoots
    # Ensure corrected values stay within original neighborhood bounds
    q_min, q_max = compute_clamp_bounds(q)
    q_new = np.clip(q_corrected, q_min, q_max)
    
    return q_new


def compute_clamp_bounds(q):
    """
    Compute min/max bounds from 3x3 neighborhoods for clamping.
    
    For each cell, finds the min and max values in its neighborhood
    (itself + 8 surrounding cells).
    
    Args:
        q: Field [H, W]
    
    Returns:
        q_min: Array [H, W] of local minimums
        q_max: Array [H, W] of local maximums
    """
    # scipy does exactly what we need!
    # size=3 means 3x3 neighborhood
    # mode='nearest' handles boundaries by repeating edge values
    q_min = minimum_filter(q, size=3, mode='nearest')
    q_max = maximum_filter(q, size=3, mode='nearest')
    
    return q_min, q_max


# ============================================================================
# DEMO / VISUALIZATION CODE
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def demo_advection_steps():
        """Visualize how a blob moves through advection."""
        print("Running advection demo...")
        
        # Create initial blob at center
        grid_size = 21
        q = np.zeros((grid_size, grid_size))
        q[10, 10] = 1.0
        
        # Constant diagonal velocity
        u = np.ones((grid_size, grid_size)) * 1.0  # Right
        v = np.ones((grid_size, grid_size)) * 0.5  # Up
        
        dt = 0.5
        h = 1.0
        
        # Visualize 4 steps
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        
        axes[0].imshow(q, origin='lower', cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('Initial (t=0)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].grid(True, alpha=0.3)
        
        for step in range(4):
            q = semi_lagrangian_advect(q, u, v, dt, h)
            axes[step+1].imshow(q, origin='lower', cmap='hot', vmin=0, vmax=1)
            axes[step+1].set_title(f'Step {step+1} (t={dt*(step+1):.1f})')
            axes[step+1].set_xlabel('x')
            axes[step+1].set_ylabel('y')
            axes[step+1].grid(True, alpha=0.3)
        
        plt.suptitle('Semi-Lagrangian Advection: Blob Moving Diagonally', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('advection_demo.png', dpi=100, bbox_inches='tight')
        print("✓ Saved visualization to 'advection_demo.png'")
        plt.show()
        print("✓ Blob should move diagonally up-right!")
    
    
    def demo_velocity_field():
        """Show how different velocity patterns advect a blob."""
        print("\nRunning velocity field comparison...")
        
        grid_size = 21
        center = grid_size // 2
        
        # Create initial blob
        q_init = np.zeros((grid_size, grid_size))
        for j in range(grid_size):
            for i in range(grid_size):
                r2 = (i - center)**2 + (j - center)**2
                q_init[j, i] = np.exp(-r2 / 8.0)
        
        dt = 0.3
        h = 1.0
        steps = 3
        
        # Different velocity patterns
        velocities = {
            'Rightward': (np.ones((grid_size, grid_size)) * 1.0, 
                         np.zeros((grid_size, grid_size))),
            'Upward': (np.zeros((grid_size, grid_size)), 
                      np.ones((grid_size, grid_size)) * 1.0),
            'Diagonal': (np.ones((grid_size, grid_size)) * 0.7, 
                        np.ones((grid_size, grid_size)) * 0.7),
            'Rotation': (None, None)  # Will compute below
        }
        
        # Rotational flow
        u_rot = np.zeros((grid_size, grid_size))
        v_rot = np.zeros((grid_size, grid_size))
        for j in range(grid_size):
            for i in range(grid_size):
                dx = i - center
                dy = j - center
                u_rot[j, i] = -dy * 0.3
                v_rot[j, i] = dx * 0.3
        velocities['Rotation'] = (u_rot, v_rot)
        
        fig, axes = plt.subplots(4, steps+1, figsize=(12, 12))
        
        for row, (name, (u, v)) in enumerate(velocities.items()):
            q = q_init.copy()
            
            # Initial state
            axes[row, 0].imshow(q, origin='lower', cmap='hot', vmin=0, vmax=1)
            axes[row, 0].set_title(f'{name}\n(t=0)')
            axes[row, 0].set_ylabel(name, fontsize=11, fontweight='bold')
            
            # Advect and show steps
            for step in range(steps):
                q = semi_lagrangian_advect(q, u, v, dt, h)
                axes[row, step+1].imshow(q, origin='lower', cmap='hot', vmin=0, vmax=1)
                axes[row, step+1].set_title(f't={dt*(step+1):.1f}')
        
        plt.suptitle('Different Velocity Patterns', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig('advection_velocity_patterns.png', dpi=100, bbox_inches='tight')
        print("✓ Saved visualization to 'advection_velocity_patterns.png'")
        plt.show()


    def demo_semi_lagrangian_vs_maccormack():
        """Compare semi-Lagrangian vs MacCormack quality."""
        print("\nRunning Semi-Lagrangian vs MacCormack comparison...")
        
        # Create sharp initial blob
        grid_size = 40
        q_init = np.zeros((grid_size, grid_size))
        center = grid_size // 2
        for j in range(grid_size):
            for i in range(grid_size):
                r2 = (i - center)**2 + (j - center)**2
                q_init[j, i] = np.exp(-r2 / 20.0)
        
        # Constant diagonal velocity
        u = np.ones((grid_size, grid_size)) * 1.0
        v = np.ones((grid_size, grid_size)) * 0.5
        
        dt = 0.5
        h = 1.0
        num_steps = 10
        
        # Advect with both methods
        q_sl = q_init.copy()
        q_mc = q_init.copy()
        
        fig, axes = plt.subplots(2, num_steps + 1, figsize=(20, 6))
        
        # Initial state
        axes[0, 0].imshow(q_init, origin='lower', cmap='hot', vmin=0, vmax=1)
        axes[0, 0].set_title('Initial')
        axes[0, 0].set_ylabel('Semi-Lagrangian', fontweight='bold')
        axes[1, 0].imshow(q_init, origin='lower', cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_ylabel('MacCormack', fontweight='bold')
        
        for step in range(num_steps):
            # Semi-Lagrangian
            q_sl = semi_lagrangian_advect(q_sl, u, v, dt, h)
            axes[0, step+1].imshow(q_sl, origin='lower', cmap='hot', vmin=0, vmax=1)
            axes[0, step+1].set_title(f't={dt*(step+1):.1f}')
            
            # MacCormack
            q_mc = advect_maccormack(q_mc, u, v, dt, h)
            axes[1, step+1].imshow(q_mc, origin='lower', cmap='hot', vmin=0, vmax=1)
        
        plt.suptitle('Quality Comparison: Semi-Lagrangian (top) vs MacCormack (bottom)', 
                    fontsize=14)
        plt.tight_layout()
        plt.savefig('advection_comparison.png', dpi=100, bbox_inches='tight')
        print("✓ Saved comparison to 'advection_comparison.png'")
        print("  MacCormack should stay sharper and preserve peak better!")
        plt.show()
    
    
    # Run demos
    demo_advection_steps()
    demo_velocity_field()
    demo_semi_lagrangian_vs_maccormack()
    
    print("\n" + "="*60)
    print("✓✓✓ All advection demos completed successfully! ✓✓✓")
    print("="*60)