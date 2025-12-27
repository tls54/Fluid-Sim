# main.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.core.grid import FluidGrid
from src.core.solver import FluidSolver
from config import scale_default


def main(params):
    """Run the full fluid simulation."""
    # Create grid and solver
    grid = FluidGrid(params.height, params.width, params.h)
    solver = FluidSolver(grid, params)
    
    # Set up visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Density plot
    im1 = ax1.imshow(
        grid.density,
        origin='lower',
        cmap=params.colormap,
        vmin=params.vmin,
        vmax=params.vmax,
        animated=True
    )
    ax1.set_title('Density Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='Density')
    
    # Velocity magnitude plot (use initial zero velocity)
    im2 = ax2.imshow(
        solver.velocity_magnitude,
        origin='lower',
        cmap='viridis',
        animated=True
    )
    ax2.set_title('Velocity Magnitude')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, label='|v|')
    
    def update(frame):
        """Update function for animation."""
        solver.step()
        
        # Update density
        im1.set_array(grid.density)

        # Use cached velocity magnitude (performance optimization)
        vel_magnitude = solver.velocity_magnitude
        im2.set_array(vel_magnitude)
        im2.set_clim(vmin=0, vmax=vel_magnitude.max())
        
        # Update titles with frame count
        ax1.set_title(f'Density Field (frame {solver.frame_count})')
        ax2.set_title(f'Velocity Magnitude (frame {solver.frame_count})')
        
        return [im1, im2]
    
    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=500,
        interval=50,  # 50ms = 20 FPS
        blit=False
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    params = scale_default(1)
    main(params)