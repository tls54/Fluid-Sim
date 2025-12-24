# main.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.core.grid import FluidGrid
from src.core.advection import advect_maccormack
from config import SimParams


def add_initial_density_blob(grid, center_x, center_y, radius, strength):
    """
    Add a Gaussian blob of density to the grid.
    
    Args:
        grid: FluidGrid instance
        center_x, center_y: Center position
        radius: Standard deviation of Gaussian
        strength: Peak density value
    """
    for j in range(grid.height):
        for i in range(grid.width):
            # Distance from center
            dx = i - center_x
            dy = j - center_y
            r_squared = dx**2 + dy**2
            
            # Gaussian falloff
            density_value = strength * np.exp(-r_squared / (2 * radius**2))
            grid.density[j, i] += density_value


class SimpleAdvectionSim:
    """Simple simulation that only does advection (no forces yet)."""
    
    def __init__(self, params):
        self.params = params
        self.grid = FluidGrid(params.height, params.width, params.h)
        self.frame_count = 0
        self.setup_initial_conditions()
        
    def setup_initial_conditions(self):
        """Set up initial density blob and constant velocity."""
        # Add density blob
        add_initial_density_blob(
            self.grid,
            center_x=self.params.source_x,
            center_y=self.params.source_y,
            radius=self.params.source_radius,
            strength=self.params.source_strength
        )
        
        # Set constant velocity (rightward and upward drift)
        self.grid.u[:, :] = 0.5  # Move right
        self.grid.v[:, :] = 0.3  # Move up
        
    def step(self):
        """Advance simulation one timestep."""
        # Advect density by velocity field using MacCormack
        self.grid.density = advect_maccormack(
            self.grid.density,
            self.grid.u,
            self.grid.v,
            self.params.dt,
            self.params.h
        )
        
        self.frame_count += 1


def main():
    """Run the advection-only simulation."""
    params = SimParams()
    sim = SimpleAdvectionSim(params)
    
    # Set up visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        sim.grid.density,
        origin='lower',
        cmap=params.colormap,
        vmin=params.vmin,
        vmax=params.vmax,
        animated=True
    )
    plt.colorbar(im, ax=ax, label='Density')
    ax.set_title('Advection Test (Frame 0)')
    ax.set_xlabel('x (grid units)')
    ax.set_ylabel('y (grid units)')
    
    def update(frame):
        """Update function for animation."""
        sim.step()
        im.set_array(sim.grid.density)
        ax.set_title(f'Advection Test (Frame {sim.frame_count})')
        return [im]
    
    # Create animation
    # interval=50 means 50ms between frames (20 FPS)
    anim = FuncAnimation(
        fig, 
        update, 
        frames=200, 
        interval=50, 
        blit=True
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()