import numpy as np
import matplotlib.pyplot as plt
from src.core.grid import FluidGrid
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

def main():
    # Load parameters
    params = SimParams()
    
    # Create grid
    grid = FluidGrid(
        height=params.height,
        width=params.width,
        h=params.h
    )
    
    print(f"Created {grid}")
    
    # Add initial density blob
    add_initial_density_blob(
        grid,
        center_x=params.source_x,
        center_y=params.source_y,
        radius=params.source_radius,
        strength=params.source_strength
    )
    
    # Visualize with matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(
        grid.density,
        origin='lower',  # y=0 at bottom
        cmap=params.colormap,
        vmin=params.vmin,
        vmax=params.vmax
    )
    plt.colorbar(label='Density')
    plt.title('Initial Density Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()