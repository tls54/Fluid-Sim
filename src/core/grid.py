import numpy as np
from .grid_viz import print_velocity_grid, print_velocity_arrows, print_density_heatmap


class FluidGrid:
    """
    Holds the state of a 2D fluid simulation on a uniform grid.
    
    Attributes:
        width (int): Number of grid points in x-direction
        height (int): Number of grid points in y-direction
        h (float): Grid spacing (distance between points)
        velocity (ndarray): Velocity field [H, W, 2]
        density (ndarray): Density/temperature field [H, W]
    """
    def __init__(self, height:int, width:int, h):
        self.height = height
        self.width = width
        self.h = h

        self.velocity = np.zeros((self.height, self.width, 2))  # [y, x, (u,v)]
        self.density = np.zeros((height, width))       # [y, x]

    
    def visualize(self):
        """Print grid state in human-readable format."""
        print_velocity_grid(self.velocity, 'Current Velocity Field')
        print_velocity_arrows(self.velocity, "Current Velocity Field")
        print_density_heatmap(self.density, "Current Density Field")

    @property
    def u(self):
        """X-component of velocity"""
        return self.velocity[:, :, 0]

    @property
    def v(self):
        """Y-component of velocity"""
        return self.velocity[:, :, 1]

    @u.setter
    def u(self, value):
        """Set X-component of velocity"""
        self.velocity[:, :, 0] = value

    @v.setter
    def v(self, value):
        """Set Y-component of velocity"""
        self.velocity[:, :, 1] = value

    def reset(self):
        """Reset all fields to zero"""
        self.velocity = np.zeros((self.height, self.width, 2))
        self.density = np.zeros((self.height, self.width))

    def __repr__(self):
        """String representation of grid"""
        return f"FluidGrid({self.width}x{self.height}, h={self.h})"


if __name__ == '__main__':
    # Create a small test grid
    grid = FluidGrid(height=5, width=5, h=1.0)
    print(grid)  # Test __repr__
    
    # Test 1: Set some velocity values
    print("\n=== Test 1: Rightward flow in middle row ===")
    grid.u[2, :] = 1.0  # All positions at y=2 flow right
    grid.visualize()
    
    # Test 2: Reset and try upward flow
    print("\n=== Test 2: Upward flow in middle column ===")
    grid.reset()
    grid.v[:, 2] = 0.8  # All positions at x=2 flow up
    grid.visualize()
    
    # Test 3: Add density blob
    print("\n=== Test 3: Density blob with diagonal flow ===")
    grid.reset()
    grid.density[2, 2] = 1.0
    grid.density[2, 1] = 0.7
    grid.density[2, 3] = 0.7
    grid.density[1, 2] = 0.5
    grid.density[3, 2] = 0.5
    
    # Diagonal flow
    grid.u[:, :] = 0.5
    grid.v[:, :] = 0.5
    grid.visualize()
    
    print("\n✓ All grid tests passed!")