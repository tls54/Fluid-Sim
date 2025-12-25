import numpy as np


def apply_buoyancy(velocity, density, alpha, dt):
    """
    Apply buoyancy force to velocity field.
    
    Buoyancy makes hot/light fluid rise and cold/heavy fluid sink.
    Uses Boussinesq approximation: only vertical velocity is affected.
    
    Args:
        velocity: Velocity field [H, W, 2]
        density: Density/temperature field [H, W]
        alpha: Buoyancy strength coefficient
        dt: Timestep
    
    Returns:
        velocity: Updated velocity field [H, W, 2]
    """
    # TODO: Add -alpha * density * dt to v-component
    velocity[:, :, 1] += alpha * density * dt
    return velocity


def add_density_source(density, source_x, source_y, source_radius, source_strength, dt):
    """
    Add a continuous source of density (smoke/heat injection).
    
    Creates a Gaussian blob of density centered at (source_x, source_y).
    
    Args:
        density: Density field [H, W]
        source_x: X-coordinate of source center
        source_y: Y-coordinate of source center
        source_radius: Standard deviation of Gaussian (spread)
        source_strength: Peak intensity of source
        dt: Timestep
    
    Returns:
        density: Updated density field [H, W]
    """
    height, width = density.shape
    
    # TODO: For each grid point, compute distance from source
    # TODO: Add Gaussian: density += strength * exp(-r²/radius²) * dt
    
    # Create coordinate grids
    y_grid, x_grid = np.indices((height, width))

    # Distance from source
    dx = x_grid - source_x
    dy = y_grid - source_y
    r_squared = dx**2 + dy**2

    # Add Gaussian blob
    source = source_strength * np.exp(-r_squared / (2 * source_radius**2))
    density += source * dt
    return density



def vorticity_confinement(velocity, epsilon, h, dt):
    """
    Add vorticity confinement force to maintain turbulence.

    TODO: Implement later if needed for visual enhancement.

    Args:
        velocity: Velocity field [H, W, 2]
        epsilon: Confinement strength
        h: Grid spacing
        dt: Timestep

    Returns:
        velocity: Updated velocity field [H, W, 2]
    """
    # Not implemented yet
    return velocity