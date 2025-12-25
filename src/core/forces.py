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
    
    Adds small-scale rotational detail that numerical diffusion removes.
    
    Args:
        velocity: Velocity field [H, W, 2]
        epsilon: Confinement strength (typically 0.01 - 0.5)
        h: Grid spacing
        dt: Timestep
    
    Returns:
        velocity: Updated velocity field [H, W, 2]
    """
    if epsilon == 0:
        return velocity  # Skip if disabled
    
    height, width = velocity.shape[:2]
    u = velocity[:, :, 0]
    v = velocity[:, :, 1]
    
    # Step 1: Compute vorticity ω = ∂v/∂x - ∂u/∂y
    vorticity = np.zeros((height, width))
    
    # Central differences for interior
    vorticity[1:-1, 1:-1] = (
        (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * h) -  # ∂v/∂x
        (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * h)    # ∂u/∂y
    )
    
    # Step 2: Compute gradient of |ω|
    abs_vorticity = np.abs(vorticity)
    
    # Gradient of |ω|
    dw_dx = np.zeros((height, width))
    dw_dy = np.zeros((height, width))
    
    dw_dx[1:-1, 1:-1] = (abs_vorticity[1:-1, 2:] - abs_vorticity[1:-1, :-2]) / (2 * h)
    dw_dy[1:-1, 1:-1] = (abs_vorticity[2:, 1:-1] - abs_vorticity[:-2, 1:-1]) / (2 * h)
    
    # Step 3: Normalize to get N = ∇|ω| / |∇|ω||
    gradient_mag = np.sqrt(dw_dx**2 + dw_dy**2) + 1e-10  # Add small epsilon to avoid division by zero
    
    N_x = dw_dx / gradient_mag
    N_y = dw_dy / gradient_mag
    
    # Step 4: Compute confinement force
    # In 2D: f = ε·h·(N × ω) where × is perpendicular product
    # (N_x, N_y) × ω gives force perpendicular to gradient
    force_x = epsilon * h * N_y * vorticity  # Perpendicular: swap and negate one
    force_y = epsilon * h * (-N_x) * vorticity
    
    # Apply force
    velocity[:, :, 0] += force_x * dt
    velocity[:, :, 1] += force_y * dt
    
    return velocity