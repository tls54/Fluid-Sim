from dataclasses import dataclass

'''
SCALING GUIDE:
When you scale grid by factor s while keeping the same physical domain size.

--- Grid Params ---
width_new = width_old * s
height_new = height_old * s
h_new = h_old / s          # CRITICAL: halve spacing to maintain domain size

dt_new = dt_old / s        # Halve for s=2 (maintains CFL number for advection)


--- Source Params ---
source_x_new = source_x_old * s
source_y_new = source_y_old * s

# Radius scales to maintain same physical size
source_radius_new = source_radius_old * s

# Strength stays constant (it's per unit time, not per unit area)
source_strength_new = source_strength_old


--- Stability Params ---
# Velocity limits scale inversely with grid (same physical velocities)
max_velocity_new = max_velocity_old * s  # Can handle higher grid velocities

# Dissipation rates stay the same (they're per-frame)
dissipation_rate_new = dissipation_rate_old
velocity_damping_new = velocity_damping_old
boundary_damping_new = boundary_damping_old

# Boundary width scales with resolution
boundary_width_new = boundary_width_old * s

--- Solver Params (Optional) ---

# More grid points = harder to converge
pressure_iterations_new = pressure_iterations_old * sqrt(s)  # Heuristic
pressure_tolerance_new = pressure_tolerance_old / s          # Tighter tolerance

'''

@dataclass
class SimParams:
    # Grid 
    width: int = 72
    height: int = 128
    h: float = 1.0
    
    # Time stepping - INCREASE for faster action
    dt: float = 0.1  # Larger timestep = faster motion
    
    # Physics - STRONGER buoyancy
    rho: float = 1.0  # Fluid density [kg/m³]
    alpha: float = 1.0  # Buoyancy strength [m/s² per density unit]
    epsilon: float = 0.05  # Vorticity confinement strength [dimensionless]
    
    # Source - WEAKER but continuous
    source_x: int = 32
    source_y: int = 8  
    source_radius: float = 4.0  
    source_strength: float = 2.0  
    radius_noise = 0.15
    source_jitter_x = 2
    source_jitter_y = 1
    
    # Solver
    pressure_iterations: int = 100  # Back to reasonable
    pressure_tolerance: float = 1e-3

    # Stability parameters
    dissipation_rate: float = 0.995  # Density dissipation per frame (0.995 = lose 0.5% per frame)
    max_density: float = 10.0  # Maximum density cap to prevent accumulation
    max_velocity: float = 7.0  # Hard cap on velocity magnitude for stability
    velocity_damping: float = 0.99  # Mild velocity damping factor
    boundary_damping: float = 0.7  # Damping factor near boundaries to reduce artifacts
    boundary_width: int = 2  # Width of boundary damping region (in grid cells)

    # Boundary conditions
    boundary_type: str = 'no-slip'  # Options: 'no-slip', 'free-slip', 'periodic'

    # Performance optimization flags
    enable_maccormack_clamp: bool = True  # Set False for 30-50% speedup (less stable)
    cache_laplacian_matrix: bool = True  # Cache pressure solver matrix (15-20% speedup)

    # Visualization
    colormap: str = 'hot'
    vmin: float = 0.0
    vmax: float = 3.0



def scale_default(s: float) -> SimParams:
    """
    Scale the default parameters by factor s following the scaling guide.

    Args:
        s: Scaling factor (e.g., 2.0 for doubling resolution)

    Returns:
        SimParams object with scaled parameters
    """
    base = SimParams()  # Get default params

    return SimParams(
        # Grid - scale resolution, inverse scale spacing
        width=int(base.width * s),
        height=int(base.height * s),
        h=base.h / s,

        # Time - linear scaling to maintain CFL number (advection stability)
        dt=base.dt / s,

        # Physics - unchanged
        rho=base.rho,
        alpha=base.alpha,
        epsilon=base.epsilon,

        # Source - scale position and radius
        source_x=int(base.source_x * s),
        source_y=int(base.source_y * s),
        source_radius=base.source_radius * s,
        source_strength=base.source_strength,  # Stays constant

        # Solver - heuristic scaling
        pressure_iterations=int(base.pressure_iterations * (s ** 0.5)),  # sqrt(s)
        pressure_tolerance=base.pressure_tolerance / s,

        # Stability - rates stay same, limits and widths scale
        dissipation_rate=base.dissipation_rate,
        max_density=base.max_density,
        max_velocity=base.max_velocity * s,
        velocity_damping=base.velocity_damping,
        boundary_damping=base.boundary_damping,
        boundary_width=int(base.boundary_width * s),

        # Boundary conditions
        boundary_type=base.boundary_type,

        # Performance flags
        enable_maccormack_clamp=base.enable_maccormack_clamp,
        cache_laplacian_matrix=base.cache_laplacian_matrix,

        # Visualization
        colormap=base.colormap,
        vmin=base.vmin,
        vmax=base.vmax,
    )


def create_with_physical_size(width_meters: float, height_meters: float,
                              cells_per_meter: float = 1.0) -> SimParams:
    """
    Create SimParams by specifying physical domain size and resolution.

    This is an alternative to scale_default() that lets you directly specify
    the physical dimensions you want.

    Args:
        width_meters: Physical width of domain in meters
        height_meters: Physical height of domain in meters
        cells_per_meter: Grid resolution (cells per meter). Higher = more detail.
                        Examples:
                        - 1.0 = 1 cell per meter (default)
                        - 2.0 = 2 cells per meter (finer detail)
                        - 0.5 = 1 cell per 2 meters (coarser)

    Returns:
        SimParams configured for the specified physical domain

    Examples:
        # Small, detailed simulation: 10m × 20m with 2 cells/meter
        params = create_with_physical_size(10, 20, cells_per_meter=2.0)
        # Result: 20×40 grid, h=0.5

        # Large, coarse simulation: 200m × 400m with 0.5 cells/meter
        params = create_with_physical_size(200, 400, cells_per_meter=0.5)
        # Result: 100×200 grid, h=2.0
    """
    base = SimParams()

    # Calculate grid parameters
    width_cells = int(width_meters * cells_per_meter)
    height_cells = int(height_meters * cells_per_meter)
    h = 1.0 / cells_per_meter

    # Calculate scaling factor relative to default
    # Default domain is 72m × 128m (with h=1.0, cells_per_meter=1.0)
    s = cells_per_meter  # Scaling factor for resolution-dependent params

    return SimParams(
        # Grid
        width=width_cells,
        height=height_cells,
        h=h,

        # Time - scale with grid resolution
        dt=base.dt / s,

        # Physics - unchanged
        rho=base.rho,
        alpha=base.alpha,
        epsilon=base.epsilon,

        # Source - position in middle, scale radius with resolution
        source_x=width_cells // 2,
        source_y=height_cells // 16,  # Near bottom
        source_radius=base.source_radius * s,
        source_strength=base.source_strength,

        # Solver
        pressure_iterations=int(base.pressure_iterations * (s ** 0.5)),
        pressure_tolerance=base.pressure_tolerance / s,

        # Stability
        dissipation_rate=base.dissipation_rate,
        max_density=base.max_density,
        max_velocity=base.max_velocity * s,
        velocity_damping=base.velocity_damping,
        boundary_damping=base.boundary_damping,
        boundary_width=int(base.boundary_width * s),

        # Boundary conditions
        boundary_type=base.boundary_type,

        # Performance flags
        enable_maccormack_clamp=base.enable_maccormack_clamp,
        cache_laplacian_matrix=base.cache_laplacian_matrix,

        # Visualization
        colormap=base.colormap,
        vmin=base.vmin,
        vmax=base.vmax,
    )


slow_detailed = SimParams(width=256, height=256, dt=0.05)
fast_preview = SimParams(width=64, height=64, dt=0.2)
default = SimParams()  # Use defaults
