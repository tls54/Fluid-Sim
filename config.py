from dataclasses import dataclass

'''
SCALING GUIDE:
When you scale grid by factor s while keeping the same physical domain size.

--- Grid Params ---
width_new = width_old * s
height_new = height_old * s
h_new = h_old / s          # CRITICAL: halve spacing to maintain domain size

dt_new = dt_old / s²       # Quarter for s=2 (diffusion stability)


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



slow_detailed = SimParams(width=256, height=256, dt=0.05)
fast_preview = SimParams(width=64, height=64, dt=0.2)
default = SimParams()  # Use defaults
