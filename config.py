from dataclasses import dataclass


@dataclass
class SimParams:
    # Grid
    width: int = 128
    height: int = 128
    h: float = 1.0          # Grid spacing
    
    # Time
    dt: float = 0.1
    
    # Physics
    rho: float = 1.0        # Density
    alpha: float = 0.1      # Buoyancy strength
    epsilon: float = 0.0    # Vorticity confinement (start with 0)
    
    # Source
    source_x: int = 64
    source_y: int = 10
    source_radius: float = 5.0
    source_strength: float = 1.0

    # Solver parameters
    pressure_iterations: int = 50  # Max iterations for conjugate gradient
    pressure_tolerance: float = 1e-4
    
    # Visualization
    colormap: str = 'viridis'   # Matplotlib colormap
    vmin: float = 0.0       # Min density for colormap
    vmax: float = 1.0       # Max density for colormap


slow_detailed = SimParams(width=256, height=256, dt=0.05)
fast_preview = SimParams(width=64, height=64, dt=0.2)
default = SimParams()  # Use defaults