from dataclasses import dataclass


@dataclass
class SimParams:
    # Grid - keep small for speed
    width: int = 142
    height: int = 256
    h: float = 1.0
    
    # Time stepping - INCREASE for faster action
    dt: float = 0.3  # Larger timestep = faster motion
    
    # Physics - STRONGER buoyancy
    rho: float = 1.0
    alpha: float = 1.0  
    epsilon: float = 0.05
    
    # Source - WEAKER but continuous
    source_x: int = 64
    source_y: int = 8  
    source_radius: float = 4.0  
    source_strength: float = 2.0  
    
    # Solver
    pressure_iterations: int = 50  # Back to reasonable
    pressure_tolerance: float = 1e-3
    
    # Visualization
    colormap: str = 'hot'
    vmin: float = 0.0
    vmax: float = 3.0



slow_detailed = SimParams(width=256, height=256, dt=0.05)
fast_preview = SimParams(width=64, height=64, dt=0.2)
default = SimParams()  # Use defaults