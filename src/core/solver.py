# src/core/solver.py
import numpy as np
from src.core.grid import FluidGrid
from src.core.advection import advect_maccormack
from src.core.forces import apply_buoyancy, add_density_source, vorticity_confinement
from src.core.projections import (
    compute_divergence,
    solve_pressure_poisson,
    apply_pressure_gradient
)


class FluidSolver:
    """
    Main fluid simulation solver using operator splitting.
    
    Each timestep:
    1. Advection - move quantities by velocity
    2. Forces - apply buoyancy and sources
    3. Projection - enforce incompressibility
    """
    
    def __init__(self, grid, params):
        """
        Initialize fluid solver.
        
        Args:
            grid: FluidGrid instance
            params: SimParams configuration
        """
        self.grid = grid
        self.params = params
        self.frame_count = 0
        
    def step(self):
        """Advance simulation by one timestep using operator splitting."""
        dt = self.params.dt
        h = self.params.h
        
        # ===== STEP 1: ADVECTION =====
        self.grid.density = advect_maccormack(
            self.grid.density,
            self.grid.u,
            self.grid.v,
            dt,
            h
        )
        
        new_u = advect_maccormack(
            self.grid.u,
            self.grid.u,
            self.grid.v,
            dt,
            h
        )
        new_v = advect_maccormack(
            self.grid.v,
            self.grid.u,
            self.grid.v,
            dt,
            h
        )
        
        self.grid.velocity[:, :, 0] = new_u
        self.grid.velocity[:, :, 1] = new_v
        
        # Density dissipation - STRONGER
        dissipation_rate = 0.995  # Lose 0.5% per frame
        self.grid.density *= dissipation_rate
        
        # ===== STEP 2: FORCES =====
        self.grid.density = add_density_source(
            self.grid.density,
            source_x=self.params.source_x,
            source_y=self.params.source_y,
            source_radius=self.params.source_radius,
            source_strength=self.params.source_strength,
            dt=dt
        )
        
        # Cap density to prevent accumulation
        self.grid.density = np.clip(self.grid.density, 0, 10.0)
        
        self.grid.velocity = apply_buoyancy(
            self.grid.velocity,
            self.grid.density,
            alpha=self.params.alpha,
            dt=dt
        )

        # NEW: Add vorticity confinement
        self.grid.velocity = vorticity_confinement(
            self.grid.velocity,
            epsilon=self.params.epsilon,
            h=h,
            dt=dt
        )
        
        # ===== STEP 3: PROJECTION =====
        divergence = compute_divergence(self.grid.velocity, h)
        
        # Debug output every 20 frames
        if self.frame_count % 20 == 0:
            print(f"\nFrame {self.frame_count}:")
            print(f"  Density range: [{self.grid.density.min():.4f}, {self.grid.density.max():.4f}]")
            print(f"  Velocity range: u=[{self.grid.u.min():.4f}, {self.grid.u.max():.4f}], v=[{self.grid.v.min():.4f}, {self.grid.v.max():.4f}]")
            print(f"  Max divergence: {np.abs(divergence).max():.6f}")
        
        pressure = solve_pressure_poisson(
            divergence,
            h=h,
            rho=self.params.rho,
            dt=dt,
            max_iterations=self.params.pressure_iterations,
            tolerance=self.params.pressure_tolerance
        )
        
        self.grid.velocity = apply_pressure_gradient(
            self.grid.velocity,
            pressure,
            h=h,
            rho=self.params.rho,
            dt=dt
        )
        
        # Velocity limiting - CRITICAL for stability
        max_velocity = 5.0  # Hard cap on velocity magnitude
        vel_mag = np.sqrt(self.grid.u**2 + self.grid.v**2)
        mask = vel_mag > max_velocity
        if np.any(mask):
            scale = max_velocity / vel_mag[mask]
            self.grid.velocity[mask, 0] *= scale
            self.grid.velocity[mask, 1] *= scale
        
        # Mild velocity damping
        self.grid.velocity *= 0.99
        # Hard boundary conditions
        self.grid.velocity[0, :, 1] = 0   # Bottom wall
        self.grid.velocity[-1, :, 1] = 0  # Top wall
        self.grid.velocity[:, 0, 0] = 0   # Left wall
        self.grid.velocity[:, -1, 0] = 0  # Right wall

        # Damp near boundaries to reduce artifacts
        boundary_width = 2
        damping = 0.7
        self.grid.velocity[:, :boundary_width, :] *= damping
        self.grid.velocity[:, -boundary_width:, :] *= damping
        self.grid.velocity[:boundary_width, :, :] *= damping
        self.grid.velocity[-boundary_width:, :, :] *= damping
        
        self.frame_count += 1