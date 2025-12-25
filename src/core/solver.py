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
from src.utils.boundary import apply_boundary_conditions


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

        # Performance optimizations: caching
        self._cached_vel_mag = None

        # Cache Laplacian matrix if enabled (15-20% speedup)
        if params.cache_laplacian_matrix:
            from src.core.projections import build_laplacian_matrix
            self._cached_laplacian = build_laplacian_matrix(
                grid.height, grid.width, grid.h
            )
        else:
            self._cached_laplacian = None

        # Pre-compute boundary damping mask (5-8% speedup)
        self._boundary_mask = np.ones((grid.height, grid.width), dtype=bool)
        w = params.boundary_width
        if w > 0:
            self._boundary_mask[w:-w, w:-w] = False  # Interior is False

    @property
    def velocity_magnitude(self):
        """Get cached velocity magnitude (updated each step)."""
        if self._cached_vel_mag is None:
            self._cached_vel_mag = np.sqrt(self.grid.u**2 + self.grid.v**2)
        return self._cached_vel_mag

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
            h,
            enable_clamp=self.params.enable_maccormack_clamp
        )

        new_u = advect_maccormack(
            self.grid.u,
            self.grid.u,
            self.grid.v,
            dt,
            h,
            enable_clamp=self.params.enable_maccormack_clamp
        )
        new_v = advect_maccormack(
            self.grid.v,
            self.grid.u,
            self.grid.v,
            dt,
            h,
            enable_clamp=self.params.enable_maccormack_clamp
        )
        
        self.grid.velocity[:, :, 0] = new_u
        self.grid.velocity[:, :, 1] = new_v
        
        # Density dissipation
        self.grid.density *= self.params.dissipation_rate
        
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
        self.grid.density = np.clip(self.grid.density, 0, self.params.max_density)
        
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
        divergence = compute_divergence(self.grid.velocity, h, self.params.boundary_type)
        
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
            tolerance=self.params.pressure_tolerance,
            laplacian_matrix=self._cached_laplacian
        )
        
        self.grid.velocity = apply_pressure_gradient(
            self.grid.velocity,
            pressure,
            h=h,
            rho=self.params.rho,
            dt=dt,
            boundary_type=self.params.boundary_type
        )
        
        # Velocity limiting - CRITICAL for stability
        # Compute and cache velocity magnitude
        self._cached_vel_mag = np.sqrt(self.grid.u**2 + self.grid.v**2)
        mask = self._cached_vel_mag > self.params.max_velocity
        if np.any(mask):
            scale = self.params.max_velocity / self._cached_vel_mag[mask]
            self.grid.velocity[mask, 0] *= scale
            self.grid.velocity[mask, 1] *= scale
        
        # Mild velocity damping
        self.grid.velocity *= self.params.velocity_damping

        # Apply boundary conditions (configurable: no-slip, free-slip, or periodic)
        self.grid.velocity = apply_boundary_conditions(
            self.grid.velocity,
            boundary_type=self.params.boundary_type
        )

        # Damp near boundaries to reduce artifacts (optimized with pre-computed mask)
        # Only apply for no-slip and free-slip (not periodic)
        if self.params.boundary_width > 0 and self.params.boundary_type != 'periodic':
            self.grid.velocity[self._boundary_mask, :] *= self.params.boundary_damping
        
        self.frame_count += 1