# 2D Incompressible Fluid Simulation

A high-performance 2D fluid simulator implementing the **stable fluids** methodology with operator splitting. This project simulates smoke-like behavior with buoyancy-driven rising motion and turbulent swirling patterns.

![Simulation Features](https://img.shields.io/badge/Features-Incompressible%20Flow%20%7C%20Vorticity%20Confinement%20%7C%20MacCormack%20Advection-blue)

## Table of Contents
- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Implementation Details](#implementation-details)
  - [Operator Splitting](#operator-splitting)
  - [Advection Schemes](#advection-schemes)
  - [Force Application](#force-application)
  - [Pressure Projection](#pressure-projection)
- [Code Architecture](#code-architecture)
- [Numerical Methods](#numerical-methods)
- [Performance Optimizations](#performance-optimizations)
- [Running the Simulation](#running-the-simulation)
- [Configuration](#configuration)
- [References](#references)

## Overview

This fluid simulator solves the **Navier-Stokes equations** for 2D incompressible flow using a grid-based Eulerian approach. The simulation creates visually compelling smoke behavior with:

- **Rising plumes** driven by buoyancy forces
- **Turbulent swirling** enhanced by vorticity confinement
- **Stable evolution** through semi-Lagrangian advection
- **Divergence-free flow** enforced via pressure projection

The implementation prioritizes **educational clarity** and **computational efficiency**, making it suitable for learning CFD principles or as a foundation for extended fluid simulations.

## Mathematical Foundation

### Governing Equations

The simulation solves a simplified form of the incompressible Navier-Stokes equations:

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f    [Momentum equation]
∇·u = 0                                 [Incompressibility constraint]
∂ρ/∂t + u·∇ρ = 0                       [Passive scalar transport]
```

Where:
- **u** = (u, v) is the velocity field
- **p** is the pressure field
- **ρ** is the density (passive scalar, represents smoke concentration)
- **ν** is the kinematic viscosity
- **f** represents external forces (buoyancy, vorticity confinement)

### Key Approximations

1. **Inviscid Flow**: The viscosity term `ν∇²u` is omitted. Numerical diffusion from discretization provides implicit viscosity.

2. **Boussinesq Approximation**: Density variations only affect buoyancy force, not the overall flow dynamics:
   ```
   f_buoyancy = α·ρ·ĵ
   ```
   where α is the buoyancy coefficient and ĵ is the upward unit vector.

3. **Zero External Pressure**: Gauge pressure is measured relative to ambient conditions.

4. **Passive Scalar**: Density is advected by the flow but doesn't affect it (except through buoyancy).

## Implementation Details

### Operator Splitting

The simulation uses **operator splitting** to decompose the complex Navier-Stokes equations into three sequential sub-steps per timestep:

```
Step 1: ADVECTION     → Transport quantities along velocity field
Step 2: FORCES        → Apply buoyancy, sources, vorticity confinement
Step 3: PROJECTION    → Enforce incompressibility constraint
```

This approach (also called fractional step method) allows each physical process to be handled with the most appropriate numerical method.

#### Implementation: [src/core/solver.py](src/core/solver.py)

```python
def step(self):
    # Step 1: Advect all quantities
    self._advect_fields()

    # Step 2: Apply forces
    self._apply_forces()

    # Step 3: Enforce incompressibility
    self._project()
```

### Advection Schemes

Advection moves quantities through the velocity field. Two schemes are implemented:

#### 1. Semi-Lagrangian Advection (First-Order)

**Theory**: Instead of pushing quantities forward (unstable), trace particles **backward** in time:

```
φⁿ⁺¹(x) = φⁿ(x - u(x)·dt)
```

For each grid point:
1. Trace backward: `x_prev = x - u(x)·dt`
2. Interpolate: `φ(x_prev)` using bilinear interpolation
3. Assign: `φ_new(x) = φ(x_prev)`

**Advantages**:
- Unconditionally stable (no CFL constraint)
- Simple to implement

**Disadvantages**:
- First-order accurate (O(dt))
- Introduces numerical diffusion

#### 2. MacCormack Scheme (Second-Order)

**Theory**: A predictor-corrector method that estimates and removes first-order error:

```
φ̂ = forward_advect(φⁿ, dt)              [Predictor]
φ̃ = backward_advect(φ̂, dt)             [Corrector]
error = (φⁿ - φ̃) / 2
φⁿ⁺¹ = φ̂ + error                        [Error correction]
```

**Clamping**: To prevent overshoots near discontinuities, the result is clamped to the min/max values in the local 3×3 neighborhood.

**Advantages**:
- Second-order accurate (O(dt²))
- Sharper features, less diffusion
- Still unconditionally stable

**Implementation**: [src/core/advection.py](src/core/advection.py)

The MacCormack scheme is the default, with optional clamping (can be disabled for 30-50% speedup).

### Force Application

#### Buoyancy Force (Boussinesq)

Adds upward acceleration proportional to density:

```
v_new = v + α·ρ·dt
```

Only the vertical velocity component is affected. This creates the characteristic rising smoke plume.

**Implementation**: [src/core/forces.py:40-51](src/core/forces.py#L40-L51)

#### Density Sources

Continuous Gaussian injection at specified locations:

```
source(x,y) = A·exp(-r² / (2·σ²))
```

where r is distance from source center and σ controls the spread.

**Implementation**: [src/core/forces.py:54-77](src/core/forces.py#L54-L77)

#### Vorticity Confinement

**Problem**: Numerical diffusion from discretization damps small-scale turbulent details.

**Solution**: Artificially restore vorticity (rotation) lost to diffusion.

**Algorithm**:
1. Compute vorticity (scalar in 2D):
   ```
   ω = ∂v/∂x - ∂u/∂y
   ```

2. Compute gradient direction:
   ```
   N = ∇|ω| / |∇|ω||
   ```

3. Apply force perpendicular to gradient:
   ```
   f_conf = ε·h·(N × ω)
   ```

where ε is the confinement strength and h is grid spacing.

**Effect**: Amplifies swirling motions, creating more visually interesting turbulence.

**Implementation**: [src/core/forces.py:80-127](src/core/forces.py#L80-L127)

### Pressure Projection

The most mathematically sophisticated component, ensuring the velocity field remains **divergence-free** (incompressible).

#### Theory: Helmholtz-Hodge Decomposition

Any vector field can be decomposed into divergence-free and curl-free parts:

```
u = u_div-free + ∇φ
```

The projection step removes the gradient component (∇φ) by solving for pressure.

#### Three-Step Algorithm

**Step 1: Compute Divergence**

```
div(u) = ∂u/∂x + ∂v/∂y
```

Uses central differences in the interior:
```
div[i,j] = (u[i+1,j] - u[i-1,j])/(2h) + (v[i,j+1] - v[i,j-1])/(2h)
```

**Step 2: Solve Pressure Poisson Equation**

To find pressure field p that eliminates divergence:

```
∇²p = (ρ/dt)·∇·u
```

Discretized using a 5-point Laplacian stencil:

```
(p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1] - 4p[i,j]) / h² = (ρ/dt)·div[i,j]
```

This creates a sparse linear system **Ap = b** where:
- **A** is the Laplacian matrix (sparse, pentadiagonal structure)
- **b** is the right-hand side derived from divergence
- **p** is the unknown pressure field (flattened)

**Boundary Conditions**: Zero Dirichlet (p = 0) at domain edges. The origin is pinned to break pressure degeneracy (pressure is defined up to a constant).

**Solver**: Conjugate Gradient (CG) iterative method via `scipy.sparse.linalg.cg`. Typically converges in 10-50 iterations with tolerance 1e-3.

**Step 3: Apply Pressure Gradient**

Subtract the pressure gradient to make velocity divergence-free:

```
u_new = u - (dt/ρ)·∇p
```

Using central differences for the gradient:
```
u_new[i,j] = u[i,j] - (dt/ρ)·(p[i+1,j] - p[i-1,j])/(2h)
v_new[i,j] = v[i,j] - (dt/ρ)·(p[i,j+1] - p[i,j-1])/(2h)
```

**Verification**: After projection, `∇·u_new ≈ 0` (within solver tolerance).

**Implementation**: [src/core/projections.py](src/core/projections.py)

The Laplacian matrix can be pre-built and cached for significant speedup (15-20%).

## Code Architecture

### Module Organization

```
/
├── config.py                    # SimParams dataclass, scaling guide
├── main.py                      # Entry point, visualization loop
├── src/
│   ├── core/                    # Core simulation algorithms
│   │   ├── grid.py              # FluidGrid: velocity + density storage
│   │   ├── solver.py            # FluidSolver: main simulation loop
│   │   ├── advection.py         # Semi-Lagrangian & MacCormack
│   │   ├── forces.py            # Buoyancy, sources, vorticity
│   │   ├── projections.py       # Pressure projection (Poisson solver)
│   │   └── grid_viz.py          # ASCII debug visualization
│   └── utils/                   # Utility functions
│       ├── interpolation.py     # Bilinear interpolation
│       └── boundary.py          # Boundary conditions
└── tests/                       # Unit tests (pytest)
```

### Key Classes

#### FluidGrid ([src/core/grid.py](src/core/grid.py))

Stores the simulation state:
- `velocity: ndarray[H, W, 2]` - Velocity components (u, v)
- `density: ndarray[H, W]` - Scalar density field

Provides convenient accessors:
```python
grid.u  # u-component (horizontal velocity)
grid.v  # v-component (vertical velocity)
```

#### FluidSolver ([src/core/solver.py](src/core/solver.py))

Orchestrates the simulation:
- `step()`: Execute one timestep (advect → forces → project)
- Manages cached matrices and pre-computed masks
- Handles boundary conditions and stability constraints

#### SimParams ([config.py](config.py))

Configuration dataclass containing all simulation parameters:
- Grid dimensions (height, width, spacing)
- Timestep (dt)
- Physical parameters (buoyancy, vorticity, viscosity)
- Source configuration
- Numerical options (advection scheme, solver tolerance)
- Performance toggles

## Numerical Methods

### Spatial Discretization

- **Grid Type**: Collocated (all quantities at cell centers)
  - Alternative: Staggered MAC grid (u/v at cell faces) - more accurate but complex
- **Finite Differences**: Second-order central differences for gradients
  - Interior: `∂f/∂x ≈ (f[i+1] - f[i-1]) / (2h)`
  - Boundaries: One-sided differences
- **Interpolation**: Bilinear for off-grid sampling during advection

### Temporal Discretization

- **Advection**: Semi-Lagrangian (implicit backward tracing)
- **Forces**: Explicit forward Euler
- **Projection**: Implicit (solves linear system)

### Boundary Conditions

Three types supported ([src/utils/boundary.py](src/utils/boundary.py)):

1. **No-Slip** (default): Zero velocity at walls
   - Most realistic for solid boundaries
   - `u[boundary] = 0`

2. **Free-Slip**: Zero normal velocity, free tangential
   - Frictionless walls
   - `u_normal[boundary] = 0`, `u_tangent` unconstrained

3. **Periodic**: Wrap-around topology
   - Toroidal domain (useful for turbulence studies)
   - `u[0] = u[end]`

### Stability Controls

To prevent numerical blow-up:
- **Velocity clamping**: Hard cap on magnitude (default: 5.0)
- **Density dissipation**: Multiply by 0.995 per timestep
- **Velocity damping**: Multiply by damping factor (default: 0.99)
- **Boundary damping**: Additional damping near edges (smooth mask)
- **MacCormack clamping**: Prevents overshoots in advection

## Performance Optimizations

### Implemented Optimizations

1. **Cached Laplacian Matrix** (15-20% speedup)
   - Pre-build sparse matrix once
   - Toggle: `cache_laplacian_matrix = True` in [config.py](config.py)

2. **Optional MacCormack Clamping** (30-50% speedup when disabled)
   - Clamping requires 3×3 neighborhood checks per cell
   - Disable for faster, slightly less stable advection
   - Toggle: `use_maccormack_clamping = True` in [config.py](config.py)

3. **Vectorized Operations**
   - Pure NumPy array operations (no Python loops)
   - Broadcasting for element-wise computations

4. **Cached Velocity Magnitude**
   - Store `sqrt(u² + v²)` between timesteps
   - Invalidate when velocity changes

5. **Pre-computed Boundary Masks**
   - Boundary damping mask computed once at initialization
   - Applied via element-wise multiplication

### Performance Scaling

From [config.py](config.py) scaling guide:

| Resolution | Cells   | Relative Cost | Use Case          |
|------------|---------|---------------|-------------------|
| 36 × 64    | 2,304   | 1×            | Fast preview      |
| 72 × 128   | 9,216   | 4×            | Default           |
| 144 × 256  | 36,864  | 16×           | High detail       |
| 288 × 512  | 147,456 | 64×           | Very high detail  |

Cost scales roughly as O(N²) where N is linear resolution.

## Running the Simulation

### Requirements

```bash
pip install numpy scipy matplotlib
```

- **NumPy**: Array operations and core computation
- **SciPy**: Sparse matrix solver (Conjugate Gradient)
- **Matplotlib**: Real-time visualization

### Execution

```bash
python main.py
```

Creates two side-by-side plots:
- **Left**: Density field (smoke concentration)
- **Right**: Velocity magnitude (flow speed)

Animation runs at 20 FPS for 500 frames (25 seconds).

### Testing

```bash
pytest tests/
```

Tests cover:
- Pressure projection (divergence reduction)
- Force application (buoyancy, sources)
- Interpolation accuracy
- Boundary conditions

## Configuration

Edit [config.py](config.py) to modify simulation parameters:

### Grid & Time
```python
height: int = 72          # Grid height
width: int = 128          # Grid width
h: float = 1.0            # Grid spacing
dt: float = 0.1           # Timestep
```

### Physics
```python
buoyancy_alpha: float = 1.0      # Buoyancy strength
vorticity_epsilon: float = 0.05  # Vorticity confinement
```

### Sources
```python
source_strength: float = 5.0     # Density injection rate
source_radius: float = 3.0       # Gaussian spread
source_position: tuple = (0.5, 0.1)  # (x, y) normalized coords
```

### Numerical Options
```python
advection_scheme: str = "maccormack"     # or "semi_lagrangian"
use_maccormack_clamping: bool = True     # Stability vs speed
pressure_iterations: int = 100           # Max CG iterations
pressure_tolerance: float = 1e-3         # CG convergence
cache_laplacian_matrix: bool = True      # Speedup toggle
```

### Presets

Three presets available:
- `SimParams.slow_detailed()`: High quality, slow (144×256 grid)
- `SimParams.fast_preview()`: Low quality, fast (36×64 grid)
- `SimParams.default()`: Balanced (72×128 grid)

## References

### Theory & Algorithms

1. **Stam, J.** (1999). *Stable Fluids*. SIGGRAPH 99.
   - Original stable fluids paper, foundation for this implementation

2. **Fedkiw, R., Stam, J., & Jensen, H. W.** (2001). *Visual Simulation of Smoke*. SIGGRAPH 01.
   - Vorticity confinement technique

3. **Selle, A., Fedkiw, R., Kim, B., Liu, Y., & Rossignac, J.** (2008). *An Unconditionally Stable MacCormack Method*. Journal of Scientific Computing.
   - MacCormack advection with clamping

4. **Bridson, R.** (2015). *Fluid Simulation for Computer Graphics* (2nd ed.). CRC Press.
   - Comprehensive textbook on grid-based fluid simulation

### Numerical Methods

5. **Chorin, A. J.** (1968). *Numerical Solution of the Navier-Stokes Equations*. Mathematics of Computation.
   - Original projection method paper

6. **Trefethen, L. N. & Bau, D.** (1997). *Numerical Linear Algebra*. SIAM.
   - Iterative solvers (Conjugate Gradient)

---

**License**: MIT
**Last Updated**: December 2025
