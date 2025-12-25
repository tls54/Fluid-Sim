"""
Boundary condition utilities for fluid simulation.

Provides unified boundary handling across advection, projection, and solver steps.
"""
import numpy as np


def apply_boundary_conditions(velocity, boundary_type='no-slip'):
    """
    Apply boundary conditions to velocity field.

    Args:
        velocity: Velocity field [H, W, 2]
        boundary_type: Type of boundary ('no-slip', 'free-slip', 'periodic')

    Returns:
        velocity: Velocity with boundary conditions applied
    """
    if boundary_type == 'no-slip':
        return apply_no_slip_boundary(velocity)
    elif boundary_type == 'free-slip':
        return apply_free_slip_boundary(velocity)
    elif boundary_type == 'periodic':
        return apply_periodic_boundary(velocity)
    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}. Options: 'no-slip', 'free-slip', 'periodic'")


def apply_no_slip_boundary(velocity):
    """
    Apply no-slip boundary conditions (zero velocity at walls).

    Most physically realistic for solid walls.

    Args:
        velocity: Velocity field [H, W, 2]

    Returns:
        velocity: Modified velocity with zero velocity at boundaries
    """
    # Zero normal and tangential components at all walls
    velocity[0, :, :] = 0      # Bottom wall
    velocity[-1, :, :] = 0     # Top wall
    velocity[:, 0, :] = 0      # Left wall
    velocity[:, -1, :] = 0     # Right wall

    return velocity


def apply_free_slip_boundary(velocity):
    """
    Apply free-slip boundary conditions (zero normal component only).

    Allows tangential flow along walls (frictionless).

    Args:
        velocity: Velocity field [H, W, 2]

    Returns:
        velocity: Modified velocity with zero normal component at boundaries
    """
    # Zero normal component only
    velocity[0, :, 1] = 0      # Bottom wall - zero v
    velocity[-1, :, 1] = 0     # Top wall - zero v
    velocity[:, 0, 0] = 0      # Left wall - zero u
    velocity[:, -1, 0] = 0     # Right wall - zero u

    return velocity


def apply_periodic_boundary(velocity):
    """
    Apply periodic boundary conditions (wrap-around).

    Opposite edges are connected (toroidal topology).

    Args:
        velocity: Velocity field [H, W, 2]

    Returns:
        velocity: Modified velocity with periodic wrapping
    """
    # Copy values from opposite edges
    velocity[0, :, :] = velocity[-2, :, :]    # Bottom = second-to-top
    velocity[-1, :, :] = velocity[1, :, :]    # Top = second-to-bottom
    velocity[:, 0, :] = velocity[:, -2, :]    # Left = second-to-right
    velocity[:, -1, :] = velocity[:, 1, :]    # Right = second-to-left

    return velocity


def compute_gradient_with_boundaries(field, h, boundary_type='no-slip'):
    """
    Compute gradient ∇field with appropriate boundary stencils.

    Args:
        field: Scalar field [H, W]
        h: Grid spacing
        boundary_type: Boundary condition type

    Returns:
        grad_x: ∂field/∂x [H, W]
        grad_y: ∂field/∂y [H, W]
    """
    height, width = field.shape
    grad_x = np.zeros((height, width))
    grad_y = np.zeros((height, width))

    # Interior: central differences (second-order accurate)
    grad_x[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * h)
    grad_y[1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2 * h)

    if boundary_type == 'periodic':
        # Periodic: wrap around for boundary points
        grad_x[:, 0] = (field[:, 1] - field[:, -2]) / (2 * h)
        grad_x[:, -1] = (field[:, 0] - field[:, -2]) / (2 * h)
        grad_y[0, :] = (field[1, :] - field[-2, :]) / (2 * h)
        grad_y[-1, :] = (field[0, :] - field[-2, :]) / (2 * h)
    else:
        # No-slip and free-slip: one-sided differences at boundaries
        # Left edge (i=0): forward difference
        grad_x[1:-1, 0] = (field[1:-1, 1] - field[1:-1, 0]) / h
        # Right edge (i=-1): backward difference
        grad_x[1:-1, -1] = (field[1:-1, -1] - field[1:-1, -2]) / h

        # Bottom edge (j=0): forward difference
        grad_y[0, 1:-1] = (field[1, 1:-1] - field[0, 1:-1]) / h
        # Top edge (j=-1): backward difference
        grad_y[-1, 1:-1] = (field[-1, 1:-1] - field[-2, 1:-1]) / h

        # Corners: both one-sided
        grad_x[0, 0] = (field[0, 1] - field[0, 0]) / h
        grad_x[0, -1] = (field[0, -1] - field[0, -2]) / h
        grad_x[-1, 0] = (field[-1, 1] - field[-1, 0]) / h
        grad_x[-1, -1] = (field[-1, -1] - field[-1, -2]) / h

        grad_y[0, 0] = (field[1, 0] - field[0, 0]) / h
        grad_y[0, -1] = (field[1, -1] - field[0, -1]) / h
        grad_y[-1, 0] = (field[-1, 0] - field[-2, 0]) / h
        grad_y[-1, -1] = (field[-1, -1] - field[-2, -1]) / h

    return grad_x, grad_y


def compute_divergence_with_boundaries(velocity, h, boundary_type='no-slip'):
    """
    Compute divergence ∇·v with appropriate boundary stencils.

    Args:
        velocity: Velocity field [H, W, 2]
        h: Grid spacing
        boundary_type: Boundary condition type

    Returns:
        divergence: Divergence field [H, W]
    """
    u = velocity[:, :, 0]
    v = velocity[:, :, 1]

    # Use gradient function for consistency
    du_dx, _ = compute_gradient_with_boundaries(u, h, boundary_type)
    _, dv_dy = compute_gradient_with_boundaries(v, h, boundary_type)

    divergence = du_dx + dv_dy

    return divergence
