import numpy as np

def bilinear_interpolate(field, x, y, boundary='clamp'):
    """
    Interpolate a 2D field at non-integer positions using bilinear interpolation.

    Args:
        field: 2D numpy array [H, W]
        x: X-coordinates (float or array)
        y: Y-coordinates (float or array)
        boundary: 'clamp', 'periodic', or 'zero'

    Returns:
        Interpolated values at (x, y)
    """
    height, width = field.shape

    # Handle boundary conditions based on type
    if boundary == 'clamp':
        # Clamp to edges
        x_safe = np.clip(x, 0, width - 1)
        y_safe = np.clip(y, 0, height - 1)
    elif boundary == 'periodic':
        # Wrap around (toroidal topology)
        x_safe = np.mod(x, width)
        y_safe = np.mod(y, height)
    elif boundary == 'zero':
        # Out-of-bounds will be handled below by extrapolating as zero
        x_safe = x
        y_safe = y
    else:
        raise ValueError(f"Unsupported boundary condition: {boundary}. Options: 'clamp', 'periodic', 'zero'")
    
    # Find integer grid coordinates of lower-left corner
    x0 = np.floor(x_safe).astype(int)
    y0 = np.floor(y_safe).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clip x1, y1 to stay in bounds
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    
    # Compute interpolation weights
    alpha = x_safe - x0  # Weight in x-direction
    beta = y_safe - y0   # Weight in y-direction
    
    # Get values at 4 corners
    Q00 = field[y0, x0]  # Lower-left
    Q10 = field[y0, x1]  # Lower-right
    Q01 = field[y1, x0]  # Upper-left
    Q11 = field[y1, x1]  # Upper-right
    
    # Bilinear interpolation formula
    result = (1 - alpha) * (1 - beta) * Q00 + alpha * (1 - beta) * Q10 + (1 - alpha) * beta * Q01 + alpha * beta * Q11

    # For 'zero' boundary: set result to zero where original coords were out of bounds
    if boundary == 'zero':
        out_of_bounds = (x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)
        if np.any(out_of_bounds):
            result = np.where(out_of_bounds, 0.0, result)

    return result
    