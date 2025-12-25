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
    if boundary != 'clamp':
        raise ValueError(f"Unsupported boundary condition: {boundary}. Only 'clamp' is supported.")


    height, width = field.shape
    
    # Handle boundary conditions

    x_safe = np.clip(x, 0, width - 1)
    y_safe = np.clip(y, 0, height - 1)
    
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
    
    return result
    