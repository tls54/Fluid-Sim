import numpy as np

def print_velocity_grid(velocity, label="Velocity Grid"):
    """
    Print velocity field in an intuitive spatial layout.
    Shows arrows/vectors at each grid position.
    """
    height, width, _ = velocity.shape
    
    print(f"\n{label}")
    print("=" * (width * 12))
    
    # Print from top to bottom (reverse y for visual clarity)
    for j in range(height - 1, -1, -1):
        print(f"y={j} |", end=" ")
        for i in range(width):
            u = velocity[j, i, 0]
            v = velocity[j, i, 1]
            print(f"({u:5.2f},{v:5.2f})", end=" ")
        print()
    
    # Print x-axis labels
    print("    " + "-" * (width * 12))
    print("     ", end="")
    for i in range(width):
        print(f"   x={i}    ", end="")
    print("\n")


def print_velocity_arrows(velocity, label="Velocity Arrows"):
    """
    Print velocity field using ASCII arrows for direction.
    """
    height, width, _ = velocity.shape
    
    print(f"\n{label}")
    print("=" * (width * 8))
    
    for j in range(height - 1, -1, -1):
        print(f"y={j} |", end=" ")
        for i in range(width):
            u = velocity[j, i, 0]
            v = velocity[j, i, 1]
            
            # Convert to arrow
            arrow = velocity_to_arrow(u, v)
            print(f"{arrow:^6}", end=" ")
        print()
    
    print("    " + "-" * (width * 8))
    print("     ", end="")
    for i in range(width):
        print(f" x={i}  ", end="")
    print("\n")


def velocity_to_arrow(u, v, threshold=0.1):
    """Convert velocity components to ASCII arrow."""
    magnitude = np.sqrt(u**2 + v**2)
    
    if magnitude < threshold:
        return "  ·  "  # Stationary
    
    # Determine primary direction
    if abs(u) > abs(v):
        return "  →  " if u > 0 else "  ←  "
    else:
        return "  ↑  " if v > 0 else "  ↓  "


def print_density_grid(density, label="Density Grid"):
    """
    Print density field with values.
    """
    height, width = density.shape
    
    print(f"\n{label}")
    print("=" * (width * 8))
    
    for j in range(height - 1, -1, -1):
        print(f"y={j} |", end=" ")
        for i in range(width):
            d = density[j, i]
            print(f"{d:6.3f}", end=" ")
        print()
    
    print("    " + "-" * (width * 8))
    print("     ", end="")
    for i in range(width):
        print(f"  x={i} ", end="")
    print("\n")


def print_density_heatmap(density, label="Density Heatmap"):
    """
    Print density as ASCII heatmap with intensity characters.
    """
    height, width = density.shape
    
    print(f"\n{label}")
    print("=" * (width * 3))
    
    # Normalize to 0-1 range
    d_min, d_max = density.min(), density.max()
    if d_max > d_min:
        normalized = (density - d_min) / (d_max - d_min)
    else:
        normalized = density
    
    # Intensity characters from light to dark
    chars = " .:-=+*#%@"
    
    for j in range(height - 1, -1, -1):
        print(f"y={j} |", end=" ")
        for i in range(width):
            idx = int(normalized[j, i] * (len(chars) - 1))
            print(chars[idx] * 2, end=" ")
        print()
    
    print("    " + "-" * (width * 3))
    print("     ", end="")
    for i in range(width):
        print(f"x={i}", end=" ")
    print(f"\n    Scale: {chars[0]}=min, {chars[-1]}=max\n")