import numpy as np
from src.utils.interpolation import bilinear_interpolate

def test_exact_grid_points():
    """Interpolating at grid points should return exact values."""
    field = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
    
    # Test center point
    assert bilinear_interpolate(field, 1.0, 1.0) == 5.0
    
    # Test corners
    assert bilinear_interpolate(field, 0.0, 0.0) == 1.0
    assert bilinear_interpolate(field, 2.0, 2.0) == 9.0
    
    print("✓ Exact grid points test passed")

def test_halfway_points():
    """Interpolating halfway should average neighbors."""
    field = np.array([[0.0, 0.0, 0.0],
                      [0.0, 4.0, 0.0],
                      [0.0, 0.0, 0.0]])
    
    # Halfway between center (4.0) and left (0.0) should be 2.0
    assert bilinear_interpolate(field, 0.5, 1.0) == 2.0
    
    # Halfway between center (4.0) and bottom (0.0) should be 2.0
    assert bilinear_interpolate(field, 1.0, 0.5) == 2.0
    
    print("✓ Halfway points test passed")

def test_boundary_clamping():
    """Out-of-bounds positions should clamp to edges."""
    field = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
    
    # Beyond right edge should clamp to x=1
    result = bilinear_interpolate(field, 10.0, 0.0, boundary='clamp')
    assert result == 2.0
    
    # Beyond top edge should clamp to y=1
    result = bilinear_interpolate(field, 0.0, 10.0, boundary='clamp')
    assert result == 3.0
    
    print("✓ Boundary clamping test passed")

if __name__ == "__main__":
    test_exact_grid_points()
    test_halfway_points()
    test_boundary_clamping()
    print("\n✓✓✓ All interpolation tests passed! ✓✓✓")