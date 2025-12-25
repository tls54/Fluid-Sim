import numpy as np


def apply_buoyancy(velocity, density, alpha, dt):
    """
    Apply buoyancy force to velocity field.
    
    Buoyancy makes hot/light fluid rise and cold/heavy fluid sink.
    Uses Boussinesq approximation: only vertical velocity is affected.
    
    Args:
        velocity: Velocity field [H, W, 2]
        density: Density/temperature field [H, W]
        alpha: Buoyancy strength coefficient
        dt: Timestep
    
    Returns:
        velocity: Updated velocity field [H, W, 2]
    """
    # TODO: Add -alpha * density * dt to v-component
    velocity[:, :, 1] += alpha * density * dt
    return velocity


def add_density_source(density, source_x, source_y, source_radius, source_strength, dt):
    """
    Add a continuous source of density (smoke/heat injection).
    
    Creates a Gaussian blob of density centered at (source_x, source_y).
    
    Args:
        density: Density field [H, W]
        source_x: X-coordinate of source center
        source_y: Y-coordinate of source center
        source_radius: Standard deviation of Gaussian (spread)
        source_strength: Peak intensity of source
        dt: Timestep
    
    Returns:
        density: Updated density field [H, W]
    """
    height, width = density.shape
    
    # TODO: For each grid point, compute distance from source
    # TODO: Add Gaussian: density += strength * exp(-r²/radius²) * dt
    
    # Create coordinate grids
    y_grid, x_grid = np.indices((height, width))

    # Distance from source
    dx = x_grid - source_x
    dy = y_grid - source_y
    r_squared = dx**2 + dy**2

    # Add Gaussian blob
    source = source_strength * np.exp(-r_squared / (2 * source_radius**2))
    density += source * dt
    return density



def vorticity_confinement(velocity, epsilon, h, dt):
    """
    Add vorticity confinement force to maintain turbulence.
    
    TODO: Implement later if needed for visual enhancement.
    
    Args:
        velocity: Velocity field [H, W, 2]
        epsilon: Confinement strength
        h: Grid spacing
        dt: Timestep
    
    Returns:
        velocity: Updated velocity field [H, W, 2]
    """
    # Not implemented yet
    return velocity


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def test_buoyancy():
        """Test that buoyancy adds upward force."""
        print("Testing buoyancy force...")
        
        # Create velocity and density fields
        velocity = np.zeros((20, 20, 2))
        density = np.zeros((20, 20))
        
        # Add density blob at center
        density[10, 10] = 1.0
        
        # Apply buoyancy
        alpha = 0.1
        dt = 0.1
        velocity = apply_buoyancy(velocity, density, alpha, dt)
        
        # Check that v-component increased at center
        assert velocity[10, 10, 1] > 0, "Buoyancy should add upward velocity"
        assert velocity[10, 10, 0] == 0, "Buoyancy shouldn't affect u-component"
        
        print(f"✓ Buoyancy force at center: v = {velocity[10, 10, 1]:.4f}")
        print("✓ Buoyancy test passed!")
    
    
    def test_density_source():
        """Test that density source adds mass."""
        print("\nTesting density source...")
        
        density = np.zeros((30, 30))
        
        # Add source at center
        density = add_density_source(
            density,
            source_x=15,
            source_y=15,
            source_radius=3.0,
            source_strength=1.0,
            dt=0.1
        )
        
        # Check that density was added
        assert density[15, 15] > 0, "Source should add density at center"
        assert density.sum() > 0, "Total density should increase"
        
        print(f"✓ Peak density: {density.max():.4f}")
        print(f"✓ Total density: {density.sum():.4f}")
        
        # Visualize
        plt.figure(figsize=(6, 6))
        plt.imshow(density, origin='lower', cmap='hot')
        plt.colorbar(label='Density')
        plt.title('Density Source Test')
        plt.savefig('density_source_test.png')
        print("✓ Saved visualization to 'density_source_test.png'")
        plt.show()
        
        print("✓ Density source test passed!")
    
    
    # Run tests
    test_buoyancy()
    test_density_source()
    
    print("\n" + "="*60)
    print("✓✓✓ All force tests passed! ✓✓✓")
    print("="*60)