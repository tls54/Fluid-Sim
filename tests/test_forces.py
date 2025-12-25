import numpy as np
import matplotlib.pyplot as plt
from src.core.forces import apply_buoyancy, add_density_source


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
    plt.savefig('output_figs/density_source_test.png')
    print("✓ Saved visualization to 'output_figs/density_source_test.png'")
    plt.close()

    print("✓ Density source test passed!")


if __name__ == "__main__":
    # Run tests
    test_buoyancy()
    test_density_source()

    print("\n" + "="*60)
    print("✓✓✓ All force tests passed! ✓✓✓")
    print("="*60)
