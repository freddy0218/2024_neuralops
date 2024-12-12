import numpy as np
import torch

def generate_3d_volume(wavenumber, grid_size=(32, 32, 32)):
    depth, height, width = grid_size
    if wavenumber == 0:
        # DC component: Uniform value
        volume = torch.ones(grid_size, dtype=torch.float32)
    else:
        z, y, x = np.meshgrid(
            np.linspace(0, 1, depth, endpoint=False),
            np.linspace(0, 1, height, endpoint=False),
            np.linspace(0, 1, width, endpoint=False),
            indexing='ij',
        )
        # Create sinusoidal pattern
        volume = np.sin(2 * np.pi * wavenumber * z) + \
                 np.sin(2 * np.pi * wavenumber * y) + \
                 np.sin(2 * np.pi * wavenumber * x)
        volume = torch.tensor(volume, dtype=torch.float32)
    return volume