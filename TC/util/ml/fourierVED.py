import torch
import torch.nn as nn
import torch.fft

class FourierLayer(nn.Module):
    def __init__(self, max_wavenumber):
        """
        Initialize the Fourier layer.
        Args:
            max_wavenumber (int): Maximum wavenumber to consider for features.
        """
        super(FourierLayer, self).__init__()
        self.max_wavenumber = max_wavenumber

    def forward(self, x):
        """
        Compute the Fourier-transformed features for the given volume.
        Args:
            x (Tensor): Input tensor with shape (batch_size, channels, depth, height, width).
        Returns:
            Tensor: Wavenumber-based features with shape (batch_size, max_wavenumber + 1).
        """
        # Apply 3D Fourier transform
        x_fft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="forward")  # Shape: (B, C, D, H, W//2+1)
        magnitude = torch.abs(x_fft)  # Use magnitude of Fourier coefficients

        # Compute contributions for wavenumbers up to max_wavenumber
        wavenumber_features = []
        for k in range(self.max_wavenumber + 1):
            mask = self._wavenumber_mask(x_fft.shape, k).to(x.device)  # Get mask for current wavenumber
            wavenumber_contribution = (magnitude * mask).sum(dim=(-3, -2, -1))
            wavenumber_features.append(wavenumber_contribution)

        return torch.stack(wavenumber_features, dim=-1)  # Shape: (B, max_wavenumber + 1)
    
    def _wavenumber_mask(self, shape, k):
        """
        Create a mask for isolating a specific wavenumber.
        Args:
        shape (tuple): Shape of the Fourier-transformed data (B, C, D, H, W//2+1).
        k (int): Wavenumber to isolate.
        Returns:
        Tensor: Mask with the same spatial dimensions as the input.
        """
        D, H, W_half = shape[-3], shape[-2], shape[-1]
        W = 2 * (W_half - 1)  # Original spatial width before FFT
        coords_d = torch.fft.fftfreq(D).view(-1, 1, 1).to(dtype=torch.float32)
        coords_h = torch.fft.fftfreq(H).view(1, -1, 1).to(dtype=torch.float32)
        coords_w = torch.fft.rfftfreq(W).view(1, 1, -1).to(dtype=torch.float32)

        # Compute Euclidean norm of frequencies
        coords = torch.sqrt(coords_d**2 + coords_h**2 + coords_w**2)
    
        # Improved mask logic for isolating wavenumber k
        mask = (coords.round() == k).float()
        return mask