"""Convolutional autoencoder on fixed-size log-mel spectrograms."""

from __future__ import annotations

import torch
from torch import nn

from backend.ml.features import DEFAULT_MEL_TIME_FRAMES, N_MELS


class MelConvAutoencoder(nn.Module):
    """
    Conv2d AE for input shape (B, 1, n_mels, time).

    Default 128x128: four stride-2 convs -> 8x8 spatial, 256 channels.
    """

    def __init__(
        self,
        *,
        n_mels: int = N_MELS,
        time_frames: int = DEFAULT_MEL_TIME_FRAMES,
        latent_dim: int = 256,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if n_mels != 128 or time_frames != 128:
            raise ValueError("This architecture expects n_mels=128 and time_frames=128.")
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.n_mels = n_mels
        self.time_frames = time_frames
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self._c4 = c4
        self._flat = c4 * 8 * 8
        self.fc_enc = nn.Linear(self._flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self._flat)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, self._c4, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
