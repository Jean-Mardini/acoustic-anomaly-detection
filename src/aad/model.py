from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MemoryModule(nn.Module):
    """
    Stores N memory slots (prototypes of normal patterns).
    Each input latent is reconstructed as a weighted sum of memory slots,
    where weights are based on cosine similarity.
    Forces reconstruction through the normal prototype space.
    """

    def __init__(self, num_slots: int = 100, latent_dim: int = 64) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_slots, latent_dim))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: [B, latent_dim]
        # Normalize both z and memory for cosine similarity
        z_norm = F.normalize(z, dim=1)                          # [B, latent_dim]
        mem_norm = F.normalize(self.memory, dim=1)              # [N, latent_dim]

        # Attention weights: similarity of z to each memory slot
        attn = torch.softmax(z_norm @ mem_norm.T, dim=1)       # [B, N]

        # Reconstructed latent: weighted sum of memory slots
        z_hat = attn @ self.memory                              # [B, latent_dim]

        return z_hat, attn


class ConvAutoencoder(nn.Module):
    """
    Input shape: [B, 1, 128, 64]
    Conv autoencoder with memory module between encoder and decoder.
    """

    def __init__(self, latent_dim: int = 64, base_channels: int = 64, num_memory_slots: int = 100) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
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
        # After 4 stride-2 convs on [128×64] input: [c4 × 8 × 4]
        self.flat_dim = c4 * 8 * 4
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.memory = MemoryModule(num_slots=num_memory_slots, latent_dim=latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.c4 = c4
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
        h = self.encoder(x).flatten(1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, self.c4, 8, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        z_hat, attn = self.memory(z)
        out = self.decode(z_hat)
        return out, z, z_hat, attn
