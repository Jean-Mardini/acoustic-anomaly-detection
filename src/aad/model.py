from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import math


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


class TransformerAutoencoder(nn.Module):
    """
    Patch-based Transformer Autoencoder with memory module.
    Input shape: [B, 1, 128, 64]
    Splits spectrogram into patches, encodes with self-attention, bottlenecks
    through memory module, then decodes back to original shape.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_memory_slots: int = 100,
        img_h: int = 128,
        img_w: int = 64,
        patch_h: int = 8,
        patch_w: int = 8,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.img_h = img_h
        self.img_w = img_w
        self.num_patches_h = img_h // patch_h
        self.num_patches_w = img_w // patch_w
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = patch_h * patch_w
        self.d_model = d_model

        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers, enable_nested_tensor=False)

        self.fc_enc = nn.Linear(d_model, latent_dim)

        self.memory = MemoryModule(num_slots=num_memory_slots, latent_dim=latent_dim)

        self.fc_dec = nn.Linear(latent_dim, d_model)
        dec_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerEncoder(dec_layer, num_decoder_layers, enable_nested_tensor=False)
        self.patch_proj = nn.Linear(d_model, self.patch_dim)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.squeeze(1)  # [B, H, W]
        x = x.unfold(1, self.patch_h, self.patch_h).unfold(2, self.patch_w, self.patch_w)
        # [B, num_patches_h, num_patches_w, patch_h, patch_w]
        x = x.contiguous().view(B, self.num_patches, self.patch_dim)
        return x  # [B, num_patches, patch_dim]

    def _from_patches(self, patches: torch.Tensor) -> torch.Tensor:
        B = patches.shape[0]
        x = patches.view(B, self.num_patches_h, self.num_patches_w, self.patch_h, self.patch_w)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.img_h, self.img_w)
        return x.unsqueeze(1)  # [B, 1, H, W]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._to_patches(x)                              # [B, num_patches, patch_dim]
        tokens = self.patch_embed(patches) + self.pos_embed        # [B, num_patches, d_model]
        tokens = self.transformer_encoder(tokens)                  # [B, num_patches, d_model]
        z = tokens.mean(dim=1)                                     # [B, d_model] mean pooling
        return self.fc_enc(z)                                      # [B, latent_dim]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).unsqueeze(1).expand(-1, self.num_patches, -1)  # [B, num_patches, d_model]
        h = h + self.pos_embed                                             # add positional info
        h = self.transformer_decoder(h)                                    # [B, num_patches, d_model]
        patches = self.patch_proj(h)                                       # [B, num_patches, patch_dim]
        return self._from_patches(patches)                                 # [B, 1, H, W]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        z_hat, attn = self.memory(z)
        out = self.decode(z_hat)
        return out, z, z_hat, attn
