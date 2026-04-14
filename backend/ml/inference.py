"""Load MelConvAutoencoder and score clips; per-mel-band errors for explanations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from backend.ml.features import DEFAULT_MEL_TIME_FRAMES, wav_to_fixed_mel
from backend.ml.model import MelConvAutoencoder


def load_mel_conv_ae(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[MelConvAutoencoder, dict]:
    """Returns (model, checkpoint dict) so callers can read model_kwargs (time_frames, etc.)."""
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location="cpu")
    kwargs = ckpt.get("model_kwargs", {})
    model = MelConvAutoencoder(**kwargs)
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_mel_reconstruction(
    model: MelConvAutoencoder,
    mel: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns (scalar_mse, per_frequency_mse[n_mels], residual_mel same shape as input).
    mel: (1, n_mels, time) or (n_mels, time) float32
    """
    if isinstance(mel, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(mel)).float()
    else:
        t = mel.float()
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    t = t.to(device)
    out = model(t)
    err = (out - t) ** 2
    mse = err.mean().item()
    # per mel bin: mean over time
    per_f = err.squeeze(0).squeeze(0).mean(dim=1).cpu().numpy()
    residual = (t - out).squeeze(0).squeeze(0).cpu().numpy()
    return mse, per_f, residual


def score_wav_file(
    model: MelConvAutoencoder,
    wav_path: str | Path,
    *,
    device: torch.device,
    time_frames: int | None = None,
    max_seconds: float | None = 8.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    if time_frames is None:
        time_frames = getattr(model, "time_frames", DEFAULT_MEL_TIME_FRAMES)
    mel = wav_to_fixed_mel(wav_path, time_frames=int(time_frames), max_seconds=max_seconds)
    x = torch.from_numpy(mel).unsqueeze(0)  # (1, H, W)
    return score_mel_reconstruction(model, x, device=device)


def top_mel_band_explanation(per_f_mse: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Return k (mel_bin_index, mse) highest bands."""
    idx = np.argsort(-per_f_mse)[:k]
    return [(int(i), float(per_f_mse[i])) for i in idx]
