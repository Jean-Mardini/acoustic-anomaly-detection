import torch

from backend.ml.features import DEFAULT_MEL_TIME_FRAMES, N_MELS
from backend.ml.model import MelConvAutoencoder


def test_mel_conv_autoencoder_forward() -> None:
    m = MelConvAutoencoder(
        n_mels=N_MELS,
        time_frames=DEFAULT_MEL_TIME_FRAMES,
        latent_dim=128,
    )
    x = torch.randn(2, 1, N_MELS, DEFAULT_MEL_TIME_FRAMES)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
