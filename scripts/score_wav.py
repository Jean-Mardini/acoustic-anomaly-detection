"""Score a WAV with a trained MelConvAutoencoder checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ml.inference import load_mel_conv_ae, score_wav_file, top_mel_band_explanation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_mel_conv_ae(args.checkpoint, device=device)
    mse, per_f, _ = score_wav_file(
        model,
        args.wav,
        device=device,
        max_seconds=args.max_seconds,
    )
    top = top_mel_band_explanation(per_f, k=5)
    out = {
        "mse": mse,
        "top_mel_bands_by_error": [{"mel_bin": i, "mse": v} for i, v in top],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
