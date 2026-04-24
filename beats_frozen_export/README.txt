# BEATs Frozen Anomaly Detection — Export

## Files
- gmms/{machine}.pkl  : fitted GMM scorer per machine type
- embeddings_info.json: metadata (embedding dim, sample rate, etc.)
- BEATs_iter3_plus_AS2M.pt: pretrained BEATs checkpoint (copy here manually, 345MB)
- BEATs.py, backbone.py, modules.py: BEATs model code (copy from models/)

## Usage in your app
```python
import pickle, numpy as np, torch
from BEATs import BEATs, BEATsConfig

# Load BEATs
raw = torch.load("BEATs_iter3_plus_AS2M.pt", map_location="cpu")
beats = BEATs(BEATsConfig(raw["cfg"]))
beats.load_state_dict(raw["model"])
beats.eval()

# Load GMM for a machine type
with open("gmms/fan.pkl", "rb") as f:
    gmm = pickle.load(f)

# Score a 10s audio clip (numpy float32, 16kHz)
wav_t = torch.from_numpy(wav).unsqueeze(0)
pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool)
with torch.no_grad():
    feats, _ = beats.extract_features(wav_t, padding_mask=pad)
emb = feats.mean(dim=1).numpy()
anomaly_score = float(-gmm.score_samples(emb)[0])  # higher = more anomalous
```
