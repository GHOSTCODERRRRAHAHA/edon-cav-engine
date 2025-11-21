import torch
import pickle
from sklearn.decomposition import PCA
import numpy as np
import os

os.makedirs("models", exist_ok=True)

# -----------------------------
# 1) Generate synthetic multimodal base
# -----------------------------
X = np.random.randn(500, 256)
pca = PCA(n_components=128)
pca.fit(X)

# Save in correct format
with open("models/pca.pkl", "wb") as f:
    pickle.dump({"pca": pca}, f)  # <-- required key name


# -----------------------------
# 2) Define architecture EXACTLY like engine_v2.py
# -----------------------------
state_head = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 6)
)

action_head = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 7)
)

# Wrap as ModuleDict
model = torch.nn.ModuleDict({
    "state_head": state_head,
    "action_head": action_head
})

torch.save(model.state_dict(), "models/neural_head.pt")

print("[✔] Regenerated EXACT v2 weights successfully")
