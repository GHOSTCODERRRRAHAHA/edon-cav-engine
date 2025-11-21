import torch
import pickle
from sklearn.decomposition import PCA
import numpy as np
import os

os.makedirs("models", exist_ok=True)

# ---- Create synthetic multimodal base data ----
X = np.random.randn(500, 256)  # simulate multimodal features
pca = PCA(n_components=128)
pca.fit(X)

with open("models/pca.pkl", "wb") as f:
    pickle.dump({"components": pca.components_, "mean": pca.mean_}, f)

# ---- Create neural head matching v2 architecture ----
state_head = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 6),   # 6 states
)

action_head = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 7),   # 7 influence outputs
)

model = torch.nn.ModuleDict({
    "state_head": state_head,
    "action_head": action_head
})

torch.save(model.state_dict(), "models/neural_head.pt")

print("[✔] Generated v2-compatible PCA + weights")
