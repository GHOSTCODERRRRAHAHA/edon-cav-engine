cd C:\Users\cjbig\Desktop\EDON\edon-cav-engine

$env:EDON_MODE = "v2"
$env:EDON_DEVICE_PROFILE = "humanoid_full"
$env:EDON_PCA_PATH = ".\models\pca_128.pkl"
$env:EDON_NEURAL_WEIGHTS = ".\models\neural_head.pt"

python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
