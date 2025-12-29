import torch
from feature_ext import ViTFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/home/tq_naeem/Project/project_1/model/pytorch_model.bin"
model = ViTFeatureExtractor('deit_small_patch16_224', checkpoint_path=checkpoint_path).to(device)
dummy_input = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    features = model(dummy_input)
print(f"Feature dimension: {features.size(1)}")
print(f"Features shape: {features.shape}")