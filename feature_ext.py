import os
import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional

class ViTFeatureExtractor(nn.Module):
    """Lightweight Vision Transformer (ViT) for feature extraction with local checkpoint loading."""
    def __init__(self, model_name: str = 'deit_small_patch16_224', checkpoint_path: Optional[str] = None):
        super(ViTFeatureExtractor, self).__init__()
        # Initialize model without pretrained weights
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.feature_dim = self.model.embed_dim  
        
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # Handle possible dict nesting
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded local checkpoint from {checkpoint_path}")
        
        # Add dropout for stability
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        features = self.model(x)  # Shape: [batch_size, feature_dim]
        features = self.dropout(features)
        return features

def extract_features(
    mixed_images: torch.Tensor,
    target_images: torch.Tensor,
    model: nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from mixed and target images using ViT."""
    model.eval()
    with torch.no_grad():
        mixed_images = mixed_images.to(device)
        target_images = target_images.to(device)
        
        F_mixed = model(mixed_images)
        F_target = model(target_images)
    
    return F_mixed, F_target

def test_feature_extraction():
    """Test function to demonstrate feature extraction."""
    config = {'batch_size': 64, 'image_size': (224, 224)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = "/home/tq_naeem/Project/project_1/model/pytorch_model.bin"
    model = ViTFeatureExtractor(
        model_name='deit_small_patch16_224',
        checkpoint_path=checkpoint_path
    ).to(device)
    
    mixed_images = torch.randn(config['batch_size']//2, 3, *config['image_size'])
    target_images = torch.randn(config['batch_size']//2, 3, *config['image_size'])
    
    F_mixed, F_target = extract_features(mixed_images, target_images, model, device)
    print(f"Mixed features shape: {F_mixed.shape}, Target features shape: {F_target.shape}")

if __name__ == "__main__":
    market_path = "/home/tq_naeem/Project/.venv/main/datasets/Market-1501-v15.09.15"
    duke_path = "/home/datasets/dukemtmc/DukeMTMC-reID/"
    
    config = {
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.00035,
        'moco_queue_size': 65536,
        'moco_momentum': 0.999,
        'moco_temperature': 0.07,
        'meta_frequency': 5,
        'meta_lr': 1e-05,
        'triplet_margin': 0.3,
        'image_size': (224, 224),
        'data_dir': './datasets'
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize data loaders first
    from data import prepare_data
    try:
        market_loader, duke_loader, meta_loader = prepare_data(market_path, duke_path, config)
    except Exception as e:
        print(f"Error initializing data loaders: {e}")
        raise
    
    from Uncer_Mixup import uncertainty_adaptive_mixup, compute_entropy, compute_ece
    checkpoint_path = "/home/tq_naeem/Project/project_1/model/pytorch_model.bin"
    model = ViTFeatureExtractor('deit_small_patch16_224', checkpoint_path=checkpoint_path).to(device)
    
    # Get one batch from market_loader and duke_loader
    try:
        market_iter = iter(market_loader)
        market_batch = next(market_iter)
        print(f"Market batch: {[x.shape for x in market_batch] if isinstance(market_batch, (list, tuple)) else market_batch.shape}")
        market_images = market_batch[0].to(device)  # First element: images
        market_labels = market_batch[1].to(device)  # Second element: labels
        
        duke_iter = iter(duke_loader)
        duke_batch = next(duke_iter)
        print(f"Duke batch: {[x.shape for x in duke_batch] if isinstance(duke_batch, (list, tuple)) else duke_batch.shape}")
        duke_images = duke_batch[0].to(device)  # First element: images
        duke_labels = duke_batch[1].to(device)  # Second element: labels (pseudo-labels)
        
        # Extract target features for uncertainty_adaptive_mixup
        with torch.no_grad():
            target_features = model(duke_images)
    except Exception as e:
        print(f"Error loading batch from data loaders: {e}")
        raise
    
    # Use uncertainty_adaptive_mixup to get mixed features
    try:
        mixed_features, mixed_labels, lambda_adaptive = uncertainty_adaptive_mixup(
            source_images=market_images,
            source_labels=market_labels,
            target_features=target_features,
            pseudo_labels=duke_labels,
            config=config,
            device=device,
            model=model
        )
    except Exception as e:
        print(f"Error in uncertainty_adaptive_mixup: {e}")
        raise
    
    # Since uncertainty_adaptive_mixup returns features, use them directly
    F_mixed = mixed_features
    F_target = target_features
    print(f"✅ Feature extraction complete: Mixed={F_mixed.shape}, Target={F_target.shape}")