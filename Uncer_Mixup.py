import torch
import torch.nn.functional as F
import logging
import os
from typing import Tuple

def setup_logger():
    logger = logging.getLogger("UPL_ReID_UncerMixup")
    if not logger.handlers:
        os.makedirs('./logs', exist_ok=True)
        file_handler = logging.FileHandler('./logs/uncer_mixup.log')
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
    return logger

logger = setup_logger()

def uncertainty_adaptive_mixup(
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    config: dict,
    device: torch.device,
    query_encoder: torch.nn.Module,
    soft_probs: torch.Tensor,
    batch_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug(f"Batch {batch_idx+1}: Starting uncertainty_adaptive_mixup...")
    try:
        batch_size = source_images.size(0)
        num_classes = config.get('max_num_classes', 751)

        with torch.no_grad():
            source_features = query_encoder(source_images)

        entropy = -torch.sum(soft_probs * torch.log(soft_probs + 1e-10), dim=1)
        entropy_min, entropy_max = entropy.min(), entropy.max()
        if entropy_max - entropy_min < 1e-6 or torch.isnan(entropy).any() or torch.isinf(entropy).any():
            logger.debug(f"Batch {batch_idx+1}: Invalid entropy detected (NaN, inf, or zero range), returning zeros")
            lambda_adaptive = torch.zeros(batch_size, device=device)
        else:
            lambda_adaptive = (entropy_max - entropy) / (entropy_max - entropy_min + 1e-6)
            lambda_adaptive = torch.clamp(lambda_adaptive, 0.0, 1.0)

        logger.debug(f"Batch {batch_idx+1}: Lambda_adaptive: min={lambda_adaptive.min():.4f}, max={lambda_adaptive.max():.4f}")

        mixed_features = lambda_adaptive.view(-1, 1) * source_features + (1 - lambda_adaptive.view(-1, 1)) * target_features

        # Convert labels to one-hot
        source_labels = torch.clamp(source_labels, 0, num_classes - 1)
        pseudo_labels = torch.clamp(pseudo_labels, 0, num_classes - 1)
        source_labels_onehot = F.one_hot(source_labels, num_classes=num_classes).float()
        pseudo_labels_onehot = F.one_hot(pseudo_labels, num_classes=num_classes).float()

        # Mix labels
        mixed_labels = lambda_adaptive.view(-1, 1) * source_labels_onehot + (1 - lambda_adaptive.view(-1, 1)) * pseudo_labels_onehot

        logger.debug(f"Batch {batch_idx+1}: Source labels: min={source_labels.min().item()}, max={source_labels.max().item()}")
        logger.debug(f"Batch {batch_idx+1}: Pseudo probs shape={soft_probs.shape}, Pseudo labels: min={pseudo_labels.min().item()}, max={pseudo_labels.max().item()}")
        logger.debug(f"Batch {batch_idx+1}: Mixed features shape={mixed_features.shape}, Mixed labels shape={mixed_labels.shape}")

        return mixed_features, mixed_labels, lambda_adaptive
    except Exception as e:
        logger.error(f"Error in uncertainty_adaptive_mixup at batch {batch_idx+1}: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    config = {
        'batch_size': 64,
        'max_num_classes': 751,
        'image_size': (224, 224),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    from feature_ext import ViTFeatureExtractor

    source_images = torch.randn(32, 3, 224, 224).to(device)
    source_labels = torch.randint(0, 751, (32,)).to(device)
    target_features = torch.randn(32, 384).to(device)
    pseudo_labels = torch.randint(0, 751, (32,)).to(device)
    soft_probs = torch.softmax(torch.randn(32, 751), dim=1).to(device)
    query_encoder = ViTFeatureExtractor('deit_small_patch16_224').to(device)

    mixed_features, mixed_labels, lambda_adaptive = uncertainty_adaptive_mixup(
        source_images, source_labels, target_features, pseudo_labels,
        config, device, query_encoder, soft_probs
    )
    logger.info(f"Test Mixup: mixed_features shape={mixed_features.shape}, mixed_labels shape={mixed_labels.shape}, "
                f"lambda_adaptive min={lambda_adaptive.min():.4f}, max={lambda_adaptive.max():.4f}")