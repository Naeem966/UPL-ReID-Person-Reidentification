import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np
import logging
import os
from typing import Tuple

def setup_logger():
    logger = logging.getLogger("UPL_ReID_GMM")
    logger.handlers = []
    os.makedirs('./logs', exist_ok=True)
    file_handler = logging.FileHandler('./logs/gmm.log')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger

logger = setup_logger()

def gmm_pseudo_labeling(
    all_target_features: torch.Tensor,
    batch_features: torch.Tensor,
    all_camera_ids: torch.Tensor,
    batch_camera_ids: torch.Tensor,
    num_classes: int,
    device: torch.device,
    batch_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.info(f"Batch {batch_idx+1}: Running GMM pseudo-labeling...")
    try:
        all_target_features = all_target_features.cpu().numpy()
        batch_features = batch_features.cpu().numpy()

        logger.info(f"Batch {batch_idx+1}: Reducing features to 50 dimensions")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50, random_state=42)
        all_target_features = pca.fit_transform(all_target_features)
        batch_features = pca.transform(batch_features)

        logger.info(f"Batch {batch_idx+1}: Completed feature reduction, shapes: all={all_target_features.shape}, batch={batch_features.shape}")

        n_components = min(num_classes, 500)  # Limit components
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            max_iter=100,
            random_state=42,
            reg_covar=1e-2  # Increased from 1e-3
        )
        logger.info(f"Batch {batch_idx+1}: Fitting GMM with {n_components} components")
        gmm.fit(all_target_features)

        logger.info(f"Batch {batch_idx+1}: Completed GMM clustering")
        soft_probs = gmm.predict_proba(batch_features)
        soft_probs = torch.from_numpy(soft_probs).float().to(device)
        pseudo_labels = soft_probs.argmax(dim=1)

        soft_probs = soft_probs + 1e-6
        soft_probs = soft_probs / soft_probs.sum(dim=1, keepdim=True)

        entropy = -torch.sum(soft_probs * torch.log(soft_probs + 1e-10), dim=1)
        entropy_min, entropy_max = entropy.min(), entropy.max()
        logger.debug(f"Batch {batch_idx+1}: GMM output: soft_probs={soft_probs.shape}, pseudo_labels={pseudo_labels.shape}, "
                     f"pseudo_labels min={pseudo_labels.min().item()}, max={pseudo_labels.max().item()}")

        if entropy_max - entropy_min < 1e-6 or torch.isnan(entropy).any() or torch.isinf(entropy).any():
            logger.warning(f"Batch {batch_idx+1}: Invalid entropy detected (min={entropy_min:.4f}, max={entropy_max:.4f}), returning zeros")
            entropy = torch.zeros_like(entropy)

        logger.info(f"Batch {batch_idx+1}: GMM pseudo-labeling complete")
        return soft_probs, pseudo_labels, entropy
    except Exception as e:
        logger.error(f"Error in gmm_pseudo_labeling at batch {batch_idx+1}: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    config = {
        'batch_size': 16,
        'image_size': (224, 224),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    all_target_features = torch.randn(2016, 384)
    batch_features = torch.randn(16, 384)
    all_camera_ids = torch.randint(0, 8, (2016,))
    batch_camera_ids = torch.randint(0, 8, (16,))

    soft_probs, pseudo_labels, entropy = gmm_pseudo_labeling(
        all_target_features, batch_features, all_camera_ids, batch_camera_ids,
        num_classes=702, device=device
    )
    logger.info(f"Test GMM: soft_probs shape={soft_probs.shape}, pseudo_labels shape={pseudo_labels.shape}, "
                f"entropy min={entropy.min():.4f}, max={entropy.max():.4f}")