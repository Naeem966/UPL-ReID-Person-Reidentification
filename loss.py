import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from torch.cuda.amp import autocast
import logging
import os

def setup_logger():
    logger = logging.getLogger("UPL_ReID_Loss")
    if not logger.handlers:
        os.makedirs('./logs', exist_ok=True)
        file_handler = logging.FileHandler('./logs/loss.log')
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

class LossComputation(nn.Module):
    def __init__(self, config: Dict, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = max(751, num_classes)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.triplet_loss = nn.TripletMarginLoss(margin=config['triplet_margin'], p=2)
        self.classifier = nn.Linear(384, self.num_classes).to(config.get('device', 'cpu'))

    def sample_triplets(self, features: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample triplets ensuring batch_size positive and negative pairs."""
        anchor = features
        positive = torch.zeros_like(anchor, device=device)
        negative = torch.zeros_like(anchor, device=device)

        for i in range(batch_size):
            pos_mask = labels == labels[i]
            neg_mask = labels != labels[i]
            pos_mask[i] = False  # Exclude the anchor itself

            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]

            if len(pos_indices) > 0:
                pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,), device=device)]
                positive[i] = features[pos_idx]
            else:
                # If no positive, use a random sample (self-supervised fallback)
                positive[i] = features[torch.randint(0, batch_size, (1,), device=device)]

            if len(neg_indices) > 0:
                neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,), device=device)]
                negative[i] = features[neg_idx]
            else:
                # If no negative, use a random sample
                negative[i] = features[torch.randint(0, batch_size, (1,), device=device)]

        return anchor, positive, negative

    def forward(
        self,
        F_mixed: torch.Tensor,
        F_target: torch.Tensor,
        F_source: torch.Tensor,
        mixed_labels: torch.Tensor,
        pseudo_labels: torch.Tensor,
        source_labels: torch.Tensor,
        soft_probs: torch.Tensor,
        moco_loss: torch.Tensor,
        batch_idx: int = 0
    ) -> Tuple[torch.Tensor, Dict]:
        logger.debug(f"Batch {batch_idx+1}: Computing losses...")
        try:
            source_labels = torch.clamp(source_labels, 0, self.num_classes - 1)
            pseudo_labels = torch.clamp(pseudo_labels, 0, self.num_classes - 1)

            F_mixed = F.normalize(F_mixed, dim=1)
            F_target = F.normalize(F_target, dim=1)
            F_source = F.normalize(F_source, dim=1)

            batch_size = F_mixed.size(0)

            # Sample triplets for mixed features
            anchor_mixed, positive_mixed, negative_mixed = self.sample_triplets(F_mixed, mixed_labels.argmax(dim=1), batch_size, F_mixed.device)
            loss_triplet_mixed = self.triplet_loss(anchor_mixed, positive_mixed, negative_mixed)
            logger.debug(f"Batch {batch_idx+1}: Triplet mixed loss computed, shape: anchor={anchor_mixed.shape}, positive={positive_mixed.shape}, negative={negative_mixed.shape}")

            # Sample triplets for target features
            anchor_target, positive_target, negative_target = self.sample_triplets(F_target, pseudo_labels, batch_size, F_target.device)
            loss_triplet_target = self.triplet_loss(anchor_target, positive_target, negative_target)
            logger.debug(f"Batch {batch_idx+1}: Triplet target loss computed, shape: anchor={anchor_target.shape}, positive={positive_target.shape}, negative={negative_target.shape}")

            logits = self.classifier(F_source)
            loss_ce = self.ce_loss(logits, source_labels)

            total_loss = moco_loss + 0.5 * (loss_triplet_mixed + loss_triplet_target) + loss_ce

            loss_dict = {
                'total': total_loss,
                'moco': moco_loss,
                'triplet_mixed': loss_triplet_mixed,
                'triplet_target': loss_triplet_target,
                'cross_entropy': loss_ce
            }

            logger.debug(f"Batch {batch_idx+1}: Loss components: total={total_loss:.4f}, moco={moco_loss:.4f}, "
                         f"triplet_mixed={loss_triplet_mixed:.4f}, triplet_target={loss_triplet_target:.4f}, "
                         f"cross_entropy={loss_ce:.4f}")
            logger.debug(f"Batch {batch_idx+1}: Loss computation complete")
            if total_loss == 0:
                logger.warning(f"Batch {batch_idx+1}: Total loss is zero, possible issue in loss computation")

            return total_loss, loss_dict
        except Exception as e:
            logger.error(f"Error in LossComputation.forward at batch {batch_idx+1}: {str(e)}")
            raise

    def compute_meta_loss(
        self,
        meta_loader: DataLoader,
        query_encoder: nn.Module,
        loss_computation: 'LossComputation',
        device: torch.device
    ) -> torch.Tensor:
        logger.info("Computing meta-loss...")
        try:
            if not isinstance(query_encoder, nn.Module):
                raise ValueError(f"query_encoder must be an nn.Module, got {type(query_encoder)}")
            query_encoder.eval()
            meta_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for images, labels, _ in meta_loader:
                    images, labels = images.to(device), labels.to(device)
                    with autocast():
                        features = query_encoder(images)
                    logits = self.classifier(features)
                    labels = torch.clamp(labels, 0, self.num_classes - 1)
                    meta_loss += self.ce_loss(logits, labels)
                    num_batches += 1

            meta_loss = meta_loss / max(1, num_batches)
            logger.info(f"Meta-loss: {meta_loss:.4f}")
            return meta_loss
        except Exception as e:
            logger.error(f"Error in compute_meta_loss: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    config = {
        'triplet_margin': 0.3,
        'batch_size': 64,
        'source_num_classes': 751,
        'image_size': (224, 224),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    from data import prepare_data
    from feature_ext import ViTFeatureExtractor

    market_path = "/home/tq_naeem/Project/.venv/main/datasets/Market-1501-v15.09.15"
    duke_path = "/home/datasets/dukemtmcreid/DukeMTMC-reID"

    logger.info("Loading datasets...")
    market_loader, duke_loader, meta_loader = prepare_data(market_path, duke_path, config)

    logger.info("Initializing ViTFeatureExtractor...")
    query_encoder = ViTFeatureExtractor('deit_small_patch16_224').to(device)

    logger.info("Initializing LossComputation...")
    loss_computation = LossComputation(config, config['source_num_classes']).to(device)

    logger.info("Computing meta-loss...")
    meta_loss = loss_computation.compute_meta_loss(meta_loader, query_encoder, loss_computation, device)