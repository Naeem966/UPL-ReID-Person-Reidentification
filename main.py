import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
import os
import csv
from tqdm import tqdm
from typing import Dict, Tuple
from feature_ext import ViTFeatureExtractor
from loss import LossComputation
from data import prepare_data
from moco_memory import MoCoMemory
from GMM import gmm_pseudo_labeling
from Uncer_Mixup import uncertainty_adaptive_mixup

def setup_logger():
    logger = logging.getLogger("UPL_ReID")
    logger.handlers = []
    os.makedirs('./logs', exist_ok=True)
    file_handler = logging.FileHandler('./logs/training.log')
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

def save_metrics(metrics: Dict, output_dir: str, filename: str = "metrics.csv"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    logger.debug(f"Metrics saved to {filepath}")

def train_upl_reid(
    source_loader: DataLoader,
    target_loader: DataLoader,
    meta_loader: DataLoader,
    query_encoder: nn.Module,
    key_encoder: nn.Module,
    loss_computation: LossComputation,
    moco_memory: MoCoMemory,
    config: Dict,
    device: torch.device
) -> None:
    logger.info("Starting UPL-ReID training...")
    scaler = GradScaler()
    optimizer = optim.Adam(query_encoder.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    cache_valid_batches = 20
    num_epochs = config.get('num_epochs', 50)
    output_dir = config.get('output_dir', './checkpoints')
    os.makedirs(output_dir, exist_ok=True)

    all_target_features = None
    all_camera_ids = None
    cached_features_path = os.path.join(output_dir, 'target_features.pt')
    if os.path.exists(cached_features_path):
        logger.info(f"Loading cached target features from {cached_features_path}")
        cached_data = torch.load(cached_features_path)
        all_target_features = cached_data['features']
        all_camera_ids = cached_data['camera_ids']
    else:
        logger.info("Extracting initial target features...")
        all_target_features, all_camera_ids = [], []
        key_encoder.eval()
        with torch.no_grad():
            for images, _, camera_ids in target_loader:
                images = images.to(device)
                features = key_encoder(images)
                all_target_features.append(features.cpu())
                all_camera_ids.append(camera_ids.cpu())
        all_target_features = torch.cat(all_target_features, dim=0)
        all_camera_ids = torch.cat(all_camera_ids, dim=0)
        torch.save({'features': all_target_features, 'camera_ids': all_camera_ids}, cached_features_path)
        logger.info(f"Saved target features to {cached_features_path}")

    query_encoder.train()
    key_encoder.train()
    skipped_batches = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        progress_bar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)),
                           desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, ((source_images, source_labels, _), (target_images, _, target_camera_ids)) in enumerate(progress_bar):
            if source_images.size(0) != target_images.size(0):
                logger.debug(f"Skipping batch {batch_idx+1}: source_images={source_images.shape}, target_images={target_images.shape}")
                skipped_batches += 1
                continue

            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images, target_camera_ids = target_images.to(device), target_camera_ids.to(device)

            with autocast():
                query_features = query_encoder(source_images)
                key_features = key_encoder(target_images)
                target_features = key_features.detach()

                if batch_idx % cache_valid_batches == 0:
                    soft_probs, pseudo_labels, entropy = gmm_pseudo_labeling(
                        all_target_features, target_features.cpu(), all_camera_ids, target_camera_ids.cpu(),
                        config['target_num_classes'], device, batch_idx
                    )
                else:
                    soft_probs, pseudo_labels, entropy = gmm_pseudo_labeling(
                        all_target_features, target_features.cpu(), all_camera_ids, target_camera_ids.cpu(),
                        config['target_num_classes'], device, batch_idx
                    )

                mixed_features, mixed_labels, lambda_adaptive = uncertainty_adaptive_mixup(
                    source_images, source_labels, target_features, pseudo_labels,
                    config, device, query_encoder, soft_probs, batch_idx
                )

                moco_loss = moco_memory(query_features, key_features, target_features, batch_idx)

                total_loss_batch, loss_dict = loss_computation(
                    mixed_features, target_features, query_features, mixed_labels,
                    pseudo_labels, source_labels, soft_probs, moco_loss, batch_idx
                )

                logger.debug(f"Batch {batch_idx+1}: {loss_dict}")

                optimizer.zero_grad()
                scaler.scale(total_loss_batch).backward()
                scaler.step(optimizer)
                scaler.update()

                grad_norm = torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), max_norm=5.0)
                logger.debug(f"Batch {batch_idx+1}: grad_norm={grad_norm:.4f}")

                with torch.no_grad():
                    for param_q, param_k in zip(query_encoder.parameters(), key_encoder.parameters()):
                        param_k.data = param_k.data * config['momentum'] + param_q.data * (1. - config['momentum'])
                logger.debug(f"Updated key encoder with momentum")

                total_loss += total_loss_batch.item()
                batch_count += 1
                progress_bar.set_postfix(loss=total_loss_batch.item())

                all_target_features[batch_idx * config['batch_size']:(batch_idx + 1) * config['batch_size']] = target_features.cpu()

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Processed {batch_count}/{len(source_loader)} batches, Skipped {skipped_batches}")
            metrics = {'epoch': epoch + 1, 'loss': avg_loss, 'mAP': 0.0, 'Rank-1': 0.0}
            save_metrics(metrics, output_dir)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} | mAP: {metrics['mAP']:.4f} | Rank-1: {metrics['Rank-1']:.4f}")

        torch.save({
            'query_encoder': query_encoder.state_dict(),
            'key_encoder': key_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    config = {
        'batch_size': 16,  # Changed to match data loader output
        'source_num_classes': 751,  # Market-1501
        'target_num_classes': 702,  # DukeMTMC-ReID
        'image_size': (224, 224),
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'momentum': 0.999,
        'num_epochs': 50,
        'output_dir': './checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'triplet_margin': 0.3,
        'feature_dim': 384,
        'moco_k': 4096,
        'moco_m': 0.999,
        'moco_t': 0.07
    }
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    market_path = "/home/tq_naeem/Project/.venv/main/datasets/Market-1501-v15.09.15"
    duke_path = "/home/datasets/dukemtmcreid/DukeMTMC-reID"

    logger.info("Loading datasets...")
    source_loader, target_loader, meta_loader = prepare_data(market_path, duke_path, config)

    logger.info("Initializing ViTFeatureExtractor...")
    query_encoder = ViTFeatureExtractor('deit_small_patch16_224').to(device)
    key_encoder = ViTFeatureExtractor('deit_small_patch16_224').to(device)

    logger.info("Initializing MoCoMemory and LossComputation...")
    moco_memory = MoCoMemory(config).to(device)
    loss_computation = LossComputation(config, config['source_num_classes']).to(device)

    logger.info("Starting training...")
    train_upl_reid(source_loader, target_loader, meta_loader, query_encoder, key_encoder, loss_computation, moco_memory, config, device)