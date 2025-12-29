import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import re

class ReIDDataset(Dataset):
    """Custom Dataset for ReID datasets with flat directory structure, including camera IDs."""
    def __init__(self, root: str, transform=None, max_classes: int = None):
        self.root = root
        self.transform = transform
        self.max_classes = max_classes
        self.images = []
        self.labels = []
        self.camera_ids = []
        self.label_map = {}
        self.next_label_id = 0
        
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset directory {root} does not exist or is not a directory")
        
        print(f"Scanning directory: {root}")
        dir_contents = sorted(os.listdir(root))
        print(f"Found {len(dir_contents)} items in {root}")
        
        for img_name in dir_contents:
            if not img_name.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(root, img_name)
            
            try:
                # Parse person ID (prefix before first '_')
                label_str = img_name.split('_')[0]
                label = int(label_str)
                if label < 0:
                    continue  # Skip junk images
                
                # Parse camera ID (look for 'cX' pattern)
                parts = img_name.split('_')
                if len(parts) < 2:
                    raise ValueError("Filename lacks expected structure")
                
                camera_part = parts[1]  # e.g., 'c1s1_000151' or 'c1'
                camera_match = re.match(r'c(\d+)', camera_part)
                if not camera_match:
                    raise ValueError("No valid camera ID found")
                camera_id = int(camera_match.group(1))
                
                # Map original person IDs to 0..N-1
                if label not in self.label_map:
                    if self.max_classes is not None and self.next_label_id >= self.max_classes:
                        continue
                    self.label_map[label] = self.next_label_id
                    self.next_label_id += 1
                mapped_label = self.label_map[label]
                
                self.images.append(img_path)
                self.labels.append(mapped_label)
                self.camera_ids.append(camera_id)
            
            except (ValueError, IndexError) as e:
                print(f"Skipping invalid filename: {img_name} ({str(e)})")
                continue
        
        if not self.images:
            raise ValueError(f"No valid images found in {root}")
        
        print(f"Loaded {len(self.images)} images, "
              f"Classes: {len(set(self.labels))} (IDs {min(self.labels)} to {max(self.labels)}), "
              f"Cameras: {len(set(self.camera_ids))} (IDs {min(self.camera_ids)} to {max(self.camera_ids)})")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        camera_id = self.camera_ids[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, camera_id


def get_transforms(image_size: tuple):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform


def prepare_data(market_path: str, duke_path: str, config: dict):
    market_train_path = os.path.join(market_path, "bounding_box_train")
    duke_train_path = os.path.join(duke_path, "bounding_box_train")
    
    if not os.path.exists(market_train_path):
        raise FileNotFoundError(f"Market-1501 path not found: {market_train_path}")
    if not os.path.exists(duke_train_path):
        raise FileNotFoundError(f"DukeMTMC-ReID path not found: {duke_train_path}")
    
    train_transform, _ = get_transforms(config['image_size'])
    
    print("\nLoading Market-1501...")
    market_dataset = ReIDDataset(market_train_path, transform=train_transform)
    
    print("\nLoading DukeMTMC-ReID...")
    duke_dataset = ReIDDataset(duke_train_path, transform=train_transform)
    
    market_indices = np.random.choice(len(market_dataset), len(market_dataset)//2, replace=False)
    duke_indices = np.random.choice(len(duke_dataset), len(duke_dataset)//2, replace=False)
    
    meta_dataset = torch.utils.data.ConcatDataset([
        torch.utils.data.Subset(market_dataset, market_indices),
        torch.utils.data.Subset(duke_dataset, duke_indices)
    ])
    print(f"Created meta-set with {len(meta_dataset)} samples")
    
    market_loader = DataLoader(market_dataset, batch_size=config['batch_size']//2,
                               shuffle=True, num_workers=4, pin_memory=True)
    duke_loader = DataLoader(duke_dataset, batch_size=config['batch_size']//2,
                             shuffle=True, num_workers=4, pin_memory=True)
    meta_loader = DataLoader(meta_dataset, batch_size=config['batch_size']//8,
                             shuffle=True, num_workers=4, pin_memory=True)
    
    return market_loader, duke_loader, meta_loader

if __name__ == "__main__":
    config = {"image_size": (256, 128), "batch_size": 64}
    market_loader, duke_loader, meta_loader = prepare_data(
        "/home/datasets/market1501/Market-1501-v15.09.15",
        "/home/datasets/dukemtmcreid/DukeMTMC-reID",
        config
    )
    print(f"Market Loader: {len(market_loader)} batches")
    print(f"Duke Loader: {len(duke_loader)} batches")
    print(f"Meta Loader: {len(meta_loader)} batches")