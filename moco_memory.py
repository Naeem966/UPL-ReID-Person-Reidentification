import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from typing import Dict, Optional

def setup_logger():
    logger = logging.getLogger("UPL_ReID_MoCo")
    logger.handlers = []  # Clear existing handlers
    os.makedirs('./logs', exist_ok=True)
    file_handler = logging.FileHandler('./logs/moco_memory.log')
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

class MoCoMemory(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.dim = config.get('feature_dim', 384)
        self.K = config.get('moco_k', 4096)
        self.m = config.get('moco_m', 0.999)
        self.T = config.get('moco_t', 0.07)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        logger.info(f"MoCoMemory initialized: dim={self.dim}, K={self.K}, m={self.m}, T={self.T}, device={self.device}")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)
        if self.K % batch_size != 0:
            logger.warning(f"K ({self.K}) not divisible by batch_size ({batch_size}), adjusting batch_size")
            keys = keys[:self.K - ptr] if ptr + batch_size > self.K else keys
            batch_size = keys.size(0)
        
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
        logger.debug(f"Updated queue: ptr={ptr}, queue shape={self.queue.shape}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, k_all: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
        try:
            logger.debug(f"Batch {batch_idx+1}: Computing MoCo loss, q shape={q.shape}, k shape={k.shape}")
            
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)
            
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.T
            
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
            
            loss = F.cross_entropy(logits, labels)
            
            self._dequeue_and_enqueue(k)
            
            logger.debug(f"Batch {batch_idx+1}: MoCo loss={loss.item():.4f}, logits shape={logits.shape}")
            return loss
        except Exception as e:
            logger.error(f"Error in MoCoMemory.forward at batch {batch_idx+1}: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    config = {
        'batch_size': 16,
        'feature_dim': 384,
        'moco_k': 4096,
        'moco_m': 0.999,
        'moco_t': 0.07,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    moco_memory = MoCoMemory(config).to(device)
    q = torch.randn(16, 384).to(device)
    k = torch.randn(16, 384).to(device)
    k_all = torch.randn(16, 384).to(device)
    loss = moco_memory(q, k, k_all, batch_idx=1)
    logger.info(f"Test MoCo loss: {loss.item():.4f}")