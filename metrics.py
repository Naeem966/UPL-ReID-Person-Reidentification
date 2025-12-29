import torch
import numpy as np
from sklearn.metrics import average_precision_score

def compute_reid_metrics(query_features, query_labels, gallery_features, gallery_labels, device):
    query_features = torch.nn.functional.normalize(query_features, dim=1)
    gallery_features = torch.nn.functional.normalize(gallery_features, dim=1)

    # Cosine similarity -> distance
    dist_matrix = 1 - torch.mm(query_features, gallery_features.t())

    num_query = query_features.size(0)
    aps, rank1 = [], []

    for i in range(num_query):
        q_label = query_labels[i].item()
        dist = dist_matrix[i].cpu().numpy()
        sorted_idx = np.argsort(dist)
        matches = (gallery_labels[sorted_idx].cpu().numpy() == q_label).astype(int)

        if np.any(matches):
            aps.append(average_precision_score(matches, -dist[sorted_idx]))
            rank1.append(matches[0])
        else:
            aps.append(0)
            rank1.append(0)

    return {
        "mAP": float(np.mean(aps)),
        "Rank-1": float(np.mean(rank1))
    }
