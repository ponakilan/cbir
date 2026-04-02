import time

import torch
from torch import nn
from torchvision import io, transforms
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights

from cbir.utils import multi_scale_image


class QueryFeatureExtractor(nn.Module):

    def __init__(self, num_features: int = 500):
        super().__init__()

        self.num_features = num_features

        resnet_backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(resnet_backbone.children())[:-2])

        for p in self.resnet_backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_images = multi_scale_image(x)

        all_local_features = []
        for scaled_image in scaled_images:
            spatial_tensor = self.resnet_backbone(scaled_image)
            B, D, H, W = spatial_tensor.shape
            local_features = spatial_tensor.view(B, D, -1).transpose(1, 2)
            all_local_features.append(local_features)

        combined_features = torch.cat(all_local_features, dim=1)
        l2_norms = torch.norm(combined_features, p=2, dim=-1)
        N = min(self.num_features, combined_features.shape[1]) 
        _, top_indices = torch.topk(l2_norms, k=N, dim=1)
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, combined_features.shape[-1])
        selected_features = torch.gather(combined_features, 1, top_indices_expanded)

        return selected_features


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1, keepdim=False).pow(1./self.p)


class DatabaseFeatureExtractor(nn.Module):
    def __init__(self, num_features: int = 500, num_clusters: int = 10):
        super().__init__()
        self.num_features = num_features
        self.num_clusters = num_clusters

        resnet_backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(resnet_backbone.children())[:-2])
        self.gem = GeM(p=3)

    def perform_clustering_and_gem(self, features: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
        """Fully vectorized K-Means and GeM pooling (No Python for-loops over batch/clusters)"""
        B, N, D = features.shape
        K = self.num_clusters
        
        # Smart initialization (evenly spaced instead of just the first K)
        indices = torch.linspace(0, N - 1, steps=K).long()
        centroids = features[:, indices, :].clone()

        for _ in range(num_iters):
            distances = torch.cdist(features, centroids)
            assignments = distances.argmin(dim=2)
            mask = F.one_hot(assignments, num_classes=K).float()
            
            sum_coords = torch.bmm(mask.transpose(1, 2), features)
            counts = mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
            centroids = sum_coords / counts

        p_val = self.gem.p.view(1, 1, 1)
        feat_p = features.clamp(min=self.gem.eps).pow(p_val)
        sum_p = torch.bmm(mask.transpose(1, 2), feat_p)
        
        gem_clusters = (sum_p / counts).pow(1.0 / p_val)
        is_empty = (mask.sum(dim=1) == 0).unsqueeze(-1)
        
        return torch.where(is_empty, centroids, gem_clusters)

    def forward(self, x: torch.Tensor) -> tuple:
        scaled_images = multi_scale_image(x)

        all_local_features = []
        for scaled_image in scaled_images:
            spatial_tensor = self.resnet_backbone(scaled_image)
            B, D, H, W = spatial_tensor.shape
            local_features = spatial_tensor.view(B, D, -1).transpose(1, 2)
            all_local_features.append(local_features)

        combined_features = torch.cat(all_local_features, dim=1)
        l2_norms = torch.norm(combined_features, p=2, dim=-1)
        N = min(self.num_features, combined_features.shape[1]) 
        
        _, top_indices = torch.topk(l2_norms, k=N, dim=1)
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, combined_features.shape[-1])
        selected_features = torch.gather(combined_features, 1, top_indices_expanded)
        
        cluster_centers = self.perform_clustering_and_gem(selected_features)

        # --- NEW MACRO-PRUNING MATH ---
        clusters_norm = F.normalize(cluster_centers, p=2, dim=-1)
        centroids = clusters_norm.mean(dim=1) # [B, D]
        distances = torch.norm(clusters_norm - centroids.unsqueeze(1), p=2, dim=-1)
        radii = distances.max(dim=1)[0] # [B]

        return cluster_centers, centroids, radii

if __name__ == "__main__":
    start = time.time()

    query_extractor = QueryFeatureExtractor(num_features=500)
    db_extractor = DatabaseFeatureExtractor(num_features=500, num_clusters=10)
    gem = GeM(p=3)

    query_extractor.eval()
    db_extractor.eval()

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sample_image_path = "/home/ubuntu/data/datasets/roxford5k/jpg/oxford_002881.jpg"
    sample_image = transform(io.read_image(sample_image_path)).unsqueeze(0)

    with torch.no_grad():
        top_500_features = query_extractor(sample_image)
        V_q = gem(top_500_features)
        X_c_K = db_extractor(sample_image)

    print(f"Global query vector shape: {V_q.shape}")
    print(f"Clustered Database Vector shape: {X_c_K.shape}")

    end = time.time()
    print(f"Time taken: {end - start}")
