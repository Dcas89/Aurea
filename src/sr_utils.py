import torch
from torch import nn
import torch.nn.functional as F


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, radius, alpha, beta, gamma, w1, w2, shift):
        super().__init__()
        self.radius = radius
        self.diameter = 2 * radius + 1
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.shift = shift

    def forward(self, guidance, features):
        device = features.device
        B, C_g, H, W = guidance.shape
        C_f = features.shape[1]

        guidance_padded = F.pad(guidance, pad=[self.radius] * 4, mode='reflect')
        features_padded = F.pad(features, pad=[self.radius] * 4, mode='reflect')

        guidance_unfolded = F.unfold(guidance_padded, kernel_size=self.diameter, padding=0)
        features_unfolded = F.unfold(features_padded, kernel_size=self.diameter, padding=0)

        guidance_patches = guidance_unfolded.reshape(B, C_g, self.diameter**2, H*W)
        features_patches = features_unfolded.reshape(B, C_f, self.diameter**2, H*W)

        center_idx = self.diameter**2 // 2
        center_guidance = guidance_patches[:, :, center_idx:center_idx+1, :]
        center_features = features_patches[:, :, center_idx:center_idx+1, :]

        guidance_diff = (guidance_patches - center_guidance).square().sum(1)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(-self.radius, self.radius + 1, device=device),
            torch.arange(-self.radius, self.radius + 1, device=device),
            indexing='ij'
        )
        
        norm_y = y_grid.flatten() / self.radius
        norm_x = x_grid.flatten() / self.radius
        
        coord_diff = norm_y.square() + norm_x.square()
        coord_diff = coord_diff.reshape(1, self.diameter**2, 1).expand(B, -1, H*W)

        sim_kernel = (
            self.w1 * torch.exp(- coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta)) +
            self.w2 * torch.exp(- coord_diff / (2 * self.gamma))
        ) - self.shift

        features_patches_norm = F.normalize(features_patches, dim=1, eps=1e-8)
        center_features_norm = F.normalize(center_features, dim=1, eps=1e-8)
        sim = (features_patches_norm * center_features_norm).sum(1)

        feat_diff = (1 - sim).clamp(min=0.0)
        weighted_diff = feat_diff * sim_kernel

        return weighted_diff.mean()


def feature_diversity_loss(features, alpha=2.0):
    x = F.normalize(features, p=2, dim=2)
    sim = torch.bmm(x, x.transpose(1, 2))
    dist = 1.0 - sim
    p = features.shape[1]
    mask = 1.0 - torch.eye(p, device=features.device)
    dist_masked = dist * mask
    fd = torch.sum(dist_masked, dim=(1, 2)) / (p * (p - 1))
    loss_div = (1.0 - fd).mean()
    return alpha * loss_div


def feature_diversity(features):
    p = features.shape[1]
    
    x = F.normalize(features, p=2, dim=2)
    sim = torch.bmm(x, x.transpose(1, 2))
    dist = 1.0 - sim
    mask = 1.0 - torch.eye(p, device=features.device)
    dist_masked = dist * mask
    div_scores = torch.sum(dist_masked, dim=(1, 2)) / (p * (p - 1))
    
    return div_scores.mean().item()


def feature_similarity(features1, features2):
    assert features1.shape[0] == features2.shape[0]
    
    x1 = F.normalize(features1, p=2, dim=2)
    x2 = F.normalize(features2, p=2, dim=2)
    b, p1, _ = x1.shape
    _, p2, _ = x2.shape
    K1 = torch.bmm(x1, x1.transpose(1, 2))
    K2 = torch.bmm(x2, x2.transpose(1, 2))
    n = min(p1, p2)
    K1 = K1[:, :n, :n]
    K2 = K2[:, :n, :n]
    ones = torch.ones(n, n, device=features1.device) / n
    H = torch.eye(n, device=features1.device) - ones
    H = H.unsqueeze(0).expand(b, -1, -1)
    K1c = torch.bmm(torch.bmm(H, K1), H)
    K2c = torch.bmm(torch.bmm(H, K2), H)
    dotp = torch.sum(K1c * K2c, dim=(1, 2))
    norm1 = torch.sqrt(torch.sum(K1c * K1c, dim=(1, 2)))
    norm2 = torch.sqrt(torch.sum(K2c * K2c, dim=(1, 2)))
    denom = torch.clamp(norm1 * norm2, min=1e-10)
    cka = dotp / denom
    
    return cka.mean().item()