
# modified from https://github.com/thunlp/LLaVA-UHD/blob/eecf39c0739891b5f64c05297b190d15a292e45a/featup/upsamplers.py
from spcu import compile_extension
from typing import Union, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange
from dataclasses import dataclass
from sr_utils import FeatureConsistencyLoss
from typing import Optional


@dataclass
class SRBlockOutput:
    sr_features: torch.Tensor
    fc_loss: Optional[torch.Tensor] = None


class SpatialRangeBlock(nn.Module):
    def __init__(
        self,
        s_dim = 1152,
        d_dim = 1536,
        patch_size: Union[int, Tuple[int, int]] = 16,
        radius: int = 3,
        use_cuda_kernel: bool = True,
        d_encoder: nn.Module = None,
        s_encoder: nn.Module = None,
        train_block: bool = False,
        depth: int = 4,
        fcl_module: FeatureConsistencyLoss = None,
        init_weights: bool = False
    ):
        super().__init__()
        self.dim = s_dim + d_dim
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.d_encoder = d_encoder
        self.s_encoder = s_encoder
        self.train_block = train_block
        self.FCL = fcl_module

        if use_cuda_kernel:
            SpatialRangeAttentionCUDAFunction.cuda_module = compile_extension()
            attn_cls = SpatialRangeAttentionCUDA
        else:
            attn_cls = SpatialRangeAttention

        self.s_norm = nn.LayerNorm(s_dim)
        self.d_norm = nn.LayerNorm(d_dim)

        self.s_proj = nn.Linear(s_dim, d_dim, bias=False)
        self.proj_out = nn.Linear(d_dim * 2, d_dim * 2, bias=False)

        self.sr_attn1_layers = nn.ModuleList([])
        for _ in range(depth):
            self.sr_attn1_layers.append(
                nn.ModuleList(
                    [
                        attn_cls(dim=d_dim, radius=radius),
                        attn_cls(dim=d_dim, radius=radius)
                    ]
                )
            )

        self.sr_attn2_layers = nn.ModuleList([])
        for _ in range(depth):
            self.sr_attn2_layers.append(
                nn.ModuleList(
                    [
                        attn_cls(dim=d_dim * 2, radius=radius),
                        attn_cls(dim=d_dim * 2, radius=radius)
                    ]
                )
            )
        
        if init_weights:
            nn.init.xavier_uniform_(self.s_proj.weight)

    def forward(self, d_inputs, s_inputs):
        fc_loss = None
        dtype = next(self.s_proj.parameters()).dtype
        
        with torch.autocast(device_type=d_inputs.device.type, enabled=False):
            d_features = self.d_encoder(d_inputs)
            d_features = d_features.last_hidden_state
            s_features = self.s_encoder(s_inputs)
            s_features = s_features.last_hidden_state

        if self.train_block:
            d_features = d_features.requires_grad_()
            s_features = s_features.requires_grad_()
            assert d_features.requires_grad, "Dinov2 output requires gradients."
            assert s_features.requires_grad, "Siglip2 output requires gradients."

        s_in = s_features.to(dtype)
        d_in = d_features.to(dtype)
        
        s_features = self.s_norm(s_in)
        s_features = self.s_proj(s_features)

        d_features = self.d_norm(d_in)

        sd1_features = torch.cat([d_features, s_features], dim=-1).to(torch.float32)

        s_features, d_features, sd1_features = [
            rearrange(x, 'b (h w) c -> b c h w', h=self.patch_size[0], w=self.patch_size[1]
                      ) for x in [
                          s_features, d_features, sd1_features
                          ]
            ]
        
        d_features = d_features.to(torch.float32)
        
        for sr1, sr2 in self.sr_attn1_layers:
            s_features = sr1(d_features, s_features) + s_features
            s_features = sr2(d_features, s_features) + s_features

        sd2_features = torch.cat([d_features, s_features], dim=1).to(dtype)

        for sr1, sr2 in self.sr_attn2_layers:
            sd2_features = sr1(sd1_features, sd2_features) + sd2_features
            sd2_features = sr2(sd1_features, sd2_features) + sd2_features
        
        if self.train_block:
            s_in, d_in = [
            rearrange(x, 'b (h w) c -> b c h w', h=self.patch_size[0], w=self.patch_size[1]
                      ) for x in [
                          s_in, d_in
                          ]
            ]
            fc_loss1 = self.FCL(d_in, sd2_features)
            fc_loss2 = self.FCL(s_in, sd2_features)
            fc_loss = fc_loss1 + fc_loss2

        output = rearrange(
                sd2_features,
                "b c h w -> b (h w) c",
                h=self.patch_size[0],
                w=self.patch_size[1]
            )
        
        output = self.proj_out(output)

        return SRBlockOutput(
            sr_features=output,
            fc_loss=fc_loss
        )


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SpatialRangeAttentionCUDAFunction(Function):
    cuda_module = None
    
    @staticmethod
    def forward(ctx, proj_feats, spatial_feats, sigma_spatial, radius):
        assert proj_feats.is_cuda and spatial_feats.is_cuda
        assert proj_feats.is_contiguous() and spatial_feats.is_contiguous()
        assert proj_feats.dim() == 4 and spatial_feats.dim() == 4
        assert proj_feats.dtype == torch.float32, f"proj_feats must be float32, got {proj_feats.dtype}"
        assert spatial_feats.dtype == torch.float32, f"spatial_feats must be float32, got {spatial_feats.dtype}"
        assert sigma_spatial.dtype == torch.float32, f"sigma_spatial must be float32, got {sigma_spatial.dtype}"

        outputs = SpatialRangeAttentionCUDAFunction.cuda_module.sp_attn_fwd(
            proj_feats,
            spatial_feats,
            sigma_spatial.item(),
            radius
        )

        attention_weights, spatial_weights, neighborhoods = outputs

        ctx.save_for_backward(
            proj_feats,
            spatial_feats,
            attention_weights,
            spatial_weights,
            sigma_spatial
        )
        ctx.radius = radius
        
        return attention_weights, neighborhoods
    
    @staticmethod
    def backward(ctx, grad_combined_kernel, grad_neighborhoods):
        proj_feats, spatial_feats, attention_weights, spatial_weights, sigma_spatial = ctx.saved_tensors
        radius = ctx.radius

        grad_proj, grad_spatial, grad_sigma = SpatialRangeAttentionCUDAFunction.cuda_module.sp_attn_bwd(
            grad_combined_kernel.contiguous(),
            grad_neighborhoods.contiguous(),
            attention_weights,
            spatial_weights,
            proj_feats,
            spatial_feats,
            sigma_spatial.item(),
            radius
        )

        return grad_proj, grad_spatial, grad_sigma, None


class SpatialRangeAttentionCUDA(nn.Module):
    def __init__(
        self,
        dim: int,
        radius: int = 3,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.radius = radius
        self.diameter = 2 * radius + 1
        self.dim = dim
        self.scale = (dim ** 0.5)

        self.range_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding),
            LayerNorm2d(dim),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding)
        )

        self.fixup_proj = nn.Sequential(
            nn.Conv2d(self.diameter ** 2 + dim, self.diameter ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
            LayerNorm2d(self.diameter ** 2),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(self.diameter ** 2, self.diameter ** 2, kernel_size=kernel_size, stride=stride, padding=padding)
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding),
            LayerNorm2d(dim),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, spatial_feats: torch.Tensor, semantic_feats: torch.Tensor) -> torch.Tensor:
        assert spatial_feats.dtype == torch.float32, f"spatial_feats must be float32, got {spatial_feats.dtype}"
        proj_x = self.range_proj(semantic_feats).to(spatial_feats.dtype)

        assert self.sigma_spatial.dtype == torch.float32, f"sigma_spatial must be float32, got {self.sigma_spatial.dtype}"
        sigma_spatial = self.sigma_spatial.view(1).contiguous()

        combined_kernel, neighborhoods = SpatialRangeAttentionCUDAFunction.apply(
            proj_x.contiguous(),
            spatial_feats.contiguous(),
            sigma_spatial,
            self.radius
        )

        combined_kernel = combined_kernel.to(semantic_feats.dtype)

        combined_kernel = combined_kernel + torch.einsum(
            'b c h w, b c h w -> b c h w',
            combined_kernel,
            torch.sigmoid(self.fixup_proj(torch.cat([combined_kernel, semantic_feats], dim=1)))
        )
        
        combined_kernel = combined_kernel / (combined_kernel.sum(1, keepdim=True) + 1e-7)

        neighborhoods = neighborhoods.to(combined_kernel.dtype)

        output = torch.einsum('b c n h w, b c n h w -> b c h w', neighborhoods, combined_kernel.unsqueeze(1))

        output = self.output_proj(output)

        return output


class SpatialRangeAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        radius: int = 3,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.radius = radius
        self.diameter = 2 * radius + 1
        self.dim = dim
        self.scale = (dim ** 0.5)

        self.range_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding),
            LayerNorm2d(dim),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding)
        )

        self.fixup_proj = nn.Sequential(
            nn.Conv2d(self.diameter ** 2 + dim, self.diameter ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
            LayerNorm2d(self.diameter ** 2),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(self.diameter ** 2, self.diameter ** 2, kernel_size=kernel_size, stride=stride, padding=padding)
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding),
            LayerNorm2d(dim),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def get_spatial_kernel(self, device: torch.device) -> torch.Tensor:
        dist_range = torch.linspace(-1, 1, self.diameter, device=device)
        x, y = torch.meshgrid(dist_range, dist_range, indexing='ij')
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        spatial_weights = torch.exp(
            -patch.square().sum(0) / (2 * self.sigma_spatial ** 2)
        ).reshape(1, self.diameter ** 2, 1, 1)

        return spatial_weights

    def get_range_kernel(self, proj_x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = proj_x.shape

        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = F.unfold(
            proj_x_padded,
            kernel_size=self.diameter,
            dilation=1,
            padding=0
        ).reshape(B, -1, self.diameter ** 2, H, W)

        proj_x = rearrange(proj_x, 'b c h w -> b c 1 h w')

        sim = torch.einsum('b c n h w, b c n h w -> b n h w', queries, proj_x) / self.scale

        attn = sim.softmax(dim=1, dtype=torch.float32).to(proj_x.dtype)

        return attn

    def forward(self, spatial_feats: torch.Tensor, semantic_feats: torch.Tensor) -> torch.Tensor:
        assert spatial_feats.dtype == torch.float32, f"spatial_feats must be float32, got {spatial_feats.dtype}"
        assert self.sigma_spatial.dtype == torch.float32, f"sigma_spatial must be float32, got {self.sigma_spatial.dtype}"
        proj_x = self.range_proj(semantic_feats)
        proj_x = proj_x.to(spatial_feats.dtype)

        spatial_kernel = self.get_spatial_kernel(spatial_feats.device).to(spatial_feats.dtype)
        range_kernel = self.get_range_kernel(proj_x)

        padded_spatial = F.pad(spatial_feats, pad=[self.radius] * 4, mode='reflect')
        neighborhoods = F.unfold(
            padded_spatial,
            kernel_size=self.diameter,
            dilation=1,
            padding=0
        ).reshape(-1, self.dim, self.diameter ** 2, spatial_feats.shape[2], spatial_feats.shape[3])
        
        combined_kernel = range_kernel * spatial_kernel

        combined_kernel = combined_kernel.to(semantic_feats.dtype)

        combined_kernel = combined_kernel + torch.einsum(
            'b c h w, b c h w -> b c h w',
            combined_kernel,
            torch.sigmoid(self.fixup_proj(torch.cat([combined_kernel, semantic_feats], dim=1)))
        )
        
        combined_kernel = combined_kernel / (combined_kernel.sum(1, keepdim=True) + 1e-7)

        neighborhoods = neighborhoods.to(combined_kernel.dtype)

        output = torch.einsum('b c n h w, b c n h w -> b c h w', neighborhoods, combined_kernel.unsqueeze(1))

        output = self.output_proj(output)

        return output
