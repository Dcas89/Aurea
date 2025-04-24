import math
from functools import partial
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class AttentionOutput:
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None

@dataclass
class BlockOutput:
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None

@dataclass
class FeaturesOutput:
    x_norm_clstoken: torch.Tensor
    x_norm_patchtokens: torch.Tensor
    x_prenorm: torch.Tensor
    masks: Optional[torch.Tensor] = None
    x_norm_regtokens: Optional[torch.Tensor] = None

@dataclass
class TransformerOutput:
    last_hidden_state: torch.Tensor
    pooler_output: Optional[torch.Tensor] = None
    register_tokens: Optional[torch.Tensor] = None
    hidden_states: Optional[list[torch.Tensor]] = None
    attentions: Optional[list[torch.Tensor]] = None

@dataclass
class IntermediateLayers:
    outputs: list[torch.Tensor]
    class_tokens: Optional[list[torch.Tensor]] = None

    def __iter__(self):
        if self.class_tokens is not None:
            return iter(zip(self.outputs, self.class_tokens))
        return iter(self.outputs)
        
    def __getitem__(self, idx):
        if self.class_tokens is not None:
            return (self.outputs[idx], self.class_tokens[idx])
        return self.outputs[idx]


class Dinov2WithRegistersConfig:
    model_type = "dinov2_with_registers"
    
    def __init__(
        self,
        image_size=518,
        patch_size=14,
        num_channels=3,
        hidden_size=1536,
        num_hidden_layers=40,
        num_attention_heads=24,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_value=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        use_swiglu_ffn=True,
        num_register_tokens=4,
        attn_implementation="sdpa",
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.layerscale_value = layerscale_value
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_swiglu_ffn = use_swiglu_ffn
        self.num_register_tokens = num_register_tokens
        self.attn_implementation = attn_implementation


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
            
        return x * random_tensor

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        self.patches_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.projection(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x

class EagerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_bias=None, output_attentions=False) -> AttentionOutput:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return AttentionOutput(hidden_states=x, attention_weights=attn if output_attentions else None)

class SDPAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, 
                x: torch.Tensor, 
                attn_bias=None,
                **kwargs,
            ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return AttentionOutput(hidden_states=x, attention_weights=None)

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_class: Callable[..., nn.Module] = SDPAttention,
        ffn_layer: Callable[..., nn.Module] = SwiGLUFFNFused,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = self._create_ffn(ffn_layer, dim, mlp_ratio, act_layer, drop, ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def _create_ffn(self, ffn_layer, dim, mlp_ratio, act_layer, drop, ffn_bias):
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if ffn_layer == Mlp:
            return ffn_layer(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                bias=ffn_bias,
            )
        elif ffn_layer == SwiGLUFFNFused:
            return ffn_layer(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                bias=ffn_bias,
            )
        else:
            raise ValueError(f"Unsupported ffn_layer: {ffn_layer}")
    def forward(self, x: torch.Tensor, attn_bias=None, output_attentions=False) -> BlockOutput:
        attn_output = self.attn(self.norm1(x), attn_bias=attn_bias, output_attentions=output_attentions)

        residual_attn = self.ls1(attn_output.hidden_states)
        x = x + self.drop_path1(residual_attn)
        
        residual_ffn = self.ls2(self.mlp(self.norm2(x)))
        x = x + self.drop_path2(residual_ffn)

        return BlockOutput(hidden_states=x, attention_weights=attn_output.attention_weights)


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None, 
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        attn_implementation="eager",
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_layer_class = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer_class = SwiGLUFFNFused
        else:
            raise NotImplementedError(f"Unknown FFN layer type: {ffn_layer}")

        if attn_implementation == "sdpa":
            attn_class = SDPAttention
        elif attn_implementation == "eager":
            attn_class = EagerAttention
        else:
            raise ValueError(f"Unknown attention implementation: {attn_implementation}")

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer_class,
                init_values=init_values,
                attn_class=attn_class,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w // self.patch_size == h // self.patch_size:
            return self.pos_embed.to(previous_dtype)
            
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_size
        h0 = h // self.patch_size

        M = int(math.sqrt(N))  
        assert N == M * M, "Position embedding grid must be square"

        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        
        assert (w0, h0) == patch_pos_embed.shape[-2:], "Interpolation yielded unexpected shape"
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        _, _, w, h = x.shape

        x = self.patch_embed(x)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1), 
                    x[:, 1:],
                ),
                dim=1,
            )

        return x
    
    def forward(
            self, 
            pixel_values, 
            masks=None, 
            output_hidden_states=False, 
            output_attentions=False, 
            attn_bias=None
        ) -> TransformerOutput:
        x = self.prepare_tokens_with_masks(pixel_values, masks)
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        for block in self.blocks:
            block_output = block(x, attn_bias=attn_bias, output_attentions=output_attentions)
            x = block_output.hidden_states
            
            if output_attentions:
                all_attentions.append(block_output.attention_weights)
                
            if output_hidden_states:
                all_hidden_states.append(x)
        
        normalized = self.norm(x)

        return TransformerOutput(
            last_hidden_state=normalized[:, self.num_register_tokens + 1:],
            pooler_output=normalized[:, 0],
            register_tokens=normalized[:, 1:self.num_register_tokens + 1] if self.num_register_tokens > 0 else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions if output_attentions else None
        )


class Dinov2WithRegistersModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        init_values = self.config.layerscale_value if self.config.layerscale_value > 0 else None
        ffn_layer = "swiglufused" if self.config.use_swiglu_ffn else "mlp"

        self.model = DinoVisionTransformer(
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            in_chans=self.config.num_channels,
            embed_dim=self.config.hidden_size,
            depth=self.config.num_hidden_layers,
            num_heads=self.config.num_attention_heads,
            mlp_ratio=self.config.mlp_ratio,
            qkv_bias=self.config.qkv_bias,
            ffn_bias=True,
            proj_bias=True,
            drop_path_rate=self.config.drop_path_rate,
            drop_path_uniform=False,
            init_values=init_values,
            embed_layer=PatchEmbed,
            act_layer=nn.GELU,
            block_fn=Block,
            ffn_layer=ffn_layer,
            num_register_tokens=self.config.num_register_tokens,
            interpolate_antialias=self.config.interpolate_antialias,
            interpolate_offset=self.config.interpolate_offset,
            attn_implementation=self.config.attn_implementation,
        )
        
    def forward(
            self, 
            pixel_values, 
            masks=None, 
            output_hidden_states=False, 
            output_attentions=False, 
            attn_bias=None
        ) -> TransformerOutput:
        
        return self.model(
            pixel_values=pixel_values,
            masks=masks,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            attn_bias=attn_bias
        )