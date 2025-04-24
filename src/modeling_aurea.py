from configuration_phi4 import Phi4Config
from transformers.activations import ACT2FN
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Union, Dict, Tuple, Any, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat
from torch import einsum, Tensor
from einops.layers.torch import Rearrange
from utils import exists, pad_at_dim


class AureaPhi4Config(Phi4Config):
    model_type = "aurea_phi4"

    def __init__(
            self,
            use_cache=True,
            use_checkpointing=False,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.use_cache = use_cache
        self.use_checkpointing = use_checkpointing
        
        print(
            f"use_checkpointing={self.use_checkpointing}\n"
            f"use_cache={self.use_cache}"
        )


# copied from Phi3
class Phi4RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi4RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# copied from Phi3
class Phi4RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# copied & modified from Phi3
class Phi4LongRoPEScaledRotaryEmbedding(Phi4RotaryEmbedding):
    def __init__(self, dim, config):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta)
        self.dim = dim
        self.long_factor = torch.tensor(config.rope_scaling["long_factor"], dtype=torch.float32)
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.max_position_embeddings = config.max_position_embeddings

        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64)
        inv_freq = 1.0 / (self.long_factor * self.base ** (inv_freq_shape.float() / self.dim))
        self.register_buffer("inv_freq", inv_freq.unsqueeze(0).unsqueeze(-1).to(dtype=torch.float32), persistent=False)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        scaling_factor = 1.0 if scale <= 1.0 else math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        self.register_buffer("scaling_factor", torch.tensor(scaling_factor, dtype=torch.float32), persistent=False)

    def forward(self, x, position_ids):
        position_ids_expanded = position_ids[:, None, :].float()
        
        with torch.autocast(device_type=x.device.type, enabled=False):
            assert self.inv_freq.dtype == torch.float32
            assert self.scaling_factor.dtype == torch.float32
            freqs = (self.inv_freq.to(x.device) @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

            cos = emb.cos() * self.scaling_factor
            sin = emb.sin() * self.scaling_factor

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# copied from Phi3
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# copied from Phi3
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


# copied from Phi3
class Phi4MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


# copied from Phi3
def repeatkv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# copied from Phi3
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeatkv(key, module.num_key_value_groups)
    value_states = repeatkv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Adapted from Phi3
class Phi4Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: AureaPhi4Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            raise ValueError(f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not allowed...")

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(self.hidden_size, op_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.config.num_attention_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class VisualKVCache:
    def __init__(self):
        self.key_cache: list[Optional[torch.Tensor]] = []
        self.value_cache: list[Optional[torch.Tensor]] = []
    
    def __len__(self) -> int:
        return len(self.key_cache)
    
    def has_layer(self, layer_idx: int) -> bool:
        if layer_idx >= len(self):
            return False
        return (self.key_cache[layer_idx] is not None and 
                self.value_cache[layer_idx] is not None)
    
    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self):
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer {layer_idx}")
            
        if not self.has_layer(layer_idx):
            raise ValueError(f"Layer {layer_idx} exists but has not been initialized")
            
        return (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx]
        )
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(self) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        
        return key_states, value_states


@dataclass
class CrossAttentionOutput:
    hidden_states: torch.Tensor
    attn_weights: Optional[torch.Tensor] = None
    v_kv_cache: Optional[VisualKVCache] = None


class CrossAttention(nn.Module):
    def __init__(
        self,
        config: AureaPhi4Config,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = Phi4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Phi4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.attention_dropout)

        self.merge_heads = Rearrange('b h n d -> b n (h d)', h=self.num_heads, d=self.head_dim)

    def repeat_kv(self, t: torch.Tensor, r: int) -> torch.Tensor:
        return rearrange(repeat(t, 'b kvh n d -> b kvh r n d', r=r), 'b kvh r n d -> b (kvh r) n d')
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        v_features: Optional[torch.Tensor] = None,
        v_kv_cache: Optional[VisualKVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> CrossAttentionOutput:

        q = self.q_proj(hidden_states)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        q = self.q_norm(q) * self.scale

        mask_value = -torch.finfo(q.dtype).max
        
        if exists(v_features):
            k = self.k_proj(v_features)
            v = self.v_proj(v_features)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_key_value_heads), (k, v))
            
            k = self.k_norm(k)
            
            if self.num_key_value_groups > 1:
                k, v = map(lambda t: self.repeat_kv(t, self.num_key_value_groups), (k, v))

            if exists(v_kv_cache):
                k, v = v_kv_cache.update(k, v, self.layer_idx)
                
        elif exists(v_kv_cache) and v_kv_cache.has_layer(self.layer_idx):
            k, v = v_kv_cache.get_layer_cache(self.layer_idx)
        else:
            raise ValueError(
                "Cross attention requires either v_features or cached key/value states!"
            )
        
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(attention_mask):
            attention_mask = rearrange(attention_mask, 'b i -> b 1 i 1')
            sim = sim.masked_fill(~attention_mask, mask_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32).to(v.dtype)
        
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = self.merge_heads(out)

        out = self.o_proj(out)
        
        return CrossAttentionOutput(
            hidden_states=out,
            attn_weights=attn if output_attentions else None,
            v_kv_cache=v_kv_cache
        )


@dataclass
class AttentionOutput:
    attn_out: torch.Tensor
    attn_weights: Optional[torch.Tensor] = None
    pkv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


class EagerAttention(Phi4Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = self.head_dim ** -0.5
        self.merge_heads = Rearrange('b h n d -> b n (h d)', h=self.num_heads, d=self.head_dim)
        self.dropout = nn.Dropout(self.attention_dropout)
        self.use_checkpointing = self.config.use_checkpointing
    
    def repeat_kv(self, t: torch.Tensor, r: int) -> torch.Tensor:
        return rearrange(repeat(t, 'b kvh n d -> b kvh r n d', r=r), 'b kvh r n d -> b (kvh r) n d')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        rotary_pos: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ) -> AttentionOutput:

        qkv = self.qkv_proj(hidden_states)

        q, k, v = qkv.split([
            self.num_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim
        ], dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_key_value_heads), (k, v))

        cos, sin = rotary_pos

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if exists(past_key_value):
            cache_kwargs = {
                "sin": sin, 
                "cos": cos
            }
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        q = q * self.scale

        if self.num_key_value_groups > 1:
            k, v = map(lambda t: self.repeat_kv(t, self.num_key_value_groups), (k, v))

        mask_value = -torch.finfo(q.dtype).max

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        i, j = sim.shape[-2:]

        causal_mask = torch.ones((i, j), dtype=torch.bool, device=sim.device).triu(j - i + 1)

        sim = torch.where(causal_mask, mask_value, sim)

        if exists(attention_mask):
            attention_mask = rearrange(attention_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attention_mask, mask_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32).to(v.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = self.merge_heads(out)
                
        out = self.o_proj(out)

        return AttentionOutput(
            attn_out=out, 
            attn_weights=attn if output_attentions else None, 
            pkv=past_key_value
        )


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4., dropout=0.):
        super().__init__()
        dim_inner = int(dim * expansion_factor * 2 / 3)
        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner * 2, bias = False),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim, bias = False)
        )
    
    def forward(self, x):
        return self.net(x)


@dataclass
class MultimodalBlockOutput:
    hidden_states: torch.Tensor
    attn_weights: Optional[torch.Tensor] = None
    v_kv_cache: Optional[VisualKVCache] = None


class MultimodalBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.cross_attn = CrossAttention(config, layer_idx)
        self.input_layernorm = Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.ffw = FeedForward(
            dim=config.hidden_size,
            expansion_factor=config.expansion_factor,
            dropout=config.resid_pdrop
        )
        self.post_attention_layernorm = Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ff_dropout = nn.Dropout(config.resid_pdrop)
        
        self.use_checkpointing = config.use_checkpointing
    
    def set_checkpointing(self, use_checkpointing):
        self.use_checkpointing = use_checkpointing
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        v_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        v_kv_cache: Optional[VisualKVCache] = None,
        output_attentions: Optional[bool] = False
    ) -> MultimodalBlockOutput:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        outputs = self.cross_attn(
            hidden_states=hidden_states,
            v_features=v_features,
            v_kv_cache=v_kv_cache,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        
        hidden_states = residual + self.resid_dropout(outputs.hidden_states)
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffw(hidden_states)
        hidden_states = residual + self.ff_dropout(hidden_states)
        
        return MultimodalBlockOutput(
            hidden_states=hidden_states,
            attn_weights=outputs.attn_weights,
            v_kv_cache=outputs.v_kv_cache
        )


@dataclass
class LayerOutput:
    hidden_states: torch.Tensor
    attn_weights: Optional[torch.Tensor] = None
    present_key_value: Optional[Tuple[torch.Tensor, ...]] = None


#adapted from Phi3DecoderLayer
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = EagerAttention(config, layer_idx)
        self.mlp = Phi4MLP(config)
        self.input_layernorm = Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_checkpointing = config.use_checkpointing

    def set_checkpointing(self, use_checkpointing):
        self.use_checkpointing = use_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        rotary_pos: Optional[Tuple[torch.Tensor]] = None
    ) -> LayerOutput:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            rotary_pos=rotary_pos
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs.attn_out)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        return LayerOutput(
            hidden_states=hidden_states, 
            attn_weights=attn_outputs.attn_weights, 
            present_key_value=attn_outputs.pkv
        )


class AureaPhi4PreTrainedModel(PreTrainedModel):
    config_class = AureaPhi4Config
    base_model_prefix = "model"
    _no_split_modules = ["Block"]
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_multimodal_attention = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    _version = "0.0.1"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class BaseModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[Tuple[torch.Tensor, ...]] = None
    v_kv_cache: Optional[VisualKVCache] = None
    all_hidden_states: Optional[list[torch.Tensor]] = None
    all_attentions: Optional[list[torch.Tensor]] = None


# adapted from Phi3Model
class AureaPhi4Model(AureaPhi4PreTrainedModel):
    def __init__(self, config: AureaPhi4Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self._attn_implementation = config._attn_implementation
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        
        self.layers = nn.ModuleList(
            [Block(
                config=config, 
                layer_idx=i, 
                **kwargs
                ) 
                for i in range(
                    self.config.num_hidden_layers
                    )
                ]
            )
        
        self.norm = Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self._init_rope()
        self.post_init()

    def _init_rope(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        dim = int(head_dim * self.config.partial_rotary_factor)
        if self.config.rope_scaling is None:
            self.rotary_emb = Phi4RotaryEmbedding(
                dim,
                max_position_embeddings=self.config.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "longrope":
                self.rotary_emb = Phi4LongRoPEScaledRotaryEmbedding(dim, self.config)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_checkpointing(self, use_checkpointing: bool):
        for layer in self.layers:
            layer.set_checkpointing(use_checkpointing)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        v_features: Optional[torch.FloatTensor] = None,
        v_kv_cache: Optional[VisualKVCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        has_image_features = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            seq_length = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if use_cache or exists(past_key_values):
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if exists(v_features) or exists(v_kv_cache):
            has_image_features = True
        
        if use_cache and exists(v_features) and v_kv_cache is None:
            v_kv_cache = VisualKVCache()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, 
                seq_length + past_key_values_length, 
                dtype=torch.long, 
                device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        
        rotary_pos = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        next_decoder_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if isinstance(layer, MultimodalBlock):
                if has_image_features:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        v_features=v_features,
                        v_kv_cache=v_kv_cache,
                        output_attentions=output_attentions
                    )
                else:
                    continue
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    rotary_pos=rotary_pos
                )

            hidden_states = layer_outputs.hidden_states

            if isinstance(layer, MultimodalBlock):
                if layer_outputs.v_kv_cache:
                    v_kv_cache = layer_outputs.v_kv_cache
            elif isinstance(layer, Block):
                if layer_outputs.present_key_value:
                    next_decoder_cache = layer_outputs.present_key_value

            if output_attentions:
                all_attentions.append(layer_outputs.attn_weights)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        next_cache = None

        if exists(next_decoder_cache):
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            v_kv_cache=v_kv_cache,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions
        )


class CausalLMLoss(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self, 
        logits: torch.FloatTensor, 
        labels: torch.LongTensor
        ) -> torch.FloatTensor:
        
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        labels = labels.to(logits.device)

        loss = F.cross_entropy(logits, labels, ignore_index=-100)

        return loss


@dataclass
class CausalLMOutput:
    loss: Optional[Tensor] = None
    logits: Tensor = None
    past_key_values: Optional[Tuple[Tensor, ...]] = None
    v_kv_cache: Optional[VisualKVCache] = None
    all_hidden_states: Optional[list[torch.Tensor]] = None
    all_attentions: Optional[list[torch.Tensor]] = None


class VProjector(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        output_dim: int = 3072,
        mult: int = 4,
        activation: nn.Module = nn.SiLU,
        init_weights: bool = False
    ):
        super().__init__()
        dim_inner = input_dim * mult
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, dim_inner, bias=False),
            activation(),
            nn.Linear(dim_inner, output_dim, bias=False)
        )
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.xavier_uniform_(self.projection[2].weight)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.projection(x)
        return out


# adapted from Phi3ForCausalLM
class AureaPhi4ForCausalLM(AureaPhi4PreTrainedModel):
    config_class = AureaPhi4Config
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: AureaPhi4Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config 
        self.model = AureaPhi4Model(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.loss = CausalLMLoss(self.vocab_size)
        self.sr_block = None
        self.v_proj = None
        self.post_init()

    def set_sr_block(self, sr_block):
        self.sr_block = sr_block
    
    def get_v_features(self, d_inputs, s_inputs):
        return self.sr_block(d_inputs=d_inputs, s_inputs=s_inputs)
    
    def set_v_projector(self, v_proj):
        self.v_proj = v_proj
    
    def project_v_features(self, vf):
        return self.v_proj(vf)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        v_features: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        v_kv_cache: Optional[VisualKVCache] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False
    ) -> Union[Tuple, CausalLMOutputWithPast, CausalLMOutput]:

        output_attentions = (
            output_attentions
            if output_attentions is not None 
            else self.config.output_attentions
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None 
            else self.config.output_hidden_states
        )
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            v_kv_cache=v_kv_cache,
            inputs_embeds=input_embeds,
            v_features=v_features,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        hidden_states = outputs.last_hidden_state
        
        logits = self.lm_head(hidden_states)
        
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            v_kv_cache=outputs.v_kv_cache,
            all_hidden_states=outputs.all_hidden_states,
            all_attentions=outputs.all_attentions
        )
    
    def prepare_args(
            self, 
            input_ids: Optional[torch.LongTensor] = None,
            input_embeds: Optional[torch.FloatTensor] = None, 
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
            position_ids: Optional[torch.LongTensor] = None
        ):
        if input_ids is None and input_embeds is None:
            raise ValueError("You have to specify either `input_ids` or `input_embeds`.")

        args = {}
        if input_embeds is not None and past_key_values is None:
            args["input_embeds"] = input_embeds
        elif input_embeds is None and past_key_values is None:
            args["input_ids"] = input_ids
        
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None
            
            if attention_mask is not None:
                if attention_mask.shape[1] > input_ids.shape[1]:
                    start_index = -(attention_mask.shape[1] - past_length)
                    input_ids = input_ids[:, start_index:]
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]

                if attention_mask.shape[1] != (input_ids.shape[1] + past_length):
                    ext_mask_len = (input_ids.shape[1] + past_length) - input_ids.shape[1]
                    attention_mask = pad_at_dim(attention_mask, (0, ext_mask_len), dim=-1, value=True)

            if max_cache_length is not None and attention_mask is not None:
                total_length = cache_length + input_ids.shape[1]
                if total_length > max_cache_length:
                    attention_mask = attention_mask[:, -max_cache_length:]
            
            args["input_ids"] = input_ids
        
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return args, past_key_values, attention_mask, position_ids

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
        use_cache: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        v_features: Optional[torch.FloatTensor] = None,
        v_kv_cache: Optional[VisualKVCache] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        args, pkv, attn_mask, pos_ids = self.prepare_args(
            input_ids=input_ids, 
            input_embeds=input_embeds, 
            past_key_values=past_key_values, 
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        return {
            **args,
            "past_key_values": pkv,
            "attention_mask": attn_mask,
            "position_ids": pos_ids,
            "use_cache": use_cache,
            "v_features": v_features,
            "v_kv_cache": v_kv_cache
        }
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


AutoConfig.register("aurea_phi4", AureaPhi4Config)
AutoModelForCausalLM.register(AureaPhi4Config, AureaPhi4ForCausalLM)