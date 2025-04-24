import torch.nn.functional as F
from typing import Tuple, Optional
import torch
from torch import Tensor, nn

def pad_at_dim(t, pad: Tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def divisible_by(num, den):
    return (num % den) == 0


def is_empty(t: Tensor):
    return t.numel() == 0


def round_down_multiple(n, mult):
    return n // mult * mult


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    indices = cumulative_probs > thres
    indices[..., 1:] = indices[..., :-1].clone()
    indices[..., 0] = 0

    indices_to_remove = indices.scatter(1, sorted_indices, indices)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


def top_k(logits, k: Optional[int] = None, frac_num_tokens=0.1):
    num_tokens = logits.shape[-1]
    k = default(k, max(int(frac_num_tokens * num_tokens), 1))
    k = min(k, num_tokens)

    topk_vals, topk_indices = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, topk_indices, topk_vals)
    return probs


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise + 1e-20) + 1e-20)


def gumbel_sample(logits, temperature=1.0, dim=-1, keepdim=True):
    noise = gumbel_noise(logits)
    y = logits / temperature + noise
    return torch.argmax(y, dim=dim, keepdim=keepdim)


def apply_repetition_penalty(logits: torch.Tensor, tokens: torch.Tensor, repetition_penalty: float) -> torch.Tensor:
    if repetition_penalty == 1.0 or tokens.numel() == 0:
        return logits

    batch_size, vocab_size = logits.size()
    device = logits.device

    mask = torch.zeros((batch_size, vocab_size), device=device, dtype=torch.bool)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(tokens)
    mask[batch_indices.flatten(), tokens.flatten()] = True

    token_logits = logits[mask]
    logits_clone = logits.clone()

    negative_mask = token_logits < 0
    positive_mask = ~negative_mask

    indices = mask.nonzero(as_tuple=False)

    neg_idx = indices[negative_mask]
    pos_idx = indices[positive_mask]

    logits_clone[neg_idx[:, 0], neg_idx[:, 1]] *= repetition_penalty
    logits_clone[pos_idx[:, 0], pos_idx[:, 1]] /= repetition_penalty

    return logits_clone


def alpha_fn(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dtype == torch.uint8, "Input tensor must be uint8"
    if tensor.shape[0] == 4:
        rgb = tensor[:3].float()
        alpha = tensor[3].float() / 255.0
        out = alpha * rgb + (1 - alpha) * 255.0
        out = out.clamp(0, 255).to(tensor.dtype)
        return out
    return tensor


class AlphaComposite(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return alpha_fn(tensor)
