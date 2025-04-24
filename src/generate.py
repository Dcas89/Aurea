import torch
from utils import top_p, top_k, gumbel_sample, apply_repetition_penalty
from tqdm import tqdm
from typing import Callable, Optional
from inspect import signature

@torch.no_grad()
def text_generation(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    d_features: Optional[torch.Tensor] = None,
    s_features: Optional[torch.Tensor] = None,
    eos_token: Optional[int] = None,
    temperature: float = 0.3,
    repetition_penalty: float = 1.1,
    use_dynamic_top_k: bool = False,
    min_top_k: int = 50,
    max_top_k: int = 90,
    filter_fn: Optional[Callable[..., torch.Tensor]] = None,
    filter_kwargs: dict[str, any] = {'thres': 0.9, 'top_k': 50},
    exclude_prompt: bool = True,
    show_progress: bool = True
) -> torch.LongTensor:

    device = input_ids.device
    model.eval()

    seq_len = input_ids.shape[1]
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0.")
    max_length = seq_len + max_new_tokens

    if d_features is not None and s_features is not None:
        d_features, s_features = d_features.to(device), s_features.to(device)
        sr = model.get_v_features(d_inputs=d_features, s_inputs=s_features)
        v_features = model.project_v_features(sr.sr_features)
    else:
        v_features = None

    if filter_fn is not None:
        sig = signature(filter_fn)
        valid_keys = set(sig.parameters.keys())
        filter_kwargs = {k: v for k, v in filter_kwargs.items() if k in valid_keys}

    out = input_ids.clone()
    generated = []
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    next_pos = seq_len
    past_kv = past_vkv = None

    it = range(seq_len, max_length)
    if show_progress:
        it = tqdm(it, desc="Generating")

    for step in it:
        if step == seq_len:
            curr_ids, curr_pos, curr_vf = out, pos_ids, v_features
        else:
            curr_ids = out[:, -1:]
            curr_pos = torch.tensor([[next_pos]], device=device)
            curr_vf = None
            next_pos += 1

        inp = model.prepare_inputs_for_generation(
            input_ids=curr_ids,
            past_key_values=past_kv,
            v_kv_cache=past_vkv,
            v_features=curr_vf,
            position_ids=curr_pos,
            use_cache=True
        )

        outp = model(**inp)
        logits = outp.logits[:, -1, :]
        past_kv = outp.past_key_values
        past_vkv = outp.v_kv_cache

        if generated and repetition_penalty != 1.0:
            tokens = torch.cat(generated, dim=1)
            logits = apply_repetition_penalty(logits, tokens, repetition_penalty)

        if use_dynamic_top_k:
            frac = (step - seq_len) / (max_length - seq_len)
            curr_k = int(max_top_k * (1 - frac) + min_top_k * frac)
            filtered = top_k(logits, k=curr_k)
        elif filter_fn is not None:
            filtered = filter_fn(logits, **filter_kwargs)
        else:
            filtered = top_p(top_k(logits, k=filter_kwargs['top_k']), thres=filter_kwargs['thres'])

        next_tok = gumbel_sample(
            filtered,
            temperature=temperature
        )

        out = torch.cat([out, next_tok], dim=-1)
        generated.append(next_tok)

        if eos_token is not None and (next_tok == eos_token).any():
            break

    gen = torch.cat(generated, dim=-1)
    return gen if exclude_prompt else torch.cat([input_ids, gen], dim=-1)
