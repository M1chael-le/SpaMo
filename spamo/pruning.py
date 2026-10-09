"""
Structured pruning of T5 attention heads and FFN neurons.

Importance is estimated with scalar gates on every head / FFN neuron
(Michel et al., 2019): I = |dL/dgate| accumulated over calibration batches.
Pruning physically removes rows/columns from q/k/v/o and wi(_0/_1)/wo.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import prune_linear_layer


def _base_model(t5: nn.Module) -> nn.Module:
    """Unwrap a PEFT model to the underlying T5ForConditionalGeneration."""
    return t5.get_base_model() if hasattr(t5, "get_base_model") else t5


def _is_attention(module: nn.Module) -> bool:
    return all(hasattr(module, a) for a in ("q", "k", "v", "o", "n_heads", "prune_heads"))


def _is_ffn(module: nn.Module) -> bool:
    return hasattr(module, "wo") and (hasattr(module, "wi") or hasattr(module, "wi_0"))


def collect_prunable(t5: nn.Module) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Module]]:
    """Return ({name: attention}, {name: ffn}) for every encoder/decoder block."""
    heads, ffns = {}, {}
    for name, module in _base_model(t5).named_modules():
        if not (name.startswith("encoder.block") or name.startswith("decoder.block")):
            continue
        if _is_attention(module):
            heads[name] = module
        elif _is_ffn(module):
            ffns[name] = module
    return heads, ffns


def _ffn_size(ffn: nn.Module) -> int:
    return ffn.wo.in_features


def translation_loss(model, batch, batch_idx: int) -> torch.Tensor:
    """
    T5 cross-entropy loss for a batch. Not via shared_step, which skips the T5
    forward during the contrastive warm-up (global_step is 0 outside a trainer).
    """
    inputs = model.get_inputs(batch)
    visual_outputs, visual_masks = model.prepare_visual_inputs(inputs)
    visual_outputs = model.fusion_proj(visual_outputs)
    input_embeds, input_masks, output_tokens, targets = model.prepare_inputs(
        visual_outputs, visual_masks, inputs, "train", batch_idx
    )
    return model.t5_model(
        inputs_embeds=input_embeds,
        attention_mask=input_masks,
        decoder_attention_mask=output_tokens.attention_mask,
        labels=targets,
        return_dict=True,
    ).loss


def compute_importance(model, dataloader, num_batches: int) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Accumulate |dL/dgate| for every attention head and FFN neuron of model.t5_model.

    Args:
        model: FlanT5SLT instance (unpruned)
        dataloader: training dataloader
        num_batches: number of calibration batches

    Returns:
        {"heads": {name: scores[n_heads]}, "ffn": {name: scores[d_ff]}}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    was_training = model.training
    model.eval()

    heads, ffns = collect_prunable(model.t5_model)
    gates, scores, handles = {}, {}, []

    def head_hook(gate, n_heads):
        def hook(module, args):
            x = args[0]
            b, t, _ = x.shape
            x = x.view(b, t, n_heads, -1) * gate.view(1, 1, n_heads, 1).to(x.dtype)
            return (x.view(b, t, -1),)
        return hook

    def ffn_hook(gate):
        def hook(module, args):
            return (args[0] * gate.to(args[0].dtype),)
        return hook

    for name, attn in heads.items():
        gates[name] = torch.ones(attn.n_heads, device=device, requires_grad=True)
        scores[name] = torch.zeros(attn.n_heads, device=device)
        handles.append(attn.o.register_forward_pre_hook(head_hook(gates[name], attn.n_heads)))
    for name, ffn in ffns.items():
        gates[name] = torch.ones(_ffn_size(ffn), device=device, requires_grad=True)
        scores[name] = torch.zeros(_ffn_size(ffn), device=device)
        handles.append(ffn.wo.register_forward_pre_hook(ffn_hook(gates[name])))

    try:
        for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Pruning importance")):
            if i >= num_batches:
                break
            batch = model.transfer_batch_to_device(batch, device, 0)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = translation_loss(model, batch, i)
            loss.backward()
            for name, gate in gates.items():
                scores[name] += gate.grad.abs().float()
                gate.grad = None
            model.zero_grad(set_to_none=True)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    return {
        "heads": {n: scores[n].cpu() for n in heads},
        "ffn": {n: scores[n].cpu() for n in ffns},
    }


def _select(unit_scores: Dict[str, torch.Tensor], ratio: float, scope: str) -> Dict[str, List[int]]:
    """Pick indices to prune in each unit; every unit keeps at least one entry."""
    pruned = {name: [] for name in unit_scores}
    if ratio <= 0:
        return pruned

    if scope == "per_layer":
        for name, s in unit_scores.items():
            k = min(int(round(ratio * s.numel())), s.numel() - 1)
            pruned[name] = sorted(torch.argsort(s)[:k].tolist())
    elif scope == "global":
        # Per-unit L2 normalisation makes scores comparable across layers (Michel et al.)
        entries = []
        for name, s in unit_scores.items():
            s = s / (s.norm() + 1e-12)
            entries.extend((v, name, idx) for idx, v in enumerate(s.tolist()))
        entries.sort(key=lambda e: e[0])
        budget = int(round(ratio * len(entries)))
        remaining = {name: s.numel() for name, s in unit_scores.items()}
        for _, name, idx in entries:
            if budget == 0:
                break
            if remaining[name] > 1:
                pruned[name].append(idx)
                remaining[name] -= 1
                budget -= 1
        pruned = {name: sorted(idx) for name, idx in pruned.items()}
    else:
        raise ValueError(f"Unknown pruning scope: {scope}")
    return pruned


def select_to_prune(scores: Dict, head_ratio: float, ffn_ratio: float, scope: str = "per_layer") -> Dict:
    """Build a prune spec {"heads": {name: [idx]}, "ffn": {name: [idx]}} from importance scores."""
    return {
        "heads": _select(scores["heads"], head_ratio, scope),
        "ffn": _select(scores["ffn"], ffn_ratio, scope),
    }


def _prune_linear(layer: nn.Linear, keep: torch.Tensor, dim: int) -> nn.Linear:
    # prune_linear_layer always returns fp32; keep the original dtype
    return prune_linear_layer(layer, keep, dim=dim).to(layer.weight.dtype)


def _cross_attention_bias_hook(full_n_heads: int):
    """
    HF T5 builds the (all-zero + mask) cross-attention position bias in the first decoder
    layer with that layer's *pruned* head count and shares it with all later layers, which
    then index it with their own original head ids. Build it with the original head count
    instead so each layer's `pruned_heads` masking selects the right heads.
    """
    def hook(module, args, kwargs):
        if kwargs.get("position_bias") is not None:
            return args, kwargs
        hidden = args[0] if args else kwargs["hidden_states"]
        kv = kwargs.get("key_value_states")
        key_len = kv.shape[1] if kv is not None else hidden.shape[1]
        mask = kwargs.get("mask")
        dtype = mask.dtype if mask is not None else hidden.dtype
        bias = torch.zeros((1, full_n_heads, hidden.shape[1], key_len), device=hidden.device, dtype=dtype)
        if mask is not None:
            bias = bias + mask[:, :, :, :key_len]
        kwargs["position_bias"] = bias
        return args, kwargs
    return hook


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def apply_prune_spec(t5: nn.Module, spec: Dict) -> None:
    """Physically remove the heads / FFN neurons listed in spec. `t5` must not be LoRA-wrapped."""
    heads, ffns = collect_prunable(t5)
    full_n_heads = t5.config.num_heads

    for name, idx in spec.get("heads", {}).items():
        if not idx:
            continue
        attn = heads[name]
        dtypes = {p: getattr(attn, p).weight.dtype for p in ("q", "k", "v", "o")}
        attn.prune_heads(set(idx))
        for p, dt in dtypes.items():
            setattr(attn, p, getattr(attn, p).to(dt))

    if any(spec.get("heads", {}).values()):
        for attn in heads.values():
            if not attn.has_relative_attention_bias and not getattr(attn, "_prune_bias_hook", False):
                attn.register_forward_pre_hook(_cross_attention_bias_hook(full_n_heads), with_kwargs=True)
                attn._prune_bias_hook = True

    for name, idx in spec.get("ffn", {}).items():
        if not idx:
            continue
        ffn = ffns[name]
        size = _ffn_size(ffn)
        drop = set(idx)
        keep = torch.tensor([i for i in range(size) if i not in drop], dtype=torch.long)
        for w in ("wi", "wi_0", "wi_1"):
            if hasattr(ffn, w):
                setattr(ffn, w, _prune_linear(getattr(ffn, w), keep, dim=0))
        ffn.wo = _prune_linear(ffn.wo, keep, dim=1)


def spec_summary(spec: Dict) -> str:
    n_heads = sum(len(v) for v in spec.get("heads", {}).values())
    n_ffn = sum(len(v) for v in spec.get("ffn", {}).values())
    return f"pruned {n_heads} attention heads, {n_ffn} FFN neurons"
