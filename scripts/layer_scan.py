"""
Zero-shot layer-importance scan for the T5 backbone of a trained SpaMo checkpoint.

A block is "dropped" with a forward hook that returns its input hidden states
unchanged. Every T5 block only adds to the residual stream, so this is exactly
equivalent to deleting it (block 0 still provides the relative position bias).
No training: only the teacher-forced translation loss on the dev split is measured.

1. Single scan: drop each of the encoder/decoder blocks alone -> dev loss increase.
2. Cumulative: drop the k least important blocks together (encoder only, decoder
   only, both) -> dev loss, to see how the damage adds up.

Run from the repo root:
    python scripts/layer_scan.py --ckpt /path/to/spamo.ckpt -c configs/finetune.yaml --out scan.json
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from utils.helpers import instantiate_from_config


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def cache_inputs(model, dataloader, device):
    """Run the visual front-end once; T5 inputs are reused by every scan entry."""
    seed_all()
    cached = []
    for i, batch in enumerate(dataloader):
        batch = model.transfer_batch_to_device(batch, device, 0)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            inputs = model.get_inputs(batch)
            vo, vm = model.prepare_visual_inputs(inputs)
            vo = model.fusion_proj(vo)
            emb, msk, out_tok, targets = model.prepare_inputs(vo, vm, inputs, "val", i)
        cached.append((emb, msk, out_tok.attention_mask, targets))
    return cached


@torch.no_grad()
def dev_loss(model, cached) -> float:
    """Token-level mean cross-entropy over the whole split."""
    total, n_tokens = 0.0, 0
    for emb, msk, dec_msk, targets in cached:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.t5_model(
                inputs_embeds=emb, attention_mask=msk,
                decoder_attention_mask=dec_msk, labels=targets, return_dict=True,
            ).logits
        total += F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=-100, reduction="sum",
        ).item()
        n_tokens += (targets != -100).sum().item()
    return total / n_tokens


def skip_hook(module, args, output):
    return (args[0],) + tuple(output[1:])


def blocks_of(model):
    t5 = model.t5_model.get_base_model() if hasattr(model.t5_model, "get_base_model") else model.t5_model
    return {"encoder": list(t5.encoder.block), "decoder": list(t5.decoder.block)}


def with_dropped(model, cached, drop):
    """drop: list of (stack, idx). Returns the dev loss with those blocks skipped."""
    blocks = blocks_of(model)
    handles = [blocks[s][i].register_forward_hook(skip_hook) for s, i in drop]
    try:
        loss = dev_loss(model, cached)
    finally:
        for h in handles:
            h.remove()
    return loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("-c", "--config", default="configs/finetune.yaml")
    p.add_argument("--split", default="dev")
    p.add_argument("--out", required=True)
    p.add_argument("--max_k", type=int, default=12, help="largest number of blocks dropped together")
    args = p.parse_args()

    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    seed_all()
    model = instantiate_from_config(cfg.model)
    model.load_pretrained_weights(args.ckpt)
    model = model.to(device).eval()

    data_cfg = OmegaConf.to_container(cfg.data.params.test, resolve=True)
    data_cfg["params"]["mode"] = args.split
    ds = instantiate_from_config(data_cfg)
    dl = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False, collate_fn=ds.collate_fn)
    cached = cache_inputs(model, dl, device)
    print(f"Cached {len(ds)} {args.split} samples in {len(cached)} batches", flush=True)

    res = {"ckpt": args.ckpt, "config": args.config, "split": args.split}
    t0 = time.time()
    base = dev_loss(model, cached)
    res["baseline_loss"] = base
    print(f"baseline loss {base:.4f} (ppl {math.exp(base):.2f}) [{time.time() - t0:.0f}s]", flush=True)

    # 1. drop one block at a time
    single = {}
    for stack, blocks in blocks_of(model).items():
        for i in range(len(blocks)):
            loss = with_dropped(model, cached, [(stack, i)])
            single[f"{stack}.{i}"] = loss - base
            print(f"drop {stack:7s} block {i:2d}: loss {loss:.4f}  delta {loss - base:+.4f}", flush=True)
    res["single_delta"] = single
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    # 2. drop the k least important blocks together
    def ranked(stacks):
        keys = [k for k in single if k.split(".")[0] in stacks]
        return [(k.split(".")[0], int(k.split(".")[1])) for k in sorted(keys, key=lambda k: single[k])]

    cumulative = {}
    for name, stacks in [("encoder", ["encoder"]), ("decoder", ["decoder"]), ("both", ["encoder", "decoder"])]:
        order = ranked(stacks)
        cumulative[name] = []
        for k in range(2, args.max_k + 1, 2):
            drop = order[:k]
            loss = with_dropped(model, cached, drop)
            cumulative[name].append({"k": k, "dropped": [f"{s}.{i}" for s, i in drop], "loss": loss, "delta": loss - base})
            print(f"cumulative {name:7s} k={k:2d}: loss {loss:.4f}  delta {loss - base:+.4f}  "
                  f"dropped {[f'{s[0]}{i}' for s, i in drop]}", flush=True)
        res["cumulative"] = cumulative
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
    print(f"DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
