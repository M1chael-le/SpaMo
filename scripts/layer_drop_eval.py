"""
Zero-shot BLEU + generation-time check for dropping whole T5 blocks (no recovery).

Blocks are physically removed (spamo.pruning.drop_blocks), so the timing reflects
the smaller model. Which blocks to drop comes from a layer_scan.py result: the
k blocks per stack whose individual removal hurt the dev loss least (block 0 is
always kept). Decoding is deterministic beam search (num_beams=5, no sampling).

Run from the repo root:
    python scripts/layer_drop_eval.py --ckpt /path/to/spamo.ckpt -c configs/finetune.yaml \
        --scan phoenix_dev.json --grid encoder --out phoenix_drop_encoder.json
"""
import argparse
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from utils.helpers import instantiate_from_config
from utils.evaluate import evaluate_results
from spamo.pruning import _base_model, count_params, drop_blocks, least_important_blocks, set_blocks
from scripts.layer_scan import cache_inputs, seed_all

# group -> {name: number of blocks to drop from (encoder, decoder)}; every group has
# its own baseline so timings within a group come from the same run
GRIDS = {
    "encoder": {"baseline": (0, 0), "enc2": (2, 0), "enc4": (4, 0), "enc6": (6, 0), "enc8": (8, 0)},
    "decoder": {"baseline": (0, 0), "dec1": (0, 1), "dec2": (0, 2), "dec3": (0, 3), "dec4": (0, 4)},
    "both": {"baseline": (0, 0), "enc4_dec1": (4, 1), "enc4_dec2": (4, 2),
             "enc8_dec1": (8, 1), "enc8_dec2": (8, 2), "enc8_dec4": (8, 4)},
}


@torch.no_grad()
def generate_all(model, cached, num_beams: int):
    tok = model.t5_tokenizer
    hyps, refs, gen_time, n_gen_tokens = [], [], 0.0, 0
    for emb, msk, _, targets in cached:
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.t5_model.generate(
                inputs_embeds=emb, attention_mask=msk, num_beams=num_beams,
                max_length=model.max_txt_len, do_sample=False,
            )
        torch.cuda.synchronize()
        gen_time += time.time() - t0
        n_gen_tokens += (out != tok.pad_token_id).sum().item()
        hyps += [s.lower() for s in tok.batch_decode(out, skip_special_tokens=True)]
        refs += [s.lower() for s in tok.batch_decode(
            targets.masked_fill(targets == -100, tok.pad_token_id), skip_special_tokens=True)]
    return hyps, refs, gen_time, n_gen_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--scan", required=True, help="layer_scan.py output for the same checkpoint")
    p.add_argument("--split", default="dev")
    p.add_argument("--out", required=True)
    p.add_argument("--grid", choices=list(GRIDS), required=True, help="drop encoder blocks, decoder blocks, or both")
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--bleu_tokenizer", default=None, help="sacrebleu tokenizer; default zh for CSL configs, else 13a")
    args = p.parse_args()
    bleu_tok = args.bleu_tokenizer or ("zh" if "csl" in os.path.basename(args.config) else "13a")

    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    seed_all()
    model = instantiate_from_config(cfg.model)
    model.load_pretrained_weights(args.ckpt)
    model = model.to(device).eval()
    t5 = _base_model(model.t5_model)
    original = {s: list(getattr(t5, s).block) for s in ("encoder", "decoder")}

    data_cfg = OmegaConf.to_container(cfg.data.params.test, resolve=True)
    data_cfg["params"]["mode"] = args.split
    ds = instantiate_from_config(data_cfg)
    dl = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False, collate_fn=ds.collate_fn)
    cached = cache_inputs(model, dl, device)
    print(f"Cached {len(ds)} {args.split} samples; BLEU tokenizer {bleu_tok}", flush=True)

    with open(args.scan) as fh:
        scan = json.load(fh)
    generate_all(model, cached[:2], args.num_beams)  # warm-up (CUDA init, kernel autotuning)

    out_dir = os.path.splitext(args.out)[0]
    os.makedirs(out_dir, exist_ok=True)
    results = {"ckpt": args.ckpt, "config": args.config, "split": args.split, "grid": args.grid,
               "num_beams": args.num_beams, "settings": {}}
    for name, (k_enc, k_dec) in GRIDS[args.grid].items():
        spec = {"encoder": least_important_blocks(scan, "encoder", k_enc),
                "decoder": least_important_blocks(scan, "decoder", k_dec)}
        drop_blocks(t5, spec)
        try:
            hyps, refs, gen_time, n_tok = generate_all(model, cached, args.num_beams)
            metrics = evaluate_results(hyps, refs, split="test", tokenizer=bleu_tok)
            entry = {
                "dropped": spec,
                "encoder_blocks": len(t5.encoder.block),
                "decoder_blocks": len(t5.decoder.block),
                "t5_params_M": round(count_params(t5) / 1e6, 1),
                **{k.split("/")[1]: round(float(v), 4) for k, v in metrics.items()},
                "generate_s": round(gen_time, 1),
                "generated_tokens": n_tok,
                "ms_per_token": round(1000 * gen_time / max(n_tok, 1), 3),
            }
        finally:
            for s in ("encoder", "decoder"):
                set_blocks(t5, s, original[s])
        results["settings"][name] = entry
        print(f"{name:10s} enc={entry['encoder_blocks']:2d} dec={entry['decoder_blocks']:2d}  "
              f"BLEU-4 {entry['bleu4']:6.2f}  gen {entry['generate_s']:6.1f}s  tokens {n_tok:6d}  "
              f"{entry['ms_per_token']:.2f} ms/token  dropped {spec}", flush=True)
        with open(os.path.join(out_dir, f"{name}.txt"), "w") as fh:
            for r, h in zip(refs, hyps):
                fh.write(f"REF: {r}\nHYP: {h}\n\n")
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
