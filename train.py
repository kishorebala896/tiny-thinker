"""Train the little reasoner. Plain next-char prediction on the traces from
data.py — no RL, no special loss. The early-exit behaviour is added later,
at inference time, which is the whole point: you don't need to retrain a
model to make it think less, you just need a decent confidence signal.
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from model import CharTransformer

torch.set_num_threads(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train.txt")
    ap.add_argument("--out", default="weights.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    text = open(args.data).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    json.dump({"stoi": stoi, "itos": itos},
              open("vocab.json", "w"))
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"chars: {len(text):,}  vocab: {len(chars)}  "
          f"params target ~1M")

    model = CharTransformer(len(chars), d_model=args.d_model,
                            n_layer=args.layers, block_size=args.block)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)

    def get_batch():
        ix = torch.randint(0, len(data) - args.block - 1, (args.batch,))
        x = torch.stack([data[i:i + args.block] for i in ix])
        y = torch.stack([data[i + 1:i + args.block + 1] for i in ix])
        return x, y

    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            torch.save({"state": model.state_dict(), "config": {
                "vocab_size": len(chars), "d_model": args.d_model,
                "n_layer": args.layers, "block_size": args.block}},
                args.out + ".ckpt")
        if step % 100 == 0 or step == 1:
            print(f"step {step}/{args.steps}  loss {loss.item():.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    torch.save({"state": model.state_dict(), "config": {
        "vocab_size": len(chars), "d_model": args.d_model,
        "n_layer": args.layers, "block_size": args.block}}, args.out)
    print(f"saved {args.out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
