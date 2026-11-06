"""Confidence-gated early exit — the LoCo idea, done at inference time.

LoCo (arXiv, Nov 3 2026) trains a reasoning model to stop thinking early,
cutting output length up to 45% with accuracy intact. Retraining a model is
out of scope for a Friday afternoon, so this is the inference-time cousin:
after every reasoning step, ask the model to answer *right now* and look at
how sure it is. If it's sure enough, stop — the remaining steps were never
needed. If it's not, keep thinking.

One traced solve per problem records the probe answer + confidence after
every step, so every threshold tau can be evaluated from the same trace.
"""
import json
import math

import torch

from model import CharTransformer

torch.set_num_threads(4)


def load(path="weights.pt", vocab="vocab.json"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    v = json.load(open(vocab))
    stoi = {k: int(i) for k, i in v["stoi"].items()}
    itos = {int(k): c for k, c in v["itos"].items()}
    model = CharTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, stoi, itos


def enc(s, stoi):
    return [stoi[c] for c in s]


def dec(ids, itos):
    return "".join(itos[i] for i in ids)


def gen_line(model, ctx, stoi, itos, max_chars=48):
    """Greedy-generate one line (until newline). Returns the line string."""
    ids = torch.tensor([ctx], dtype=torch.long)
    out, _ = model.generate(ids, max_chars, newline_id=stoi["\n"])
    return dec(out[0].tolist(), itos).rstrip("\n")


def probe_answer(model, ctx, stoi, itos, max_chars=6):
    """Append 'Answer: ' and greedily read the digits + their confidence.

    Returns (answer_str or None, prob) where prob is the geometric mean of
    the per-digit token probabilities — i.e. how sure the model is about
    this answer, right now, with no more thinking.
    """
    ctx = ctx + enc("Answer: ", stoi)
    ids = torch.tensor([ctx], dtype=torch.long)
    out, _ = model.generate(ids, max_chars, newline_id=stoi["\n"],
                            return_logprobs=False)
    text = dec(out[0].tolist(), itos).rstrip("\n")
    if not text or not text.isdigit():
        return None, 0.0
    # re-score the digit tokens properly (generate() already consumed them,
    # so do one clean forward pass to get their logprobs)
    full = torch.tensor([ctx + enc(text, stoi)], dtype=torch.long)
    full = full[:, -model.block_size:]
    with torch.no_grad():
        logits = model(full)
    L = full.shape[1]
    logp = 0.0
    for j, ch in enumerate(text):
        pos = L - len(text) + j - 1  # logits at pos predict token pos+1
        p = torch.softmax(logits[0, pos], dim=-1)
        logp += math.log(p[stoi[ch]].item() + 1e-12)
    return text, math.exp(logp / len(text))


def solve_traced(model, question, stoi, itos, max_steps=6):
    """Full solve, recording a probe after step 0 (direct answer) and after
    every step. Returns a dict with everything the tau sweep needs."""
    ctx = enc(f"Q: {question}=\n", stoi)
    probes = []  # (answer, prob) after k steps, k = 0..n
    steps = []
    chars = 0
    ans, prob = probe_answer(model, ctx, stoi, itos)
    probes.append((ans, prob))
    full_answer = None
    for k in range(1, max_steps + 1):
        line = gen_line(model, ctx, stoi, itos)
        chars += len(line) + 1
        if line.startswith("Answer:"):
            try:
                full_answer = int(line.split(":")[1].strip())
            except ValueError:
                full_answer = None
            break
        if not line.startswith("Step"):
            break  # model went off-script; bail out
        steps.append(line)
        ctx = ctx + enc(line + "\n", stoi)
        ans, prob = probe_answer(model, ctx, stoi, itos)
        probes.append((ans, prob))
    else:
        pass
    # chars also include the final answer line if we got one — note the loop
    # above already counted it when it was generated, so don't add it again.
    return {"steps": steps, "probes": probes, "full_answer": full_answer,
            "chars": chars, "n_steps": len(steps)}


def answer_at_tau(trace, tau):
    """First probe at/above tau wins; otherwise fall back to the full answer."""
    chars = 0
    for k, (ans, prob) in enumerate(trace["probes"]):
        if ans is not None and prob >= tau:
            # chars spent: steps 1..k + the answer itself
            spent = sum(len(s) + 1 for s in trace["steps"][:k])
            spent += len(f"Answer: {ans}") + 1
            return int(ans), spent
    return trace["full_answer"], trace["chars"]


def evaluate(model, stoi, itos, test, taus=(0.5, 0.7, 0.9, 0.99)):
    traces = []
    for i, ex in enumerate(test):
        traces.append(solve_traced(model, ex["question"], stoi, itos))
        if (i + 1) % 20 == 0:
            print(f"  solved {i + 1}/{len(test)}", flush=True)

    def acc(fn):
        ok, tot_chars = 0, 0
        for tr, ex in zip(traces, test):
            a, c = fn(tr)
            if a is not None and a == ex["answer"]:
                ok += 1
            tot_chars += c
        return ok / len(test), tot_chars / len(test)

    rows = []
    direct_acc, direct_chars = acc(lambda tr: (int(tr["probes"][0][0])
                                               if tr["probes"][0][0] else None,
                                               len(f"Answer: {tr['probes'][0][0] or ''}") + 1))
    rows.append(("direct (0 steps)", direct_acc, direct_chars, None))
    for tau in taus:
        a, c = acc(lambda tr, t=tau: answer_at_tau(tr, t))
        rows.append((f"gated tau={tau}", a, c, None))
    full_acc, full_chars = acc(lambda tr: (tr["full_answer"], tr["chars"]))
    rows.append(("full chain-of-thought", full_acc, full_chars, None))

    print(f"\n{'method':<24} {'accuracy':>9} {'avg out chars':>14} "
          f"{'saved vs full':>14}")
    for name, a, c, _ in rows:
        saved = "" if name == "full chain-of-thought" else \
            f"{100 * (1 - c / full_chars):+.0f}%"
        print(f"{name:<24} {a:>8.1%} {c:>14.0f} {saved:>14}")
    # step histogram for the middle threshold, just to see it working
    tau = 0.9
    import collections
    used = collections.Counter()
    for tr in traces:
        for k, (ans, prob) in enumerate(tr["probes"]):
            if ans is not None and prob >= tau:
                used[k] += 1
                break
        else:
            used[tr["n_steps"]] += 1
    print(f"\nsteps actually used at tau={tau} (0 = answered immediately): "
          f"{dict(sorted(used.items()))}")
    return traces


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    model, stoi, itos = load()
    test = json.load(open("test.json"))[:n]
    print(f"evaluating on {n} held-out problems...")
    evaluate(model, stoi, itos, test)
