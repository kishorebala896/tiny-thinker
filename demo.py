"""Show one problem being solved, step by step, with the model's confidence
after each step — so you can watch it decide it already knows the answer.

Usage:
    python demo.py                # a random held-out problem
    python demo.py 3              # test problem #3
    python demo.py --tau 0.9      # stop as soon as confidence hits 0.9
"""
import argparse
import json
import random

from early_exit import load, solve_traced, answer_at_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?", type=int, default=None)
    ap.add_argument("--tau", type=float, default=0.9)
    args = ap.parse_args()

    model, stoi, itos = load()
    test = json.load(open("test.json"))
    ex = test[args.index] if args.index is not None \
        else random.choice(test)

    print(f"Q: {ex['question']}=\n")
    tr = solve_traced(model, ex["question"], stoi, itos)
    ans0, p0 = tr["probes"][0]
    print(f"  direct guess (no thinking): {ans0}  confidence {p0:.2f}")
    for k, step in enumerate(tr["steps"], 1):
        ans, p = tr["probes"][k]
        print(f"  {step}")
        print(f"    -> could answer now: {ans}  confidence {p:.2f}")
    print(f"\n  full answer: {tr['full_answer']}  "
          f"(true answer: {ex['answer']})")
    gated, spent = answer_at_tau(tr, args.tau)
    print(f"  gated answer at tau={args.tau}: {gated}  "
          f"({spent} chars vs {tr['chars']} full — "
          f"{100 * (1 - spent / tr['chars']):.0f}% shorter)")


if __name__ == "__main__":
    main()
