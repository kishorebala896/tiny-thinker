"""Multi-step arithmetic with a scratchpad — the task tiny-thinker actually
trains on.

First attempt was word problems ("A baker makes 12 trays with 8 muffins
each..."). The toy model never learned it: after 2000 steps it still couldn't
copy the operands out of the story (step-1 operands exact: 0/40) and its
two-digit subtraction was unreliable (it wrote 154-2=136). The failure is
instructive — pulling numbers out of a 120-char story and doing multi-digit
arithmetic in your head is a lot to ask of a 600K-param char model — but it
isn't the point of this repo, so I switched to the known-working setup:
bare expressions with a step-by-step scratchpad. Each step's inputs are
either in the short expression right above it or the previous line's result,
so the model can actually learn the task, and then the interesting question
— when can it stop early? — has a real answer.
"""
import json
import random


def make_expression(rng):
    """A 2-4 op expression. First op is single-digit multiplication (the
    64 times-table combos are memorizable); the rest are + / - a single
    digit off the running total. All intermediates stay in [0, 99]."""
    n_ops = rng.choices([2, 3, 4], weights=[4, 4, 2])[0]
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    result = a * b
    steps = [f"{a}*{b}={result}"]
    expr = f"{a}*{b}"
    for _ in range(n_ops - 1):
        op = rng.choice("+-")
        c = rng.randint(2, 9)
        nxt = result + c if op == "+" else result - c
        if nxt < 0 or nxt > 99:
            return None
        steps.append(f"{result}{op}{c}={nxt}")
        expr += f"{op}{c}"
        result = nxt
    return expr, steps, result


def render(expr, steps, answer):
    lines = [f"Q: {expr}="]
    for i, s in enumerate(steps, 1):
        lines.append(f"Step {i}: {s}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def render_early(expr, steps, answer, k):
    """The LoCo-style training example: the trace cut after step k, then the
    answer straight away. This is what teaches the model that it *can*
    answer early — without it, asking for an early answer is off-distribution
    and the model just blurts out the intermediate result, confidently
    wrong. k is the last step kept; we use k = n-1, the one-step-early
    case, where only a single + / - remains to do in your head."""
    lines = [f"Q: {expr}="]
    for i, s in enumerate(steps[:k], 1):
        lines.append(f"Step {i}: {s}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def build_dataset(n_train=8000, n_test=200, seed=0):
    rng = random.Random(seed)
    train, seen = [], set()
    while len(train) < n_train:
        made = make_expression(rng)
        if made is None:
            continue
        expr, steps, ans = made
        if expr in seen:
            continue
        seen.add(expr)
        train.append({"question": expr, "steps": steps, "answer": ans})
    test = []
    while len(test) < n_test:
        made = make_expression(rng)
        if made is None:
            continue
        expr, steps, ans = made
        if expr in seen:
            continue
        seen.add(expr)
        test.append({"question": expr, "steps": steps, "answer": ans})
    return train, test


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default="train.txt")
    ap.add_argument("--test-out", default="test.json")
    a = ap.parse_args()
    train, test = build_dataset()
    # early-answer examples on a random 30% of problems: enough to teach the
    # model it *can* answer one step early, rare enough that full traces
    # still dominate greedy generation (so "full" stays full).
    pick = random.Random(1)
    with open(a.train_out, "w") as f:
        for ex in train:
            f.write(render(ex["question"], ex["steps"], ex["answer"]) + "\n\n")
            if len(ex["steps"]) >= 2 and pick.random() < 0.30:
                f.write(render_early(ex["question"], ex["steps"],
                                     ex["answer"], len(ex["steps"]) - 1) + "\n\n")
    with open(a.test_out, "w") as f:
        json.dump(test, f, indent=1)
    import collections
    print("train:", len(train), "test:", len(test))
    print("step-count dist:", collections.Counter(len(e["steps"]) for e in train))
    print("--- sample ---")
    print(render(train[0]["question"], train[0]["steps"], train[0]["answer"]))
    print("--- sample ---")
    print(render(train[1]["question"], train[1]["steps"], train[1]["answer"]))
