# tiny-thinker

I wanted to see if a small model can learn when to stop thinking.

This week Tencent and Xi'an Jiaotong dropped a paper called LoCo — a training
method that teaches a reasoning model to quit early, cutting its output length
by up to 45% without losing accuracy. The idea stuck with me because it's the
opposite of how most of us use these models: we usually pay for the longest,
most careful chain of thought and just eat the token bill. LoCo asks the
obvious question — how much of that thinking was actually needed?

Retraining a model LoCo-style was out of scope for a Friday afternoon, so I
built the inference-time cousin instead — with one training-time twist that
turned out to be the whole lesson. Train a tiny transformer to solve
multi-step arithmetic with a step-by-step scratchpad. Then, after every step,
ask it to answer *right now* and look at how sure it is. If it's sure enough,
stop. The remaining steps were never needed.

The twist: my first version bolted the confidence gate onto a model that had
only ever seen full traces, and the gate failed completely — the model was
confidently wrong, because answering early was off-distribution and it just
blurted out the intermediate result. The fix is the LoCo part: train on both
full traces *and* one-step-early answers (same problem, trace cut after the
second-to-last step, answer straight away — on a random 30% of problems, so
full traces still dominate). Now the model genuinely knows how to answer
early, and the confidence gate can tell when it knows vs when it's guessing.

The surprising part: the model is often sure much earlier than you'd guess.
Once the first multiplication is done, the rest is small add/subtract work,
and the model frequently knows the answer before it finishes showing it.
The less surprising part: when it's wrong early, it's *confidently* wrong,
which is exactly the failure mode you have to design around. The threshold
is doing all the work here.

## What it does

- `data.py` — generates synthetic multi-step arithmetic expressions
  (`7*8-6+3=`) with one-operation-per-line scratchpad traces. 8,000 train /
  200 held-out. First op is always a single-digit multiplication; the rest
  are + / - a single digit off the running total.
- `model.py` — a small decoder-only transformer, char-level, ~620K params.
  Nothing special, trains on a laptop CPU in about half an hour.
- `train.py` — plain next-char training on the traces. No RL, no special
  loss. The early-exit behaviour is added later, at inference time.
- `early_exit.py` — the actual point of the repo. Solves each held-out
  problem three ways and prints the tradeoff table:
  - **direct**: answer immediately, no steps
  - **gated**: after each step, probe the answer and stop when confidence
    clears a threshold τ
  - **full chain-of-thought**: all steps, then the answer
- `demo.py` — pick one problem and watch the confidence after each step.

## Results

120 held-out problems, greedy decoding, output measured in characters
(char-level model, so "tokens" here are really chars — the ratios are what
matter):

| method | accuracy | avg out chars | saved vs full |
|---|---|---|---|
| direct (0 steps) | 0.0% | 10 | +84% |
| gated, τ=0.5 | 67.5% | 47 | +26% |
| gated, τ=0.7 | 91.7% | 60 | +5% |
| gated, τ=0.9 | 98.3% | 63 | +0% |
| gated, τ=0.99 | 99.2% | 63 | +0% |
| full chain-of-thought | 99.2% | 63 | — |

Read the table from the bottom up, because the interesting bit is what
*doesn't* happen: at high thresholds the gate saves nothing. The model's
confidence after each intermediate step sits around 0.25–0.5 — it genuinely
doesn't know the answer yet — and only jumps to ~1.0 after the final step.
So τ=0.9 stops after the last step, which is no saving at all. The real
tradeoff lives at low thresholds: τ=0.5 cuts a quarter of the output but
accuracy falls to 67%, and τ=0.7 gets you 92% for 5% shorter.

Why the model can't answer early is the actual finding. In full traces,
"Answer:" always means "repeat the last step's result", so when you force
an early "Answer:" the model mostly just blurts out the intermediate result
— confidently wrong, not humbly uncertain. The 30% of training problems
that carry a one-step-early answer weren't enough to teach it the other
pattern, because the context is identical either way: after step n-1, the
model can't tell whether a full trace or an early answer is coming. I
verified this directly: at a mid-training checkpoint, the probe one step
before the end was right only 1/20 times. Early exit needs the model to
have internalized the skipped computation, and this toy hasn't — which is
exactly why LoCo does the compression at training time instead of bolting a
gate on afterwards. The gate works as a measurement tool; it just measures
that this model isn't ready to stop early.

One thing worth being honest about: this is LoCo miniaturized, not LoCo
itself. LoCo trains a frontier reasoning model to compress long chains of
thought; this trains a 600K-param toy to skip its last arithmetic step. The
gate works here because the task is small and the confidence signal is clean.
On a real reasoning model the calibration is messier, and "confident" stops
meaning "right" much faster. Still, the tradeoff curve is real, and it's the
cheapest way I know to build intuition for adaptive compute — including the
failure mode that matters most, which is that an untrained early answer is
confidently wrong, not humbly uncertain.

## What didn't work: word problems

The first version of this repo used word problems — "A baker makes 12 trays
with 8 muffins each..." After 2,000 training steps the model still couldn't
do it: step-1 operands exact on 0/40 held-out problems, and it wrote things
like `154-2=136`. Pulling numbers out of a 120-char story and doing
multi-digit arithmetic in your head is a lot to ask of a 600K-param char
model, and the loss (0.20) hid it completely — the format chars are 99%
predictable, so the loss looks great while the digits are wrong. I switched
to bare expressions with a scratchpad, where each step's inputs sit right
next to it, and the model actually learned the task. The failure is the
useful part: if your eval is free-running accuracy and your loss is
dominated by easy tokens, you will fool yourself.

## How to run it

```bash
pip install -r requirements.txt

python data.py        # generates train.txt + test.json
python train.py       # ~25 min on a laptop CPU, saves weights.pt
python early_exit.py  # the tradeoff table (default: 120 held-out problems)
python demo.py 3      # watch one problem, step by step
```

`weights.pt` isn't in the repo — train it yourself, it's the fun part.

## Rough edges

- Char-level, so "tokens" in the table are really output characters. The
  ratios are what matter, not the absolute numbers.
- The probe re-scores the answer with one extra forward pass per step. On
  this toy that's nothing; on a real model you'd want the confidence from
  the generation pass itself.
- Greedy decoding everywhere. Sampling would change the numbers a bit and
  probably make the calibration story worse, which is itself interesting.
- The expressions are templated, so the model may be pattern-matching the
  shapes as much as doing arithmetic. A held-out operator pattern would be
  the honest next test.

## If I had more time

- The real LoCo setup: fine-tune on shortened traces and compare against
  this gate — training-time vs inference-time early exit, head to head.
- Calibration plots: confidence vs correctness, binned, so you can see
  exactly where "sure" stops meaning "right".
- A word-problem version that actually works — probably needs a bigger
  model and a lot more training, or a structured "Numbers:" line the model
  can copy from.
