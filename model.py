"""A small decoder-only transformer, char-level. Nothing fancy — the same
shape as every tutorial transformer, kept small enough to train on a laptop
in a few minutes. The interesting part isn't the model, it's what we do
with it at inference time (see early_exit.py)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_layer=4, n_head=4,
                 block_size=256, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        x = self.drop(x)
        # causal mask, built once per length — cheap at this size
        mask = torch.triu(torch.ones(T, T, device=idx.device,
                                     dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx, max_new, newline_id, stop_at_newline=True,
                 return_logprobs=False):
        """Greedy decode. Returns (tokens, mean_logprob_of_generated)."""
        out = idx.clone()
        logps = []
        for _ in range(max_new):
            ctx = out[:, -self.block_size:]
            logits = self(ctx)
            nxt_logits = logits[:, -1, :]
            probs = F.softmax(nxt_logits, dim=-1)
            nxt = torch.argmax(probs, dim=-1, keepdim=True)
            logps.append(torch.log(probs[0, nxt.item()] + 1e-12).item())
            out = torch.cat([out, nxt], dim=1)
            if stop_at_newline and nxt.item() == newline_id:
                break
        lp = sum(logps) / max(len(logps), 1) if return_logprobs else None
        return out[:, idx.shape[1]:], lp

