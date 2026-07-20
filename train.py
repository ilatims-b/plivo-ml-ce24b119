"""Baseline trainer. It WORKS and it is MEDIOCRE ON PURPOSE. Your hour goes
into changing what it does — schedule, init, optimizer, architecture,
tokenizer — inside the hard caps.

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import json
import math
import os
import time

import torch
import ngram_metrics

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum_steps", type=int, default=1, help="Number of micro-batches to accumulate per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--tie_weights", action="store_true", help="Tie input and output embedding weights")
    ap.add_argument("--n_layer", type=int, default=None, help="Override number of transformer layers")
    ap.add_argument("--n_embd", type=int, default=None, help="Override hidden embedding dimension")
    ap.add_argument("--num_loops", type=int, default=None, help="Override recurrent loop iterations")
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    # Load dev tokens for tracking n-gram phase transitions across training
    dev_path = os.path.join(os.path.dirname(__file__), "../data/dev_eval.txt")
    dev_text = open(dev_path, encoding="utf-8").read() if os.path.exists(dev_path) else ""
    dev_tokens = tok.encode(dev_text) if dev_text else []

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    if args.tie_weights:
        cfg.tie_weights = True
    if args.n_layer is not None:
        cfg.n_layer = args.n_layer
    if args.n_embd is not None:
        cfg.n_embd = args.n_embd
    if args.num_loops is not None:
        cfg.num_loops = args.num_loops

    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def get_lr(step, total_steps, max_lr):
        warmup_steps = min(100, max(1, total_steps // 10))
        min_lr = max_lr * 0.01  # Deep cosine decay down to 1e-5 for stable final convergence
        if step <= warmup_steps:
            return max_lr * step / warmup_steps
        decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    model.train()
    t0 = time.time()
    losses = []
    ngram_curves = {}
    for step in range(1, args.steps + 1):
        lr = get_lr(step, args.steps, args.lr)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro_step in range(args.accum_steps):
            x, y = get_batch(ids, cfg.block_size, args.batch, device)
            _, loss = model(x, y)
            loss_scaled = loss / args.accum_steps
            loss_scaled.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        avg_loss = step_loss / args.accum_steps
        losses.append(avg_loss)
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  ({(time.time()-t0)*1000/step:.0f} ms/step)")

        # Track n-gram metrics across training phase transitions
        if dev_tokens and (step % 200 == 0 or step == 1 or step == args.steps):
            t_eval = time.time()
            m_metrics = ngram_metrics.evaluate_model_ngrams(model, dev_tokens, cfg.block_size, device)
            ngram_curves[step] = m_metrics
            model.train()

    # every public config attribute is saved — if you add fields to Config,
    # they ride along automatically and evaluate.py rebuilds the same model
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses,
                "ngram_curves": ngram_curves}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")

    # Retain all runs in run_history.json
    history_path = os.path.join(os.path.dirname(__file__), "run_history.json")
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}
    run_label = args.run_name if args.run_name else f"L{cfg.n_layer}_E{cfg.n_embd}_B{cfg.block_size}_tie{cfg.tie_weights}"
    history[run_label] = {
        "losses": losses,
        "ngram_curves": ngram_curves,
        "config": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))}
    }
    with open(history_path, "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    main()
