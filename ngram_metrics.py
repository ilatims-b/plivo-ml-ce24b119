"""Compute empirical Unigram, Bigram, and Trigram baseline entropies/metrics
and compare them against model prediction accuracy across context lengths.
Pure PyTorch + stdlib.
"""
import argparse
import math
import os
from collections import defaultdict

import torch
import tokenizer as tokenizer_mod
import model as model_mod


def compute_empirical_ngrams(tokens, vocab_size):
    """Compute empirical unigram, bigram, and trigram entropies (in nats and bits per token)."""
    # Unigram
    unigram_counts = defaultdict(int)
    for t in tokens:
        unigram_counts[t] += 1
    total_tokens = len(tokens)
    unigram_probs = {t: c / total_tokens for t, c in unigram_counts.items()}
    unigram_entropy_nats = -sum(p * math.log(p) for p in unigram_probs.values() if p > 0)

    # Bigram
    bigram_counts = defaultdict(int)
    for i in range(len(tokens) - 1):
        bigram_counts[(tokens[i], tokens[i + 1])] += 1
    bigram_entropy_nats = 0.0
    for (t1, t2), c in bigram_counts.items():
        p_t1_t2 = c / (total_tokens - 1)
        p_t2_given_t1 = c / unigram_counts[t1]
        bigram_entropy_nats -= p_t1_t2 * math.log(p_t2_given_t1)

    # Trigram
    trigram_counts = defaultdict(int)
    for i in range(len(tokens) - 2):
        trigram_counts[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1
    trigram_entropy_nats = 0.0
    for (t1, t2, t3), c in trigram_counts.items():
        p_trigram = c / (total_tokens - 2)
        p_t3_given_t1_t2 = c / bigram_counts[(t1, t2)]
        trigram_entropy_nats -= p_trigram * math.log(p_t3_given_t1_t2)

    return {
        "unigram_bpt": unigram_entropy_nats / math.log(2),
        "bigram_bpt": bigram_entropy_nats / math.log(2),
        "trigram_bpt": trigram_entropy_nats / math.log(2),
        "unigram_counts": unigram_counts,
        "bigram_counts": bigram_counts,
        "trigram_counts": trigram_counts
    }


def evaluate_model_ngrams(model, tokens, block_size, device="cpu"):
    """Evaluate model cross-entropy specifically at context lengths 0 (unigram), 1 (bigram), 2 (trigram), and >=3."""
    model.eval()
    losses_by_ctx = defaultdict(list)
    correct_by_ctx = defaultdict(int)
    total_by_ctx = defaultdict(int)

    # Slide window across tokens
    stride = block_size // 2
    for start in range(0, len(tokens) - 1, stride):
        end = min(start + block_size, len(tokens))
        chunk = tokens[start:end]
        if len(chunk) < 2:
            break

        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device)[None, :]
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device)[None, :]

        with torch.no_grad():
            logits, loss = model(x, y)

        # Calculate per-token cross entropy loss and accuracy by exact context position inside the chunk
        log_probs = torch.log_softmax(logits[0], dim=-1)
        preds = torch.argmax(logits[0], dim=-1)

        for pos in range(len(chunk) - 1):
            target = chunk[pos + 1]
            nll = -log_probs[pos, target].item()
            is_correct = (preds[pos].item() == target)

            # Determine n-gram context category
            # If pos == 0 inside chunk, context length is 0 (Unigram position)
            # If pos == 1 inside chunk, context length is 1 (Bigram position)
            # If pos == 2 inside chunk, context length is 2 (Trigram position)
            # If pos >= 3 inside chunk, context length is >=3 (Long-range N-gram position)
            ctx_len = pos if pos <= 2 else 3
            losses_by_ctx[ctx_len].append(nll)
            if is_correct:
                correct_by_ctx[ctx_len] += 1
            total_by_ctx[ctx_len] += 1

    metrics = {}
    for ctx_len in [0, 1, 2, 3]:
        nlls = losses_by_ctx[ctx_len]
        if nlls:
            avg_nll_nats = sum(nlls) / len(nlls)
            bpt = avg_nll_nats / math.log(2)
            acc = correct_by_ctx[ctx_len] / total_by_ctx[ctx_len] * 100.0
            metrics[ctx_len] = {"bpt": bpt, "acc": acc, "count": len(nlls)}
        else:
            metrics[ctx_len] = {"bpt": 0.0, "acc": 0.0, "count": 0}

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_run8_looped.pt")
    ap.add_argument("--text_file", default="../data/dev_eval.txt")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found. Trying ckpt.pt instead.")
        args.checkpoint = "ckpt.pt"
        if not os.path.exists(args.checkpoint):
            print("No checkpoint found.")
            return

    # Load Tokenizer
    tok = tokenizer_mod.load()
    text = open(args.text_file, encoding="utf-8").read()
    tokens = tok.encode(text)
    total_bytes = len(text.encode("utf-8"))
    bytes_per_token = total_bytes / len(tokens)

    print(f"\n=======================================================")
    print(f"       N-GRAM METRICS & MODEL COMPARISON REPORT        ")
    print(f"=======================================================")
    print(f"Evaluation dataset: {args.text_file}")
    print(f"Total raw UTF-8 bytes: {total_bytes:,} | Total BPE tokens: {len(tokens):,} ({bytes_per_token:.2f} bytes/token)")

    # Compute Empirical N-gram Baselines
    emp = compute_empirical_ngrams(tokens, tok.vocab_size)
    print(f"\n--- EMPIRICAL BASELINES (on dev_eval.txt) ---")
    print(f"1. Unigram Baseline (Context=0): {emp['unigram_bpt']:.4f} bits/token ({emp['unigram_bpt']/bytes_per_token:.4f} bpb)")
    print(f"2. Bigram Baseline  (Context=1): {emp['bigram_bpt']:.4f} bits/token ({emp['bigram_bpt']/bytes_per_token:.4f} bpb)")
    print(f"3. Trigram Baseline (Context=2): {emp['trigram_bpt']:.4f} bits/token ({emp['trigram_bpt']/bytes_per_token:.4f} bpb)")

    # Load Model Checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = model_mod.Config()
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            setattr(cfg, k, v)
    model = model_mod.GPT(cfg)
    model.load_state_dict(ckpt["model"])

    # Evaluate Model across Context Lengths
    mod_metrics = evaluate_model_ngrams(model, tokens, cfg.block_size)

    print(f"\n--- MODEL PREDICTION METRICS BY CONTEXT DEPTH ({args.checkpoint}) ---")
    labels = {0: "Unigram (Context=0)", 1: "Bigram  (Context=1)", 2: "Trigram (Context=2)", 3: f"N-Gram  (Context=3..{cfg.block_size-1})"}
    for ctx_len in [0, 1, 2, 3]:
        m = mod_metrics[ctx_len]
        bpb = m["bpt"] / bytes_per_token
        print(f"[{labels[ctx_len]}] -> Model Loss: {m['bpt']:6.4f} bits/token ({bpb:6.4f} bpb) | Accuracy: {m['acc']:5.2f}% ({m['count']:,} positions evaluated)")

    print(f"\n--- COMPARISON SUMMARY: EMPIRICAL vs MODEL ---")
    print(f"Unigram Gap : Model is {mod_metrics[0]['bpt'] - emp['unigram_bpt']:+.4f} bits/token vs empirical unigram frequency")
    print(f"Bigram Gap  : Model is {mod_metrics[1]['bpt'] - emp['bigram_bpt']:+.4f} bits/token vs empirical bigram transition")
    print(f"Trigram Gap : Model is {mod_metrics[2]['bpt'] - emp['trigram_bpt']:+.4f} bits/token vs empirical trigram transition")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
