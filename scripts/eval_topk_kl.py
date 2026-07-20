"""Compute exact Top-K Truncated KL Divergence between model prediction logits
and empirical N-Gram probability distributions across Context Lengths (Unigram, Bigram, Trigram).
"""
import argparse
import math
import os
from collections import defaultdict
import torch
import tokenizer as tokenizer_mod
import model as model_mod
import ngram_metrics


def compute_topk_kl(p_dict, q_tensor, top_k=50, eps=1e-9):
    """Compute Forward and Reverse KL divergence over Top-K trimmed probability support.
    
    Args:
        p_dict: dict of {token_id: empirical_count} for the context
        q_tensor: 1D torch tensor of model probabilities across vocab (size 2048)
        top_k: int, number of top tokens to retain for feasible computation
        eps: smoothing factor to avoid log(0)
    Returns:
        fwd_kl: D_KL(P || Q) in bits
        rev_kl: D_KL(Q || P) in bits
    """
    # Get top-K token indices from model probabilities
    q_topk_vals, q_topk_idx = torch.topk(q_tensor, min(top_k, len(q_tensor)))
    top_indices = set(q_topk_idx.tolist())
    
    # Also include any tokens that actually occurred in empirical counts for this context
    if p_dict:
        # Sort empirical by count descending and add up to top_k empirical items
        sorted_emp = sorted(p_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for tid, _ in sorted_emp:
            top_indices.add(tid)
            
    top_indices = list(top_indices)
    
    # Build raw trimmed P and Q
    p_raw = []
    q_raw = []
    total_emp_counts = sum(p_dict.values()) if p_dict else 0
    
    for tid in top_indices:
        q_raw.append(q_tensor[tid].item())
        if total_emp_counts > 0 and tid in p_dict:
            p_raw.append(p_dict[tid] / total_emp_counts)
        else:
            p_raw.append(eps)
            
    # Renormalize P and Q over the Top-K support
    p_sum = sum(p_raw)
    q_sum = sum(q_raw)
    
    p_norm = [max(eps, p / p_sum) for p in p_raw]
    q_norm = [max(eps, q / q_sum) for q in q_raw]
    
    # Compute Forward KL: sum P(x) * log2(P(x) / Q(x))
    fwd_kl = sum(p * math.log2(p / q) for p, q in zip(p_norm, q_norm))
    
    # Compute Reverse KL: sum Q(x) * log2(Q(x) / P(x))
    rev_kl = sum(q * math.log2(q / p) for p, q in zip(p_norm, q_norm))
    
    return max(0.0, fwd_kl), max(0.0, rev_kl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_run5_champion.pt")
    ap.add_argument("--text_file", default="../data/dev_eval.txt")
    ap.add_argument("--top_k", type=int, default=50, help="Number of top vocabulary logits to retain for feasible KL computation")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        if os.path.exists("ckpt.pt"):
            args.checkpoint = "ckpt.pt"
        else:
            print(f"Checkpoint {args.checkpoint} not found.")
            return

    # Load Tokenizer & Dev Data
    tok = tokenizer_mod.load()
    text = open(args.text_file, encoding="utf-8").read()
    tokens = tok.encode(text)
    
    # Compute empirical counts tables
    emp = ngram_metrics.compute_empirical_ngrams(tokens, tok.vocab_size)
    unigram_counts = emp["unigram_counts"]
    bigram_counts = emp["bigram_counts"]
    trigram_counts = emp["trigram_counts"]

    # Load Model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = model_mod.Config()
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            setattr(cfg, k, v)
    model = model_mod.GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"\n=========================================================================")
    print(f"    EXACT TOP-K TRUNCATED KL DIVERGENCE vs EMPIRICAL N-GRAM STATISTICS")
    print(f"=========================================================================")
    print(f"Model Checkpoint : {args.checkpoint}")
    print(f"Evaluation Data  : {args.text_file} ({len(tokens):,} BPE tokens)")
    print(f"Top-K Truncation : K = {args.top_k} (feasible exact logits divergence)")
    print(f"-------------------------------------------------------------------------")

    fwd_kl_by_ctx = defaultdict(list)
    rev_kl_by_ctx = defaultdict(list)
    
    stride = cfg.block_size // 2
    for start in range(0, len(tokens) - 1, stride):
        end = min(start + cfg.block_size, len(tokens))
        chunk = tokens[start:end]
        if len(chunk) < 2:
            break

        x = torch.tensor(chunk[:-1], dtype=torch.long)[None, :]
        y = torch.tensor(chunk[1:], dtype=torch.long)[None, :]

        with torch.no_grad():
            logits, _ = model(x, y)
        probs = torch.softmax(logits[0], dim=-1)

        for pos in range(min(3, len(chunk) - 1)):
            q_vec = probs[pos]
            if pos == 0:
                # Unigram context
                p_dict = unigram_counts
            elif pos == 1:
                # Bigram context (given chunk[0])
                w0 = chunk[0]
                p_dict = {t2: c for (t1, t2), c in bigram_counts.items() if t1 == w0}
            elif pos == 2:
                # Trigram context (given chunk[0], chunk[1])
                w0, w1 = chunk[0], chunk[1]
                p_dict = {t3: c for (t1, t2, t3), c in trigram_counts.items() if t1 == w0 and t2 == w1}
            else:
                continue

            f_kl, r_kl = compute_topk_kl(p_dict, q_vec, top_k=args.top_k)
            fwd_kl_by_ctx[pos].append(f_kl)
            rev_kl_by_ctx[pos].append(r_kl)

    labels = {0: "Unigram (Context=0)", 1: "Bigram  (Context=1)", 2: "Trigram (Context=2)"}
    for ctx in [0, 1, 2]:
        f_list = fwd_kl_by_ctx[ctx]
        r_list = rev_kl_by_ctx[ctx]
        if f_list:
            avg_fwd = sum(f_list) / len(f_list)
            avg_rev = sum(r_list) / len(r_list)
            print(f"[{labels[ctx]}]")
            print(f"  • Forward KL D(P_empirical || Q_model) : {avg_fwd:6.4f} bits/token")
            print(f"  • Reverse KL D(Q_model || P_empirical) : {avg_rev:6.4f} bits/token ({len(f_list):,} contexts)")
            print(f"-------------------------------------------------------------------------")

    print(f"=========================================================================\n")


if __name__ == "__main__":
    main()
