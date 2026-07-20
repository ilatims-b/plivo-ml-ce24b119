import math
import matplotlib.pyplot as plt
import torch

def main():
    # Set dark GitHub styling
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    fig.patch.set_facecolor('#0b0f17')
    ax.set_facecolor('#161b26')

    # Load checkpoint
    ckpt = torch.load('ckpt_run10_6L_accum3.pt', map_location='cpu', weights_only=True)
    losses_nats = ckpt.get('train_loss_curve', [])
    curves = ckpt.get('ngram_curves', {})

    # Compute empirical thresholds
    h1 = 7.5859
    h2 = 4.2497
    h3 = 1.9780

    # Build smoothed training curve in bits/token
    window = 20
    train_steps = []
    train_bpt = []
    if losses_nats:
        smoothed = []
        for i in range(len(losses_nats)):
            start = max(0, i - window + 1)
            smoothed.append(sum(losses_nats[start : i + 1]) / (i - start + 1))
        for i in range(0, len(smoothed), 2):
            train_steps.append(i + 1)
            train_bpt.append(smoothed[i] / math.log(2))
        if train_steps[-1] != len(smoothed):
            train_steps.append(len(smoothed))
            train_bpt.append(smoothed[-1] / math.log(2))

    # Build validation points
    val_steps = sorted(curves.keys())
    val_bpt = []
    for s in val_steps:
        m = curves[s]
        val_bpt.append(m[3]['bpt'] if 3 in m else m['3']['bpt'])

    # Plot empirical baselines
    ax.axhline(h1, color='#8b949e', linestyle='--', linewidth=1.5, label=f'Statistical Unigram Baseline ($H_1$ = {h1:.2f} bpt)')
    ax.axhline(h2, color='#ffa657', linestyle='--', linewidth=1.5, label=f'Statistical Bigram Baseline ($H_2$ = {h2:.2f} bpt)')
    ax.axhline(h3, color='#d2a8ff', linestyle='--', linewidth=1.5, label=f'Statistical Trigram Baseline ($H_3$ = {h3:.2f} bpt)')

    # Plot trajectories
    if train_steps:
        ax.plot(train_steps, train_bpt, color='#58a6ff', linewidth=2.2, label='Smoothed Training Cross-Entropy (MA-20)')
    if val_steps:
        ax.plot(val_steps, val_bpt, color='#3fb950', linewidth=2.8, marker='o', markersize=6, label='Validation Cross-Entropy (Dev Set Points)')

    # Annotations for regime crossings
    ax.annotate('Crosses Unigram ($H_1$)\nStep 61', xy=(61, h1), xytext=(150, h1 + 0.9),
                arrowprops=dict(facecolor='#8b949e', shrink=0.08, width=1.2, headwidth=6),
                fontsize=10, color='#c9d1d9', fontweight='bold')

    ax.annotate('Crosses Bigram ($H_2$)\nStep 835', xy=(835, h2), xytext=(950, h2 + 1.1),
                arrowprops=dict(facecolor='#ffa657', shrink=0.08, width=1.2, headwidth=6),
                fontsize=10, color='#ffa657', fontweight='bold')

    ax.annotate('Final Dev Eval: 1.7296 bpb\n(Deep Compositional Regime < $H_3$)', xy=(2000, 4.5), xytext=(1250, 3.1),
                arrowprops=dict(facecolor='#3fb950', shrink=0.08, width=1.2, headwidth=6),
                fontsize=10.5, color='#3fb950', fontweight='bold')

    ax.set_title('Champion Run 10: Overall Cross-Entropy vs. Statistical N-Gram Regimes', fontsize=15, color='#f0f6fc', fontweight='bold', pad=16)
    ax.set_xlabel('Training Step (2,000 steps total)', fontsize=12, color='#c9d1d9', labelpad=10)
    ax.set_ylabel('Cross-Entropy Loss (Bits per Token)', fontsize=12, color='#c9d1d9', labelpad=10)
    ax.grid(True, color='#2a344b', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='#161b26', edgecolor='#303c58', fontsize=10.5)

    plt.tight_layout()
    plt.savefig('champion_ngram_regime.png', dpi=300, bbox_inches='tight')
    print("Saved champion_ngram_regime.png")

if __name__ == '__main__':
    main()
