"""Plot N-Gram phase transitions with SEPARATE dedicated plots for different runs.
Generates clean multi-card/tabbed interactive Chart.js HTML (ngram_phases.html).
"""
import argparse
import glob
import json
import math
import os
import torch
import tokenizer as tokenizer_mod
import ngram_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="ngram_phases.html")
    args = ap.parse_args()

    def normalize_label(label_str, pt_name=""):
        s = (label_str + " " + pt_name).lower()
        if "run 10" in s or "accum3" in s or "accum 3x" in s:
            return "Run 10 CHAMPION (6L/136D, 3x Accum, Untied)"
        elif "run 7" in s or "tied" in s or "tie" in s:
            return "Run 7 (5L/128D Tied Weights)"
        elif "run 9" in s or "6l136d" in s or "6-layer" in s:
            return "Run 9 (6L/136D Unlooped Untied)"
        elif "run 8" in s or "looped" in s or "loop" in s:
            return "Run 8 (4L/160D Looped 2x)"
        elif "run 6" in s:
            return "Run 6 (5L/128D Untied Baseline)"
        elif "run 5" in s or "champion.pt" in s or "best config" in s:
            return "Run 5 (4L/160D Untied Baseline)"
        return label_str

    raw_runs = {}

    # 1. Load from run_history.json first
    history_path = "run_history.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
            for label, data in history.items():
                if "ngram_curves" in data and data["ngram_curves"]:
                    curves = {int(k): v for k, v in data["ngram_curves"].items()}
                    clean_label = normalize_label(label)
                    raw_runs[clean_label] = {"curves": curves, "losses": data.get("losses", [])}
        except Exception as e:
            print(f"Warning reading run_history.json: {e}")

    # 2. Load from any .pt checkpoint files in the current directory
    pt_files = sorted(glob.glob("ckpt*.pt") + glob.glob("*.pt"))
    for pt in pt_files:
        try:
            ckpt = torch.load(pt, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict) and "ngram_curves" in ckpt and ckpt["ngram_curves"]:
                curves = {int(k): v for k, v in ckpt["ngram_curves"].items()}
                clean_label = normalize_label(pt, pt)
                if pt == "ckpt.pt" and clean_label == "ckpt.pt":
                    continue
                raw_runs[clean_label] = {"curves": curves, "losses": ckpt.get("train_loss_curve", [])}
        except Exception:
            pass

    # Deduplicate exact identical curve values
    unique_runs = {}
    seen_fingerprints = set()
    for label, obj in raw_runs.items():
        curves = obj["curves"]
        sorted_steps = sorted(curves.keys())
        if not sorted_steps:
            continue
        last_step = sorted_steps[-1]
        m = curves[last_step]
        n_bpt = round(m[3]["bpt"] if 3 in m else m["3"]["bpt"], 4)
        fingerprint = (len(sorted_steps), last_step, n_bpt)
        
        if fingerprint in seen_fingerprints:
            for k in list(unique_runs.keys()):
                if unique_runs[k]["fingerprint"] == fingerprint:
                    if "CHAMPION" in label or ("Run " in label and "Run " not in k):
                        del unique_runs[k]
                        unique_runs[label] = {"curves": curves, "losses": obj.get("losses", []), "fingerprint": fingerprint}
            continue
        seen_fingerprints.add(fingerprint)
        unique_runs[label] = {"curves": curves, "losses": obj.get("losses", []), "fingerprint": fingerprint}

    if not unique_runs:
        print("No ngram_curves found across checkpoints or run_history.json.")
        return

    # Sort unique_runs so Run 10 CHAMPION is first, followed by Run 5, Run 7, Run 9, etc.
    def sort_key(item):
        label, data_obj = item
        if "CHAMPION" in label:
            return (0, label)
        elif "Run 5" in label:
            return (1, label)
        elif "Run 7" in label:
            return (2, label)
        elif "Run 9" in label:
            return (3, label)
        return (4, label)

    unique_runs = dict(sorted(unique_runs.items(), key=sort_key))

    # Compute empirical baselines
    tok = tokenizer_mod.load()
    dev_path = "../data/dev_eval.txt"
    dev_text = open(dev_path, encoding="utf-8").read()
    dev_tokens = tok.encode(dev_text)
    emp = ngram_metrics.compute_empirical_ngrams(dev_tokens, tok.vocab_size)

    print(f"\n=========================================================================")
    print(f"     SEPARATE N-GRAM PHASE PLOTS GENERATED FOR {len(unique_runs)} RUNS")
    print(f"=========================================================================")
    for label in unique_runs:
        print(f"  • {label}")
    print(f"=========================================================================\n")

    # Generate clean Chart.js plots for each run inside HTML cards showing Model Overall Cross-Entropy vs Statistical Baselines
    cards_html = ""
    chart_scripts = ""
    idx = 0

    for label, data_obj in unique_runs.items():
        curves = data_obj.get("curves", {})
        losses_nats = data_obj.get("losses", [])
        
        # If we loaded from checkpoint directly and curves exist, let's also grab losses if saved on object
        steps = sorted(curves.keys())
        if not steps and not losses_nats:
            continue
            
        val_data = []
        for s in steps:
            m = curves[s]
            # m[3]["bpt"] represents overall / long-range average across block
            val_data.append(round(m[3]["bpt"] if 3 in m else m["3"]["bpt"], 4))

        # Build smoothed training curve in bits per token
        train_steps = []
        train_data = []
        if losses_nats:
            window = 20
            smoothed = []
            for i in range(len(losses_nats)):
                start = max(0, i - window + 1)
                smoothed.append(sum(losses_nats[start : i + 1]) / (i - start + 1))
            # Sample every step or every few steps for Chart.js performance while retaining exact trajectory
            sample_rate = max(1, len(smoothed) // 300)
            for i in range(0, len(smoothed), sample_rate):
                train_steps.append(i + 1)
                train_data.append(round(smoothed[i] / math.log(2), 4))
            if train_steps[-1] != len(smoothed):
                train_steps.append(len(smoothed))
                train_data.append(round(smoothed[-1] / math.log(2), 4))

        final_val = val_data[-1] if val_data else (train_data[-1] if train_data else 0.0)

        # Determine step boundaries where model crosses empirical thresholds
        step_u = "N/A"
        step_b = "N/A"
        step_t = "N/A"
        if train_data and train_steps:
            for s_idx, val in zip(train_steps, train_data):
                if step_u == "N/A" and val < emp["unigram_bpt"]:
                    step_u = f"Step {s_idx}"
                if step_b == "N/A" and val < emp["bigram_bpt"]:
                    step_b = f"Step {s_idx}"
                if step_t == "N/A" and val < emp["trigram_bpt"]:
                    step_t = f"Step {s_idx}"
        elif val_data and steps:
            for s_idx, val in zip(steps, val_data):
                if step_u == "N/A" and val < emp["unigram_bpt"]:
                    step_u = f"Step {s_idx}"
                if step_b == "N/A" and val < emp["bigram_bpt"]:
                    step_b = f"Step {s_idx}"
                if step_t == "N/A" and val < emp["trigram_bpt"]:
                    step_t = f"Step {s_idx}"

        check_val = min([v for v in [final_val, (train_data[-1] if train_data else 999.0), (val_data[-1] if val_data else 999.0)] if v > 0])
        regime_status = "Trigram Transition"
        if check_val < emp["trigram_bpt"]:
            regime_status = "Deep Long-Range Compositional (&lt; H3)"
        elif check_val < emp["bigram_bpt"]:
            regime_status = "Bigram-to-Trigram Phase (H2 to H3)"
        elif check_val < emp["unigram_bpt"]:
            regime_status = "Unigram-to-Bigram Phase (H1 to H2)"

        cards_html += f"""
        <div class="run-card">
            <div class="run-header">
                <h2>{label}</h2>
                <div class="run-stats">
                    <span class="stat-badge">Crosses Unigram (H1={emp["unigram_bpt"]:.2f}): <strong>{step_u}</strong></span>
                    <span class="stat-badge">Crosses Bigram (H2={emp["bigram_bpt"]:.2f}): <strong>{step_b}</strong></span>
                    <span class="stat-badge">Crosses Trigram (H3={emp["trigram_bpt"]:.2f}): <strong>{step_t}</strong></span>
                    <span class="stat-badge stat-best">Current Regime: <strong>{regime_status}</strong> (Final: {final_val:.4f} bpt)</span>
                </div>
            </div>
            <div class="chart-box">
                <canvas id="chart_{idx}"></canvas>
            </div>
        </div>
        """

        # Prepare unified x-axis labels
        plot_labels = train_steps if train_steps else steps
        h1_vals = [round(emp["unigram_bpt"], 4) for _ in plot_labels]
        h2_vals = [round(emp["bigram_bpt"], 4) for _ in plot_labels]
        h3_vals = [round(emp["trigram_bpt"], 4) for _ in plot_labels]

        # Align validation points if train_steps is the primary x-axis
        val_dataset_js = ""
        if val_data and steps:
            if train_steps:
                aligned_val = []
                for label_step in plot_labels:
                    # Find closest step in validation curves
                    if label_step in steps:
                        aligned_val.append(val_data[steps.index(label_step)])
                    else:
                        aligned_val.append(None)
                val_dataset_js = f"""{{
                    label: 'Model Overall Validation Cross-Entropy (Dev Set)',
                    data: {json.dumps(aligned_val)},
                    borderColor: '#3fb950',
                    backgroundColor: '#3fb950',
                    borderWidth: 3,
                    pointRadius: 4,
                    spanGaps: true,
                    tension: 0.2
                }},"""
            else:
                val_dataset_js = f"""{{
                    label: 'Model Overall Validation Cross-Entropy (Dev Set)',
                    data: {json.dumps(val_data)},
                    borderColor: '#3fb950',
                    backgroundColor: '#3fb950',
                    borderWidth: 3.5,
                    pointRadius: 4,
                    tension: 0.2
                }},"""

        train_dataset_js = ""
        if train_data:
            train_dataset_js = f"""{{
                label: 'Model Overall Training Cross-Entropy (MA-20 Smoothed)',
                data: {json.dumps(train_data)},
                borderColor: '#58a6ff',
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.2
            }},"""

        chart_scripts += f"""
        new Chart(document.getElementById('chart_{idx}').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(plot_labels)},
                datasets: [
                    {{
                        label: 'Statistical Unigram Predictor vs Ground Truth (H1 = {emp["unigram_bpt"]:.4f} bpt)',
                        data: {json.dumps(h1_vals)},
                        borderColor: '#8b949e',
                        borderDash: [6, 6],
                        borderWidth: 1.8,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Statistical Bigram Predictor vs Ground Truth (H2 = {emp["bigram_bpt"]:.4f} bpt)',
                        data: {json.dumps(h2_vals)},
                        borderColor: '#ffa657',
                        borderDash: [6, 6],
                        borderWidth: 1.8,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Statistical Trigram Predictor vs Ground Truth (H3 = {emp["trigram_bpt"]:.4f} bpt)',
                        data: {json.dumps(h3_vals)},
                        borderColor: '#d2a8ff',
                        borderDash: [6, 6],
                        borderWidth: 1.8,
                        pointRadius: 0
                    }},
                    {train_dataset_js}
                    {val_dataset_js}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ grid: {{ color: '#2a344b' }}, title: {{ display: true, text: 'Training Step', color: '#c9d1d9' }} }},
                    y: {{ grid: {{ color: '#2a344b' }}, title: {{ display: true, text: 'Cross-Entropy Loss (Bits per Token)', color: '#c9d1d9' }} }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 12 }} }} }},
                    tooltip: {{ mode: 'index', intersect: false }}
                }}
            }}
        }});
        """
        idx += 1

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>N-Gram Phase Transitions - Dedicated Run Plots</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #0b0f17;
            color: #e6edf3;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        h1 {{ color: #58a6ff; font-size: 26px; margin-bottom: 8px; }}
        p.subtitle {{ color: #8b949e; margin-bottom: 30px; font-size: 15px; }}
        .run-card {{
            background: #161b26;
            border: 1px solid #2a344b;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 36px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }}
        .run-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2a344b;
            padding-bottom: 16px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .run-header h2 {{
            color: #f0f6fc;
            font-size: 20px;
            margin: 0;
        }}
        .run-stats {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .stat-badge {{
            background: #21283b;
            border: 1px solid #303c58;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            color: #c9d1d9;
        }}
        .stat-badge strong {{ color: #f0f6fc; }}
        .stat-best {{
            background: rgba(63, 185, 80, 0.15);
            border-color: #3fb950;
            color: #3fb950;
        }}
        .stat-best strong {{ color: #56d364; }}
        .chart-box {{ position: relative; height: 420px; width: 100%; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Transformer Overall Cross-Entropy vs. N-Gram Statistical Regimes</h1>
        <p class="subtitle">Clean training and validation cross-entropy curves (in bits per token across all positions) plotted directly against true Unigram ($H_1 = 7.59$), Bigram ($H_2 = 4.25$), and Trigram ($H_3 = 1.98$) statistical thresholds to reveal exactly when the model breaks across each complexity regime.</p>
        <div style="background: #161b26; border: 1px solid #2a344b; border-left: 4px solid #ffa657; border-radius: 8px; padding: 16px 20px; margin-bottom: 30px; font-size: 13.5px; line-height: 1.5; color: #c9d1d9;">
            <strong style="color: #ffa657;">Note on Validation vs. Training Curve Granularity:</strong><br>
            Notice that the <span style="color: #3fb950; font-weight: 600;">Validation Cross-Entropy curve (green line)</span> appears as a straight line connecting sparse points (<span style="color: #f0f6fc;">e.g., steps 1, 200, 400, ..., 2000</span> or evaluated only at the end of training phases). This is because full-corpus evaluation across <code style="color: #f0f6fc; background: #21283b; padding: 2px 6px; border-radius: 4px;">dev_eval.txt</code> (<span style="color: #f0f6fc;">61,404 tokens</span>) is computationally intensive and evaluated strictly at checkpoint intervals. Conversely, the <span style="color: #58a6ff; font-weight: 600;">Smoothed Training Cross-Entropy curve (blue line)</span> tracks real-time convergence step-by-step across all <span style="color: #f0f6fc;">2,000 optimizer steps</span>.
        </div>
        {cards_html}
    </div>
    <script>
        {chart_scripts}
    </script>
</body>
</html>"""

    with open(args.html, "w") as f:
        f.write(html_content)
    if args.html == "ngram_phases.html":
        with open("ngram_regimes.html", "w") as f:
            f.write(html_content)
    print(f"Saved dedicated individual plots for {len(unique_runs)} runs to {args.html} and ngram_regimes.html")


if __name__ == "__main__":
    main()
