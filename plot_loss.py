"""Read train_loss_curve from ckpt.pt and all runs from run_history.json and generate:
1. An ASCII terminal plot of the latest loss curve.
2. A multi-run interactive HTML plot (loss_curve.html) showing all historical runs.
Pure PyTorch + stdlib.
"""
import argparse
import json
import os
import torch


def plot_ascii(losses, title="Training Loss Curve", width=70, height=18, window=20):
    if not losses:
        print("No loss data to plot.")
        return

    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window + 1)
        smoothed.append(sum(losses[start : i + 1]) / (i - start + 1))

    min_val = min(smoothed)
    max_val = max(smoothed)
    if max_val == min_val:
        max_val += 1e-5

    print(f"\n--- {title} (Smoothed MA-{window}) ---")
    print(f"Max Loss: {max_val:.4f} | Min Loss: {min_val:.4f} | Final Loss: {smoothed[-1]:.4f}\n")

    step_indices = [int(i * (len(smoothed) - 1) / (width - 1)) for i in range(width)]
    sampled = [smoothed[idx] for idx in step_indices]

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for col, val in enumerate(sampled):
        row = int((val - min_val) / (max_val - min_val) * (height - 1))
        row = height - 1 - row
        grid[row][col] = "*"

    for r in range(height):
        y_val = max_val - (r / (height - 1)) * (max_val - min_val)
        line_str = "".join(grid[r])
        if r == 0 or r == height // 2 or r == height - 1:
            print(f"{y_val:6.3f} | {line_str}")
        else:
            print(f"       | {line_str}")
    print("       +" + "-" * width)
    print(f"       0{' '*(width-12)}Steps{' '*(width-12)}{len(losses)}\n")


def generate_multi_html(history_dict, out_path="loss_curve.html", window=20):
    # Palette of vibrant distinct colors
    colors = [
        "#58a6ff", "#3fb950", "#f85149", "#d2a8ff", "#ffa657",
        "#7ee787", "#ff7b72", "#a5d6ff", "#ffd97d", "#e3b341"
    ]

    def normalize_label(label_str):
        s = label_str.lower()
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
        elif "run 5" in s or "champion" in s or "best config" in s:
            return "Run 5 (4L/160D Untied Baseline)"
        return label_str

    unique_history = {}
    for label, data in history_dict.items():
        if label == "Latest Checkpoint":
            continue
        clean = normalize_label(label)
        unique_history[clean] = data

    datasets = []
    all_steps = []

    # Sort so Run 10 CHAMPION is first
    def sort_key(item):
        label, _ = item
        if "CHAMPION" in label:
            return (0, label)
        elif "Run 5" in label:
            return (1, label)
        elif "Run 7" in label:
            return (2, label)
        elif "Run 9" in label:
            return (3, label)
        return (4, label)

    sorted_history = dict(sorted(unique_history.items(), key=sort_key))

    for idx, (label, data) in enumerate(sorted_history.items()):
        losses = data.get("losses", [])
        if not losses:
            continue
        if len(losses) > len(all_steps):
            all_steps = list(range(1, len(losses) + 1))

        smoothed = []
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            smoothed.append(sum(losses[start : i + 1]) / (i - start + 1))

        color = colors[idx % len(colors)]
        datasets.append({
            "label": f"{label} (MA-{window})",
            "data": [round(l, 4) for l in smoothed[::2]],
            "borderColor": color,
            "borderWidth": 2.5 if "CHAMPION" in label else 1.8,
            "pointRadius": 0,
            "tension": 0.2
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Training Loss Curves Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 40px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            width: 100%;
            max-width: 1100px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }}
        h1 {{
            margin-top: 0;
            color: #58a6ff;
            font-size: 24px;
        }}
        p {{
            color: #8b949e;
            font-size: 14px;
            margin-bottom: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Historical Training & Validation Loss Curves Comparison</h1>
        <p>Click items in the legend below to show/hide specific training runs.</p>
        <div style="background: #161b22; border: 1px solid #30363d; border-left: 4px solid #ffa657; border-radius: 8px; padding: 14px 18px; margin-bottom: 24px; font-size: 13.5px; line-height: 1.5; color: #c9d1d9;">
            <strong style="color: #ffa657;">Note on Validation vs. Training Curve Granularity:</strong><br>
            Validation loss curves appear as straight/sparse lines connecting checkpoint evaluation intervals (<span style="color: #f0f6fc;">or evaluated at phase endpoints</span>), whereas smoothed training loss tracks real-time convergence continuously across all <span style="color: #f0f6fc;">2,000 steps</span>.
        </div>
        <div style="position: relative; height: 500px; width: 100%;">
            <canvas id="lossChart"></canvas>
        </div>
    </div>
    <script>
        const ctx = document.getElementById('lossChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(all_steps[::2])},
                datasets: {json.dumps(datasets)}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        grid: {{ color: '#30363d' }},
                        ticks: {{ color: '#8b949e' }},
                        title: {{ display: true, text: 'Optimizer Steps', color: '#c9d1d9' }}
                    }},
                    y: {{
                        grid: {{ color: '#30363d' }},
                        ticks: {{ color: '#8b949e' }},
                        title: {{ display: true, text: 'Smoothed Cross-Entropy Loss', color: '#c9d1d9' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 13 }} }} }},
                    tooltip: {{ mode: 'index', intersect: false }}
                }},
                interaction: {{ mode: 'nearest', axis: 'x', intersect: false }}
            }}
        }});
    </script>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved interactive multi-run comparison HTML plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt.pt")
    ap.add_argument("--html", default="loss_curve.html")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()

    history_path = "run_history.json"
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        losses = ckpt.get("train_loss_curve")
        if losses:
            # Make sure latest checkpoint is in history dict if not already added
            if "Latest Checkpoint" not in history and len(history) == 0:
                history["Latest Checkpoint"] = {"losses": losses}
            plot_ascii(losses, title="Latest Checkpoint Training Loss", window=args.window)

    if history:
        generate_multi_html(history, out_path=args.html, window=args.window)
    else:
        print("No run history found to generate multi-run HTML plot.")


if __name__ == "__main__":
    main()
