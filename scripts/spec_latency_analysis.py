#!/usr/bin/env python3
"""
Analyze latency patterns in speculative decoding.

Hypothesis:
- Draft latency is linear with n_tokens (sequential generation)
- Verify latency has step changes at batch size boundaries
"""

import json
import sys
from collections import defaultdict


def load_data(path):
    """Load rounds with timing data."""
    rounds = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "timing" in r:
                    rounds.append(r)
    return rounds


def ascii_scatter(points, width=70, height=20, x_label="", y_label=""):
    """Create ASCII scatter plot."""
    if not points:
        return "No data"

    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    # Add some padding
    y_range = y_max - y_min if y_max > y_min else 1
    y_max += y_range * 0.05
    y_min = max(0, y_min - y_range * 0.05)

    x_range = x_max - x_min if x_max > x_min else 1

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot points
    for x, y in points:
        col = int((x - x_min) / x_range * (width - 1)) if x_range > 0 else 0
        row = int((y_max - y) / (y_max - y_min) * (height - 1)) if y_max > y_min else 0
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = '●'

    # Build output
    lines = []
    lines.append(f"  {y_label}")
    lines.append(f"  {y_max:,.0f} ┤")

    for i, row in enumerate(grid):
        if i == height // 2:
            lines.append(f"       │{''.join(row)}")
        else:
            lines.append(f"       │{''.join(row)}")

    lines.append(f"  {y_min:,.0f} ┤{'─' * width}")
    lines.append(f"       └{'─' * width}")
    lines.append(f"        {x_min:<10.0f}{' ' * (width - 22)}{x_max:>10.0f}")
    lines.append(f"        {' ' * (width // 2 - len(x_label) // 2)}{x_label}")

    return '\n'.join(lines)


def analyze_by_bucket(rounds, bucket_size=4):
    """Analyze latency by n_draft buckets."""
    buckets = defaultdict(lambda: {"draft": [], "verify": [], "count": 0})

    for r in rounds:
        n = r["draft"]["n"]
        t_draft = r["timing"]["t_draft_us"] / 1000  # ms
        t_verify = r["timing"]["t_verify_us"] / 1000  # ms

        bucket = (n // bucket_size) * bucket_size
        buckets[bucket]["draft"].append(t_draft)
        buckets[bucket]["verify"].append(t_verify)
        buckets[bucket]["count"] += 1

    return buckets


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <spec_log.jsonl>")
        sys.exit(1)

    rounds = load_data(sys.argv[1])

    if not rounds:
        print("No timing data found")
        sys.exit(1)

    print("=" * 75)
    print("LATENCY ANALYSIS - SPECULATIVE DECODING")
    print("=" * 75)
    print(f"Total rounds with timing: {len(rounds)}")

    # Extract data points
    draft_points = []  # (n_draft, t_draft_ms)
    verify_points = []  # (n_draft, t_verify_ms)

    for r in rounds:
        n = r["draft"]["n"]
        t_draft = r["timing"]["t_draft_us"] / 1000
        t_verify = r["timing"]["t_verify_us"] / 1000
        draft_points.append((n, t_draft))
        verify_points.append((n, t_verify))

    # Draft latency scatter plot
    print("\n" + "=" * 75)
    print("DRAFT LATENCY vs N_TOKENS (expecting linear relationship)")
    print("=" * 75)
    print(ascii_scatter(draft_points, x_label="n_drafted", y_label="t_draft (ms)"))

    # Verify latency scatter plot
    print("\n" + "=" * 75)
    print("VERIFY LATENCY vs BATCH SIZE (looking for step patterns)")
    print("=" * 75)
    print(ascii_scatter(verify_points, x_label="n_drafted (batch size)", y_label="t_verify (ms)"))

    # Analyze by bucket
    print("\n" + "=" * 75)
    print("LATENCY BY BATCH SIZE BUCKET")
    print("=" * 75)

    buckets = analyze_by_bucket(rounds, bucket_size=4)

    print(f"\n{'Bucket':<12} {'Count':>6} {'Avg Draft':>12} {'Avg Verify':>12} {'Draft/tok':>12} {'Verify Δ':>12}")
    print("-" * 75)

    prev_verify = None
    for bucket in sorted(buckets.keys()):
        b = buckets[bucket]
        if b["count"] == 0:
            continue

        avg_draft = sum(b["draft"]) / len(b["draft"])
        avg_verify = sum(b["verify"]) / len(b["verify"])
        draft_per_tok = avg_draft / (bucket + 2) if bucket > 0 else avg_draft  # +2 for bucket midpoint

        if prev_verify is not None:
            delta = avg_verify - prev_verify
            delta_str = f"{delta:+.1f}ms"
        else:
            delta_str = "-"

        print(f"{bucket:>3}-{bucket+3:<5} {b['count']:>6} {avg_draft:>10.1f}ms {avg_verify:>10.1f}ms {draft_per_tok:>10.1f}ms {delta_str:>12}")
        prev_verify = avg_verify

    # Per-token analysis
    print("\n" + "=" * 75)
    print("PER-TOKEN LATENCY ANALYSIS")
    print("=" * 75)

    by_n = defaultdict(lambda: {"draft": [], "verify": []})
    for r in rounds:
        n = r["draft"]["n"]
        by_n[n]["draft"].append(r["timing"]["t_draft_us"] / 1000)
        by_n[n]["verify"].append(r["timing"]["t_verify_us"] / 1000)

    print(f"\n{'N':>3} {'Count':>6} {'Avg Draft':>12} {'Avg Verify':>12} {'Draft/tok':>10} {'Verify/tok':>10}")
    print("-" * 75)

    for n in sorted(by_n.keys()):
        data = by_n[n]
        count = len(data["draft"])
        avg_d = sum(data["draft"]) / count
        avg_v = sum(data["verify"]) / count
        d_per_tok = avg_d / n if n > 0 else 0
        v_per_tok = avg_v / (n + 1) if n > 0 else avg_v  # +1 for sampled token in batch
        print(f"{n:>3} {count:>6} {avg_d:>10.1f}ms {avg_v:>10.1f}ms {d_per_tok:>8.1f}ms {v_per_tok:>8.1f}ms")

    # Verify step detection
    print("\n" + "=" * 75)
    print("VERIFY LATENCY STEP DETECTION")
    print("=" * 75)

    # Group verify times by n and compute mean
    verify_by_n = {}
    for n in sorted(by_n.keys()):
        data = by_n[n]
        if len(data["verify"]) >= 1:
            verify_by_n[n] = sum(data["verify"]) / len(data["verify"])

    # Find significant jumps
    print("\nLooking for step changes in verify latency (>50ms jump):\n")
    prev_n = None
    prev_v = None
    for n in sorted(verify_by_n.keys()):
        v = verify_by_n[n]
        if prev_v is not None:
            delta = v - prev_v
            if abs(delta) > 50:  # significant jump
                direction = "↑" if delta > 0 else "↓"
                print(f"  {prev_n:>2} → {n:>2}: {prev_v:>7.1f}ms → {v:>7.1f}ms  ({direction} {delta:+.1f}ms)")
        prev_n = n
        prev_v = v

    # Linear regression for draft
    print("\n" + "=" * 75)
    print("DRAFT LINEARITY CHECK")
    print("=" * 75)

    n_vals = [p[0] for p in draft_points]
    d_vals = [p[1] for p in draft_points]

    n_mean = sum(n_vals) / len(n_vals)
    d_mean = sum(d_vals) / len(d_vals)

    numerator = sum((n - n_mean) * (d - d_mean) for n, d in zip(n_vals, d_vals))
    denominator = sum((n - n_mean) ** 2 for n in n_vals)

    if denominator > 0:
        slope = numerator / denominator
        intercept = d_mean - slope * n_mean

        # R-squared
        ss_res = sum((d - (slope * n + intercept)) ** 2 for n, d in zip(n_vals, d_vals))
        ss_tot = sum((d - d_mean) ** 2 for d in d_vals)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"\nLinear fit: t_draft = {slope:.2f} * n_tokens + {intercept:.2f}")
        print(f"R² = {r_squared:.4f}")
        print(f"Interpretation: {slope:.1f}ms per drafted token")

        if r_squared > 0.8:
            print("✓ Strong linear relationship (R² > 0.8)")
        elif r_squared > 0.5:
            print("~ Moderate linear relationship (0.5 < R² < 0.8)")
        else:
            print("✗ Weak linear relationship (R² < 0.5)")


if __name__ == "__main__":
    main()
