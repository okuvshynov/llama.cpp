#!/usr/bin/env python3
"""
Simulate different p/entropy thresholds for speculative decoding.

For each token in a draft, we know:
- p: probability of selected token
- entropy: entropy of the top-k distribution
- used: whether this token (and all before it) were accepted

This script simulates: "what if we stopped drafting at p < P or entropy > E?"
and reports drafted/used counts and efficiency ratios.
"""

import json
import sys
from collections import defaultdict


def load_rounds(path):
    """Load all rounds from JSONL."""
    rounds = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rounds.append(json.loads(line))
    return rounds


def simulate_threshold(rounds, p_min=0.0, entropy_max=float('inf')):
    """
    Simulate stopping when p < p_min or entropy > entropy_max.

    Returns:
        total_drafted: tokens we would have drafted
        total_used: tokens that were actually accepted
        rounds_data: per-round breakdown
    """
    total_drafted = 0
    total_used = 0
    rounds_data = []

    for r in rounds:
        tokens = r["draft"]["tokens"]
        n_accepted = r["verify"]["n_accepted"]

        # Find where we would stop with this threshold
        draft_stop = 0
        for t in tokens:
            if t["p"] >= p_min and t["entropy"] <= entropy_max:
                draft_stop += 1
            else:
                break

        # How many of those drafted tokens were actually used?
        # A token at position i is "used" if n_accepted > i
        used = min(draft_stop, n_accepted)

        total_drafted += draft_stop
        total_used += used
        rounds_data.append({
            "drafted": draft_stop,
            "used": used,
            "n_accepted_actual": n_accepted,
        })

    return total_drafted, total_used, rounds_data


def print_results(results, label):
    """Print results for a set of thresholds."""
    print(f"\n{label}")
    print("=" * 70)
    print(f"{'Threshold':<20} {'Drafted':>10} {'Used':>10} {'Ratio':>10} {'Saved':>10}")
    print("-" * 70)

    baseline_drafted = results[0][1] if results else 0

    for name, drafted, used, _ in results:
        ratio = used / drafted if drafted > 0 else 0
        saved = baseline_drafted - drafted
        saved_pct = saved / baseline_drafted * 100 if baseline_drafted > 0 else 0
        print(f"{name:<20} {drafted:>10} {used:>10} {ratio:>10.1%} {saved:>9.0f} ({saved_pct:.0f}%)")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <spec_log.jsonl>")
        sys.exit(1)

    rounds = load_rounds(sys.argv[1])

    if not rounds:
        print("No data found")
        sys.exit(1)

    print("=" * 70)
    print("THRESHOLD SIMULATION - SPECULATIVE DECODING")
    print("=" * 70)
    print(f"Total rounds: {len(rounds)}")

    # Baseline: no filtering
    baseline_drafted, baseline_used, _ = simulate_threshold(rounds)
    print(f"Baseline (no filter): {baseline_drafted} drafted, {baseline_used} used ({baseline_used/baseline_drafted:.1%})")

    # P_min thresholds
    p_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    p_results = []
    for p in p_thresholds:
        drafted, used, _ = simulate_threshold(rounds, p_min=p)
        p_results.append((f"p >= {p:.1f}", drafted, used, None))

    print_results(p_results, "P_MIN THRESHOLDS (stop when p < threshold)")

    # Entropy thresholds
    e_thresholds = [float('inf'), 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    e_results = []
    for e in e_thresholds:
        drafted, used, _ = simulate_threshold(rounds, entropy_max=e)
        label = "no limit" if e == float('inf') else f"H <= {e:.1f}"
        e_results.append((label, drafted, used, None))

    print_results(e_results, "ENTROPY THRESHOLDS (stop when entropy > threshold)")

    # Combined thresholds - grid search
    print(f"\nCOMBINED THRESHOLDS (p_min, entropy_max)")
    print("=" * 70)

    # Find interesting combinations
    combined = []
    for p in [0.0, 0.3, 0.5, 0.7]:
        for e in [float('inf'), 1.0, 0.5, 0.3]:
            drafted, used, _ = simulate_threshold(rounds, p_min=p, entropy_max=e)
            ratio = used / drafted if drafted > 0 else 0
            e_str = "∞" if e == float('inf') else f"{e:.1f}"
            combined.append((p, e, drafted, used, ratio))

    # Sort by ratio descending
    combined.sort(key=lambda x: -x[4])

    print(f"{'p_min':<8} {'H_max':<8} {'Drafted':>10} {'Used':>10} {'Ratio':>10} {'Tokens/Round':>12}")
    print("-" * 70)
    for p, e, drafted, used, ratio in combined:
        e_str = "∞" if e == float('inf') else f"{e:.1f}"
        tpr = drafted / len(rounds)
        print(f"{p:<8.1f} {e_str:<8} {drafted:>10} {used:>10} {ratio:>10.1%} {tpr:>12.1f}")

    # Efficiency analysis: ratio vs throughput tradeoff
    print(f"\nEFFICIENCY FRONTIER")
    print("=" * 70)
    print("Best ratio at each drafted token budget:")
    print(f"{'Min Drafted':<12} {'Best p':<8} {'Best H':<8} {'Drafted':>10} {'Used':>10} {'Ratio':>10}")
    print("-" * 70)

    # Generate all combinations
    all_combos = []
    for p in [i/10 for i in range(10)]:
        for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, float('inf')]:
            drafted, used, _ = simulate_threshold(rounds, p_min=p, entropy_max=e)
            if drafted > 0:
                all_combos.append((p, e, drafted, used, used/drafted))

    # Find Pareto frontier: best ratio for each drafted level
    all_combos.sort(key=lambda x: x[2])  # sort by drafted

    buckets = [100, 500, 1000, 2000, 3000, 4000, baseline_drafted]
    for min_drafted in buckets:
        candidates = [c for c in all_combos if c[2] >= min_drafted]
        if candidates:
            best = max(candidates, key=lambda x: x[4])
            p, e, drafted, used, ratio = best
            e_str = "∞" if e == float('inf') else f"{e:.1f}"
            print(f"{min_drafted:<12} {p:<8.1f} {e_str:<8} {drafted:>10} {used:>10} {ratio:>10.1%}")

    # Per-position analysis
    print(f"\nPER-POSITION ANALYSIS")
    print("=" * 70)
    print("Acceptance rate and avg entropy/p at each draft position:")
    print(f"{'Pos':<6} {'Count':>8} {'Accepted':>10} {'Rate':>8} {'Avg P':>10} {'Avg H':>10}")
    print("-" * 70)

    pos_stats = defaultdict(lambda: {"count": 0, "accepted": 0, "p_sum": 0, "h_sum": 0})

    for r in rounds:
        tokens = r["draft"]["tokens"]
        n_accepted = r["verify"]["n_accepted"]

        for t in tokens:
            pos = t["pos"]
            pos_stats[pos]["count"] += 1
            pos_stats[pos]["p_sum"] += t["p"]
            pos_stats[pos]["h_sum"] += t["entropy"]
            if pos < n_accepted:
                pos_stats[pos]["accepted"] += 1

    for pos in sorted(pos_stats.keys()):
        s = pos_stats[pos]
        rate = s["accepted"] / s["count"] if s["count"] > 0 else 0
        avg_p = s["p_sum"] / s["count"] if s["count"] > 0 else 0
        avg_h = s["h_sum"] / s["count"] if s["count"] > 0 else 0
        print(f"{pos:<6} {s['count']:>8} {s['accepted']:>10} {rate:>8.1%} {avg_p:>10.3f} {avg_h:>10.3f}")


if __name__ == "__main__":
    main()
