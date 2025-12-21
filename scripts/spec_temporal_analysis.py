#!/usr/bin/env python3
"""
Analyze temporal patterns in speculative decoding acceptance rates.

Investigates whether acceptance rates are autocorrelated - e.g., if past
rounds had high acceptance, is the next round likely to as well?

Hypothesis: structured content (code) has consistently high acceptance,
while freeform text has lower/more variable acceptance.
"""

import json
import sys
from collections import defaultdict


def load_rounds(path):
    """Load rounds from JSONL, grouped by slot_id."""
    slots = defaultdict(list)
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                slots[r["slot_id"]].append(r)
    # Sort each slot by round_id
    for slot_id in slots:
        slots[slot_id].sort(key=lambda x: x["round_id"])
    return slots


def acceptance_rate(r):
    """Compute acceptance rate for a single round."""
    n_drafted = r["draft"]["n"]
    n_accepted = r["verify"]["n_accepted"]
    return n_accepted / n_drafted if n_drafted > 0 else 0.0


def analyze_autocorrelation(rates, max_lag=5):
    """Compute autocorrelation at different lags."""
    n = len(rates)
    if n < max_lag + 2:
        return {}

    mean = sum(rates) / n
    var = sum((r - mean) ** 2 for r in rates) / n
    if var == 0:
        return {}

    autocorr = {}
    for lag in range(1, max_lag + 1):
        if n - lag < 2:
            break
        cov = sum((rates[i] - mean) * (rates[i + lag] - mean)
                  for i in range(n - lag)) / (n - lag)
        autocorr[lag] = cov / var
    return autocorr


def analyze_runs(rates, threshold=0.7):
    """Analyze runs of high/low acceptance."""
    runs = []
    current_run = {"high": rates[0] >= threshold, "length": 1, "start": 0}

    for i, rate in enumerate(rates[1:], 1):
        is_high = rate >= threshold
        if is_high == current_run["high"]:
            current_run["length"] += 1
        else:
            runs.append(current_run)
            current_run = {"high": is_high, "length": 1, "start": i}
    runs.append(current_run)

    return runs


def conditional_probability(rates, window=2, threshold=0.7):
    """
    Compute P(next round is high | past `window` rounds were all high/low).
    """
    n = len(rates)
    if n <= window:
        return None, None

    # Count cases where past `window` rounds were all high
    high_given_high_count = 0
    high_given_high_total = 0
    high_given_low_count = 0
    high_given_low_total = 0

    for i in range(window, n):
        past = rates[i - window:i]
        current_high = rates[i] >= threshold

        if all(r >= threshold for r in past):
            high_given_high_total += 1
            if current_high:
                high_given_high_count += 1
        elif all(r < threshold for r in past):
            high_given_low_total += 1
            if current_high:
                high_given_low_count += 1

    p_high_given_high = (high_given_high_count / high_given_high_total
                         if high_given_high_total > 0 else None)
    p_high_given_low = (high_given_low_count / high_given_low_total
                        if high_given_low_total > 0 else None)

    return {
        "p_high_given_past_high": p_high_given_high,
        "n_past_high": high_given_high_total,
        "p_high_given_past_low": p_high_given_low,
        "n_past_low": high_given_low_total,
    }


def ascii_sparkline(rates, width=60):
    """Generate ASCII sparkline of acceptance rates."""
    if not rates:
        return ""

    # Bucket rates into width bins
    bucket_size = max(1, len(rates) // width)
    buckets = []
    for i in range(0, len(rates), bucket_size):
        chunk = rates[i:i + bucket_size]
        buckets.append(sum(chunk) / len(chunk))

    # Map to ASCII blocks
    blocks = " ▁▂▃▄▅▆▇█"
    result = ""
    for b in buckets[:width]:
        idx = int(b * (len(blocks) - 1))
        result += blocks[idx]
    return result


def print_analysis(slots):
    """Print comprehensive temporal analysis."""
    print("=" * 70)
    print("TEMPORAL PATTERN ANALYSIS - SPECULATIVE DECODING")
    print("=" * 70)

    for slot_id, rounds in sorted(slots.items()):
        rates = [acceptance_rate(r) for r in rounds]
        n = len(rates)

        if n < 3:
            continue

        print(f"\nSLOT {slot_id} ({n} rounds)")
        print("-" * 70)

        # Basic stats
        avg_rate = sum(rates) / n
        min_rate = min(rates)
        max_rate = max(rates)
        print(f"  Avg acceptance: {avg_rate:.1%}  Min: {min_rate:.1%}  Max: {max_rate:.1%}")

        # Sparkline visualization
        print(f"\n  Timeline (each char ≈ {max(1, n // 60)} rounds):")
        print(f"  {ascii_sparkline(rates)}")
        print(f"  {'0%':<29}{'50%':^12}{'100%':>29}")

        # Autocorrelation
        print(f"\n  AUTOCORRELATION (how much does round i predict round i+lag?)")
        print(f"  " + "-" * 50)
        autocorr = analyze_autocorrelation(rates)
        for lag, corr in autocorr.items():
            bar_len = int(abs(corr) * 30)
            bar = "█" * bar_len if corr >= 0 else "░" * bar_len
            sign = "+" if corr >= 0 else "-"
            print(f"    Lag {lag}: {sign}{abs(corr):.3f}  {bar}")

        # Conditional probabilities for different thresholds
        print(f"\n  CONDITIONAL PROBABILITIES")
        print(f"  " + "-" * 50)
        for threshold in [0.5, 0.7, 0.8]:
            for window in [1, 2, 3]:
                cond = conditional_probability(rates, window=window, threshold=threshold)
                if cond:
                    p_hh = cond["p_high_given_past_high"]
                    n_hh = cond["n_past_high"]
                    p_hl = cond["p_high_given_past_low"]
                    n_hl = cond["n_past_low"]

                    p_hh_str = f"{p_hh:.1%}" if p_hh is not None else "N/A"
                    p_hl_str = f"{p_hl:.1%}" if p_hl is not None else "N/A"

                    print(f"    Threshold {threshold:.0%}, window={window}:")
                    print(f"      P(high | past {window} high): {p_hh_str:>6} (n={n_hh})")
                    print(f"      P(high | past {window} low):  {p_hl_str:>6} (n={n_hl})")

        # Run analysis
        print(f"\n  RUN ANALYSIS (consecutive high/low acceptance)")
        print(f"  " + "-" * 50)
        for threshold in [0.5, 0.7]:
            runs = analyze_runs(rates, threshold=threshold)
            high_runs = [r for r in runs if r["high"]]
            low_runs = [r for r in runs if not r["high"]]

            avg_high = sum(r["length"] for r in high_runs) / len(high_runs) if high_runs else 0
            avg_low = sum(r["length"] for r in low_runs) / len(low_runs) if low_runs else 0
            max_high = max((r["length"] for r in high_runs), default=0)
            max_low = max((r["length"] for r in low_runs), default=0)

            print(f"    Threshold {threshold:.0%}:")
            print(f"      High runs: {len(high_runs):3d} (avg len: {avg_high:.1f}, max: {max_high})")
            print(f"      Low runs:  {len(low_runs):3d} (avg len: {avg_low:.1f}, max: {max_low})")

        # Transition matrix
        print(f"\n  STATE TRANSITIONS (threshold=70%)")
        print(f"  " + "-" * 50)
        transitions = {"HH": 0, "HL": 0, "LH": 0, "LL": 0}
        for i in range(len(rates) - 1):
            curr = "H" if rates[i] >= 0.7 else "L"
            next_ = "H" if rates[i + 1] >= 0.7 else "L"
            transitions[curr + next_] += 1

        total_from_h = transitions["HH"] + transitions["HL"]
        total_from_l = transitions["LH"] + transitions["LL"]

        print(f"    From HIGH: → HIGH {transitions['HH']:3d} ({transitions['HH']/total_from_h:.1%})  → LOW {transitions['HL']:3d} ({transitions['HL']/total_from_h:.1%})" if total_from_h > 0 else "    From HIGH: N/A")
        print(f"    From LOW:  → HIGH {transitions['LH']:3d} ({transitions['LH']/total_from_l:.1%})  → LOW {transitions['LL']:3d} ({transitions['LL']/total_from_l:.1%})" if total_from_l > 0 else "    From LOW:  N/A")

        # Detailed timeline with position markers
        print(f"\n  DETAILED TIMELINE (showing rate at each round)")
        print(f"  " + "-" * 50)

        # Show in chunks of 20
        for start in range(0, n, 20):
            chunk = rates[start:start + 20]
            line = f"  {start:4d}: "
            for i, rate in enumerate(chunk):
                if rate >= 0.8:
                    line += "█"
                elif rate >= 0.6:
                    line += "▓"
                elif rate >= 0.4:
                    line += "▒"
                elif rate >= 0.2:
                    line += "░"
                else:
                    line += " "
            # Add rate values at significant transitions
            print(line)
        print(f"         Legend: █≥80% ▓≥60% ▒≥40% ░≥20% (space)<20%")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <spec_log.jsonl>")
        sys.exit(1)

    slots = load_rounds(sys.argv[1])

    if not slots:
        print("No data found in log file")
        sys.exit(1)

    print_analysis(slots)


if __name__ == "__main__":
    main()
