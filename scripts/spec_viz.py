#!/usr/bin/env python3
"""
Visualize speculation cycles in human-readable format.

Each line shows one round:
- Number of accepted tokens
- Probability of each drafted token (color/symbol coded)
- Special marker for last accepted position
"""

import json
import sys


def p_to_char(p):
    """Map probability to a character."""
    if p >= 0.99:
        return "█"
    elif p >= 0.95:
        return "▓"
    elif p >= 0.90:
        return "▒"
    elif p >= 0.80:
        return "░"
    elif p >= 0.60:
        return "○"
    elif p >= 0.40:
        return "◦"
    elif p >= 0.20:
        return "·"
    else:
        return " "


def entropy_to_char(h):
    """Map entropy to a character."""
    if h <= 0.1:
        return "█"
    elif h <= 0.3:
        return "▓"
    elif h <= 0.5:
        return "▒"
    elif h <= 0.8:
        return "░"
    elif h <= 1.0:
        return "○"
    elif h <= 1.5:
        return "◦"
    elif h <= 2.0:
        return "·"
    else:
        return " "


def visualize_round(r, show_entropy=False):
    """Generate visualization for a single round."""
    tokens = r["draft"]["tokens"]
    n_accepted = r["verify"]["n_accepted"]
    n_drafted = r["draft"]["n"]

    # Get timing data if available
    timing = r.get("timing", {})
    t_draft_us = timing.get("t_draft_us", 0)
    t_verify_us = timing.get("t_verify_us", 0)

    # Build the probability/entropy line
    line = ""
    for i, t in enumerate(tokens):
        if show_entropy:
            char = entropy_to_char(t["entropy"])
        else:
            char = p_to_char(t["p"])

        # Mark the rejection point
        if i == n_accepted - 1 and n_accepted < n_drafted:
            # Last accepted token (next one was rejected)
            line += f"[{char}]"
        elif i == n_accepted and n_accepted < n_drafted:
            # First rejected token
            line += f"✗{char}"
        elif i == n_drafted - 1 and n_accepted == n_drafted:
            # All accepted - mark last
            line += f"[{char}]"
        else:
            line += f" {char}"

    return n_accepted, n_drafted, line, t_draft_us, t_verify_us


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <spec_log.jsonl> [--entropy]")
        sys.exit(1)

    show_entropy = "--entropy" in sys.argv
    path = sys.argv[1]

    rounds = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rounds.append(json.loads(line))

    if not rounds:
        print("No data found")
        sys.exit(1)

    # Group by slot
    slots = {}
    for r in rounds:
        slot_id = r["slot_id"]
        if slot_id not in slots:
            slots[slot_id] = []
        slots[slot_id].append(r)

    for slot_id in slots:
        slots[slot_id].sort(key=lambda x: x["round_id"])

    mode = "Entropy (H)" if show_entropy else "Probability (p)"
    print(f"SPECULATION VISUALIZATION - {mode}")
    print("=" * 80)
    if show_entropy:
        print("Legend: █≤0.1 ▓≤0.3 ▒≤0.5 ░≤0.8 ○≤1.0 ◦≤1.5 ·≤2.0 (space)>2.0")
    else:
        print("Legend: █≥99% ▓≥95% ▒≥90% ░≥80% ○≥60% ◦≥40% ·≥20% (space)<20%")
    print("[x] = last accepted   ✗x = first rejected (if any)")
    print("=" * 80)

    # Check if timing data is available
    has_timing = any("timing" in r for r in rounds)

    for slot_id, slot_rounds in sorted(slots.items()):
        print(f"\nSlot {slot_id} ({len(slot_rounds)} rounds)")
        print("-" * 120 if has_timing else "-" * 80)
        if has_timing:
            print(f"{'Rnd':>4} {'Acc':>3}/{'Dft':<3} {'Draft':>7} {'Verify':>7} {'Vis':<}")
        else:
            print(f"{'Rnd':>4} {'Acc':>3}/{'Dft':<3} {'Vis':<}")
        print("-" * 120 if has_timing else "-" * 80)

        total_t_draft = 0
        total_t_verify = 0

        for r in slot_rounds:
            n_accepted, n_drafted, vis, t_draft_us, t_verify_us = visualize_round(r, show_entropy)
            round_id = r["round_id"]
            total_t_draft += t_draft_us
            total_t_verify += t_verify_us

            if has_timing:
                t_draft_ms = t_draft_us / 1000
                t_verify_ms = t_verify_us / 1000
                print(f"{round_id:>4} {n_accepted:>3}/{n_drafted:<3} {t_draft_ms:>6.1f}ms {t_verify_ms:>6.1f}ms {vis}")
            else:
                print(f"{round_id:>4} {n_accepted:>3}/{n_drafted:<3} {vis}")

        # Summary stats
        total_drafted = sum(r["draft"]["n"] for r in slot_rounds)
        total_accepted = sum(r["verify"]["n_accepted"] for r in slot_rounds)
        rate = total_accepted / total_drafted if total_drafted > 0 else 0
        print("-" * 120 if has_timing else "-" * 80)
        if has_timing and total_t_draft > 0:
            total_t_draft_ms = total_t_draft / 1000
            total_t_verify_ms = total_t_verify / 1000
            avg_t_draft_ms = total_t_draft_ms / len(slot_rounds)
            avg_t_verify_ms = total_t_verify_ms / len(slot_rounds)
            print(f"     {total_accepted:>3}/{total_drafted:<4} ({rate:.1%} acceptance)  "
                  f"Avg: {avg_t_draft_ms:.1f}ms draft, {avg_t_verify_ms:.1f}ms verify")
        else:
            print(f"     {total_accepted:>3}/{total_drafted:<4} ({rate:.1%} acceptance)")

    # Position-wise acceptance histogram
    print("\n" + "=" * 80)
    print("POSITION-WISE ACCEPTANCE RATE")
    print("=" * 80)

    max_pos = max(len(r["draft"]["tokens"]) for r in rounds)
    pos_stats = {i: {"count": 0, "accepted": 0} for i in range(max_pos)}

    for r in rounds:
        n_accepted = r["verify"]["n_accepted"]
        for i, t in enumerate(r["draft"]["tokens"]):
            pos_stats[i]["count"] += 1
            if i < n_accepted:
                pos_stats[i]["accepted"] += 1

    print(f"{'Pos':>3} {'Rate':>6} {'Bar':<50}")
    for pos in range(max_pos):
        if pos_stats[pos]["count"] > 0:
            rate = pos_stats[pos]["accepted"] / pos_stats[pos]["count"]
            bar_len = int(rate * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"{pos:>3} {rate:>5.0%}  {bar}")


if __name__ == "__main__":
    main()
