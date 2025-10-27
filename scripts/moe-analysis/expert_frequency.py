#!/usr/bin/env python3
"""
Analyze MoE expert selection frequency per layer.

Usage:
    python expert_frequency.py moe_expert_selection.log

Output:
    For each layer, shows frequency of each expert selection.
"""

import sys
import pandas as pd
from collections import Counter
import argparse


def analyze_expert_frequency(log_file):
    """Analyze expert selection frequency per layer."""

    # Read CSV log file
    df = pd.read_csv(log_file)

    # Parse expert_ids from space-separated string to list of integers
    df['expert_list'] = df['expert_ids'].str.split().apply(lambda x: [int(e) for e in x])

    # Group by layer
    layers = sorted(df['layer_id'].unique())

    print(f"Analyzing {len(df)} expert selections across {len(layers)} layers\n")
    print("=" * 80)

    for layer in layers:
        layer_data = df[df['layer_id'] == layer]

        # Flatten all expert selections for this layer
        all_experts = [expert for experts in layer_data['expert_list'] for expert in experts]

        # Count frequencies
        expert_counts = Counter(all_experts)
        total_selections = len(all_experts)

        print(f"\nLayer {layer}:")
        print(f"  Total expert selections: {total_selections}")
        print(f"  Unique experts used: {len(expert_counts)}")
        print(f"  Top 10 most frequent experts:")

        for expert_id, count in expert_counts.most_common(10):
            percentage = (count / total_selections) * 100
            print(f"    Expert {expert_id:3d}: {count:5d} selections ({percentage:5.2f}%)")

        # Show least used experts (bottom 5)
        print(f"  Bottom 5 least frequent experts:")
        for expert_id, count in expert_counts.most_common()[-5:]:
            percentage = (count / total_selections) * 100
            print(f"    Expert {expert_id:3d}: {count:5d} selections ({percentage:5.2f}%)")

    print("\n" + "=" * 80)

    # Overall statistics across all layers
    all_experts_global = [expert for experts in df['expert_list'] for expert in experts]
    expert_counts_global = Counter(all_experts_global)

    print("\nGlobal Statistics (all layers combined):")
    print(f"  Total expert selections: {len(all_experts_global)}")
    print(f"  Unique experts used: {len(expert_counts_global)}")
    print(f"  Top 10 most frequent experts across all layers:")

    for expert_id, count in expert_counts_global.most_common(10):
        percentage = (count / len(all_experts_global)) * 100
        print(f"    Expert {expert_id:3d}: {count:5d} selections ({percentage:5.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze MoE expert selection frequency per layer'
    )
    parser.add_argument(
        'log_file',
        help='Path to moe_expert_selection.log file'
    )

    args = parser.parse_args()

    try:
        analyze_expert_frequency(args.log_file)
    except FileNotFoundError:
        print(f"Error: File '{args.log_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
